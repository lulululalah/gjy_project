import argparse
import re
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "plane_model"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Debug" / "Detector.exe"

INJECTED_COUNT_RE = re.compile(r">>> Injected rivet count:\s*(\d+)")


@dataclass
class ScreeningResult:
    input_path: Path
    return_code: int
    stdout: str
    stderr: str
    injected_count: int = 0
    deleted_paths: list[Path] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.return_code == 0 and self.injected_count > 0


def is_step_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".step", ".stp"}


def output_paths_for(input_path: Path, model_dir: Path) -> list[Path]:
    stem = input_path.stem
    return [
        model_dir / "step" / f"{stem}_wing_rivets.step",
        model_dir / "step" / f"{stem}_rivet_only.step",
        model_dir / "label" / f"{stem}_wing_rivets.labels.json",
    ]


def success_output_paths_for(input_path: Path, success_dir: Path) -> list[Path]:
    stem = input_path.stem
    return [
        success_dir / input_path.name,
        success_dir / "step" / f"{stem}_wing_rivets.step",
        success_dir / "step" / f"{stem}_rivet_only.step",
        success_dir / "label" / f"{stem}_wing_rivets.labels.json",
    ]


def failure_output_paths_for(input_path: Path, failure_dir: Path) -> list[Path]:
    stem = input_path.stem
    return [
        failure_dir / input_path.name,
        failure_dir / "step" / f"{stem}_wing_rivets.step",
        failure_dir / "step" / f"{stem}_rivet_only.step",
        failure_dir / "label" / f"{stem}_wing_rivets.labels.json",
    ]


def sanitize_ascii_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9._ -]+", "_", ascii_text)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized or "model"


def rename_step_files_to_ascii(model_dir: Path) -> list[tuple[Path, Path]]:
    renamed: list[tuple[Path, Path]] = []
    for path in sorted(model_dir.iterdir()):
        if not is_step_file(path):
            continue

        sanitized_stem = sanitize_ascii_filename(path.stem)
        sanitized_suffix = sanitize_ascii_filename(path.suffix)
        candidate_name = f"{sanitized_stem}{sanitized_suffix}"
        if candidate_name == path.name:
            continue

        destination = model_dir / candidate_name
        counter = 1
        while destination.exists():
            destination = model_dir / f"{sanitized_stem}_{counter}{sanitized_suffix}"
            counter += 1

        path.rename(destination)
        renamed.append((path, destination))

    return renamed


def parse_injected_count(stdout: str) -> int:
    match = INJECTED_COUNT_RE.search(stdout)
    return int(match.group(1)) if match else 0


def run_detector(detector: Path, input_path: Path, timeout_seconds: int) -> ScreeningResult:
    try:
        completed = subprocess.run(
            [str(detector), "--inject-wing-rivets", str(input_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = f"{stderr}\nTimed out after {timeout_seconds} seconds.".strip()
        return ScreeningResult(
            input_path=input_path,
            return_code=-9,
            stdout=stdout,
            stderr=stderr,
            injected_count=parse_injected_count(stdout),
        )

    return ScreeningResult(
        input_path=input_path,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        injected_count=parse_injected_count(completed.stdout),
    )


def delete_failed_files(result: ScreeningResult, model_dir: Path) -> None:
    candidates = [result.input_path, *output_paths_for(result.input_path, model_dir)]
    for path in candidates:
        if path.exists() and path.is_file():
            path.unlink()
            result.deleted_paths.append(path)


def move_failed_files(result: ScreeningResult, model_dir: Path, failure_dir: Path) -> list[Path]:
    source_paths = [result.input_path, *output_paths_for(result.input_path, model_dir)]
    destination_paths = failure_output_paths_for(result.input_path, failure_dir)
    moved_paths: list[Path] = []

    for source_path, destination_path in zip(source_paths, destination_paths):
        if not source_path.exists() or not source_path.is_file():
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
        moved_paths.append(destination_path)

    return moved_paths


def move_success_files(result: ScreeningResult, model_dir: Path, success_dir: Path) -> list[Path]:
    source_paths = [result.input_path, *output_paths_for(result.input_path, model_dir)]
    destination_paths = success_output_paths_for(result.input_path, success_dir)
    moved_paths: list[Path] = []

    for source_path, destination_path in zip(source_paths, destination_paths):
        if not source_path.exists() or not source_path.is_file():
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
        moved_paths.append(destination_path)

    return moved_paths


def first_failure_line(result: ScreeningResult) -> str:
    interesting = [
        "Failed to",
        "No rivets",
        "No valid",
        "BOPAlgo_Alert",
        "became invalid",
        "does not exist",
    ]
    for line in result.stdout.splitlines():
        if any(token in line for token in interesting):
            return line.replace(">>>", "").strip()
    if "Timed out after" in result.stderr:
        return result.stderr.splitlines()[-1]
    return f"Detector return code {result.return_code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wing-rivet injection on all STEP/STP files and optionally archive successful models."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--model-timeout", type=int, default=180, help="Maximum seconds allowed for each model.")
    parser.add_argument("--delete-failed", action="store_true", help="Actually delete failed input models and outputs.")
    parser.add_argument(
        "--move-success",
        action="store_true",
        help="Move successful input models and generated outputs into <model-dir>/success.",
    )
    parser.add_argument(
        "--move-failed",
        action="store_true",
        help="Move failed input models and generated outputs into <model-dir>/defete.",
    )
    parser.add_argument(
        "--rename-ascii",
        action="store_true",
        help="Rename STEP/STP filenames in <model-dir> to ASCII before screening.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    detector = args.detector.resolve()
    success_dir = model_dir / "success"
    failure_dir = model_dir / "defete"

    if not detector.exists():
        raise FileNotFoundError(f"Detector executable not found: {detector}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.rename_ascii:
        renamed_files = rename_step_files_to_ascii(model_dir)
        print(f"ASCII rename count: {len(renamed_files)}")
        for source_path, destination_path in renamed_files:
            print(f"  RENAMED: {source_path.name} -> {destination_path.name}")

    input_files = sorted(path for path in model_dir.iterdir() if is_step_file(path))
    print(f"Found {len(input_files)} STEP/STP files in {model_dir}")

    results: list[ScreeningResult] = []
    for index, input_path in enumerate(input_files, start=1):
        print(f"[{index}/{len(input_files)}] Running: {input_path.name}", flush=True)
        result = run_detector(detector, input_path, args.model_timeout)
        results.append(result)

        if result.success:
            print(f"  OK: {result.injected_count} rivets injected", flush=True)
            if args.move_success:
                moved_paths = move_success_files(result, model_dir, success_dir)
                print(f"  Archived {len(moved_paths)} file(s) into {success_dir}", flush=True)
        else:
            print(f"  FAIL: {first_failure_line(result)}", flush=True)
            if args.move_failed:
                moved_paths = move_failed_files(result, model_dir, failure_dir)
                print(f"  Moved {len(moved_paths)} file(s) into {failure_dir}", flush=True)
            elif args.delete_failed:
                delete_failed_files(result, model_dir)
                print(f"  Deleted {len(result.deleted_paths)} file(s)", flush=True)

    success_count = sum(1 for result in results if result.success)
    failure_count = len(results) - success_count
    print(f"Success: {success_count}, Failure: {failure_count}")
    if args.move_success:
        print(f"Successful models were archived under: {success_dir}")
    if args.move_failed:
        print(f"Failed models were archived under: {failure_dir}")
    if failure_count and not args.delete_failed and not args.move_failed:
        print("Failed files were not deleted. Re-run with --delete-failed to delete them.")
    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
