"""Run downstream geometry from retained STEP intermediates and guarded CSVs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_test_removal_pipeline import verified_route


def run(command: list[str], root: Path, log_path: Path) -> int:
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return result.returncode


def intermediate_step(model_dir: Path) -> Path:
    candidates = [
        path
        for path in model_dir.glob("*_removed.step")
        if not path.name.endswith("_surface_removed.step")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one rivet-removed STEP in {model_dir}.")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retained_work_dir", type=Path)
    parser.add_argument("postprocess_dir", type=Path)
    parser.add_argument("air_plane_idea_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bridge_script = root / "python" / "batch_bridge_split_windows.py"
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}.")
    args.output_dir.mkdir(parents=True)
    summaries = []

    for model_dir in sorted(path for path in args.retained_work_dir.iterdir() if path.is_dir()):
        stage_dir = args.postprocess_dir / model_dir.name
        prediction = stage_dir / "surface.visibility.pred.csv"
        log_dir = stage_dir / "geometry"
        log_dir.mkdir()
        summary = {"model": model_dir.name, "status": "failed", "stages": []}
        summaries.append(summary)
        if model_dir.name == "Air_Plane_Idea_A":
            command = [
                str(args.detector), "--rebuild-invalid-surface-hosts",
                str(args.air_plane_idea_step), str(prediction),
                str(args.output_dir / "Air Plane Idea A.STEP"),
            ]
            if run(command, root, log_dir / "invalid_host.log") == 0:
                summary["stages"].append("invalid_surface_host_rebuild")
                summary["status"] = "succeeded"
            continue

        route = verified_route(model_dir.name)
        removed = intermediate_step(model_dir)
        if route == "generic":
            destination = args.output_dir / (model_dir.name + ".step")
            command = [
                str(args.detector), "--remove-predicted-surface-features",
                str(removed), str(prediction), str(destination),
            ]
            if run(command, root, log_dir / "generic_surface.log") == 0:
                summary["stages"].append("generic_surface_removal")
                summary["status"] = "succeeded"
            continue

        if route == "embedded":
            destination = args.output_dir / (model_dir.name + ".step")
            command = [
                str(args.detector), "--rebuild-embedded-window-hosts",
                str(removed), str(prediction), str(destination), "auto",
            ]
            if run(command, root, log_dir / "embedded.log") == 0:
                summary["stages"].append("embedded_window_rebuild")
                summary["status"] = "succeeded"
            continue

        split = log_dir / "split.step"
        command = [
            str(args.detector), "--rebuild-split-window-skins",
            str(removed), str(prediction), str(split),
        ]
        if run(command, root, log_dir / "split.log") != 0:
            summary["stages"].append("split_window_failed")
            continue
        destination = args.output_dir / (model_dir.name + ".step")
        bridge = [
            str(args.python), str(bridge_script), str(split), str(destination),
            "--detector", str(args.detector),
        ]
        if run(bridge, root, log_dir / "bridge.log") == 0:
            summary["stages"].extend(["split_window_rebuild", "split_window_bridge"])
            summary["status"] = "succeeded"

    (args.postprocess_dir / "geometry_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    succeeded = sum(item["status"] == "succeeded" for item in summaries)
    print(f"Geometry complete: {succeeded}/{len(summaries)} succeeded")
    return 0 if succeeded == len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
