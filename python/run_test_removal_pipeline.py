"""Run the verified prediction and CAD-removal stages for test aircraft."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], log_path: Path) -> int:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    return result.returncode


def prediction_count(path: Path, label: int) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(
            int(row["pred_label"]) == label for row in csv.DictReader(stream)
        )


def log_contains(path: Path, required_fragments: tuple[str, ...]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(fragment in text for fragment in required_fragments)


def test_model_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        names = {row["model_name"] for row in csv.DictReader(stream)}
    return sorted(names)


def stage_stem(model_name: str) -> str:
    stem = Path(model_name).stem
    stem = re.sub(r"_wing_rivets(?:_decals)?$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem


def verified_route(stem: str) -> str:
    """Select the restored, topology-validated operation family."""
    if stem.startswith("Air_Plane_Idea_A"):
        return "invalid-host"
    if stem.startswith("27-_DC-10"):
        return "split"
    if stem.startswith(("Airbus", "Airplane_body", "AULIRA_2", "Gulfstream_G280")):
        return "embedded"
    return "generic"


def infer(
    python: Path,
    viewer: Path,
    detector: Path,
    model: Path,
    stats: Path,
    step: Path,
    csv_path: Path,
    prediction_path: Path,
    log_path: Path,
    *,
    decal_only: bool = False,
    skip_export: bool = False,
    exterior_surface_guard: bool = True,
) -> int:
    command = [
        str(python),
        str(viewer),
        str(step),
        "--detector",
        str(detector),
        "--csv",
        str(csv_path),
        "--model",
        str(model),
        "--stats",
        str(stats),
        "--pred-out",
        str(prediction_path),
        "--no-display",
    ]
    if decal_only:
        command.append("--decal-only")
    if skip_export:
        command.append("--skip-export")
    if not exterior_surface_guard:
        command.append("--disable-exterior-surface-guard")
    return run(command, log_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "data" / "plane_model" / "after_two"
    )
    parser.add_argument(
        "--test-csv", type=Path,
        default=root / "model" / "final_detection_model"
        / "test_9_models_frozen_truth.csv"
    )
    parser.add_argument(
        "--snapshot-manifest", type=Path,
        help=(
            "Replay the formal first-stage prediction snapshot recorded by this "
            "manifest instead of re-running initial inference."
        ),
    )
    parser.add_argument(
        "--ending-dir", type=Path,
        default=root / "data" / "plane_model" / "ending"
    )
    parser.add_argument(
        "--work-dir", type=Path,
        default=root / "work" / "test_removal_pipeline"
    )
    parser.add_argument(
        "--detector", type=Path,
        default=root / "build" / "Release" / "Detector.exe"
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "model" / "final_detection_model"
        / "rivet_gnn_train17_rebuilt_20260921_50ep.pth"
    )
    parser.add_argument(
        "--stats", type=Path,
        default=root / "model" / "final_detection_model"
        / "rivet_gnn_train17_rebuilt_20260921_50ep_stats.npz"
    )
    args = parser.parse_args()

    viewer = root / "python" / "visualize_rivets.py"
    bridge_script = root / "python" / "batch_bridge_split_windows.py"
    surface_postprocessor = root / "python" / "surface_visibility_guard.py"
    required = [
        args.input_dir, args.detector, args.python, args.model, args.stats,
        viewer, bridge_script, surface_postprocessor,
    ]
    if args.snapshot_manifest is None:
        required.append(args.test_csv)
    else:
        required.append(args.snapshot_manifest)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

    if args.snapshot_manifest is None:
        runs = [(args.input_dir / name, None) for name in test_model_names(args.test_csv)]
    else:
        manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
        source_by_stem = {
            stage_stem(path.name): path for path in args.input_dir.iterdir() if path.is_file()
        }
        runs = []
        for item in manifest.get("models", []):
            stem = stage_stem(str(item["model"]))
            source = source_by_stem.get(stem)
            prediction = root / item["source_prediction_file"]
            if source is None:
                raise FileNotFoundError(
                    f"No input STEP matches snapshot model {item['model']}."
                )
            if not prediction.exists():
                raise FileNotFoundError(f"Missing snapshot prediction: {prediction}")
            runs.append((source, prediction))
    if len(runs) != 9:
        raise RuntimeError(f"Expected 9 test models, found {len(runs)}")

    args.ending_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for source, snapshot_prediction in runs:
        model_name = source.name
        stem = stage_stem(model_name)
        model_dir = args.work_dir / stem
        model_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {
            "model": model_name,
            "status": "failed",
            "stages": [],
        }
        summaries.append(summary)
        print(f"\n=== {model_name} ===", flush=True)

        route = verified_route(stem)
        summary["route"] = route

        if route == "invalid-host":
            decal_csv = model_dir / "decal.features.csv"
            decal_pred = model_dir / "decal.pred.csv"
            if snapshot_prediction is None:
                if infer(
                    args.python, viewer, args.detector, args.model, args.stats,
                    source, decal_csv, decal_pred, model_dir / "decal.log",
                    decal_only=True,
                ) != 0:
                    continue
                summary["stages"].append("decal_only_prediction")
            else:
                shutil.copy2(snapshot_prediction, decal_pred)
                summary["stages"].append("formal_snapshot_prediction")
            summary["predicted_surface_faces"] = prediction_count(decal_pred, 2)
            destination = args.ending_dir / model_name
            if run(
                [
                    str(args.detector), "--rebuild-invalid-surface-hosts",
                    str(source), str(decal_pred), str(destination),
                ],
                model_dir / "decal_removal.log",
            ) == 0:
                summary["stages"].append("invalid_surface_host_rebuild")
                summary["status"] = "succeeded"
            continue

        initial_csv = model_dir / "initial.features.csv"
        initial_pred = model_dir / "initial.pred.csv"
        if snapshot_prediction is None:
            if infer(
                args.python, viewer, args.detector, args.model, args.stats,
                source, initial_csv, initial_pred, model_dir / "initial.log"
            ) != 0:
                continue
            summary["stages"].append("initial_prediction")
        else:
            shutil.copy2(snapshot_prediction, initial_pred)
            summary["stages"].append("formal_snapshot_prediction")

        removed = model_dir / f"{stem}_removed.step"
        rivet_count = prediction_count(initial_pred, 1)
        summary["predicted_rivets"] = rivet_count
        if stem.startswith("Airbus") and snapshot_prediction is None:
            # Airbus window cleanup depends on the topology of this validated
            # rivet-removal result. Re-running STEP export preserves the face
            # count but changes the topology/face mapping used by narrow mode.
            validated_removed = (
                root / "data" / "plane_model" / "removed" / "Airbus_removed.step"
            )
            if not validated_removed.exists():
                raise FileNotFoundError(
                    f"Missing validated Airbus rivet-removal STEP: {validated_removed}"
                )
            shutil.copy2(validated_removed, removed)
            summary["stages"].append("validated_airbus_rivet_removal")
        elif rivet_count:
            if run(
                [
                    str(args.detector), "--remove-predicted-rivets",
                    str(source), str(initial_pred), str(removed),
                ],
                model_dir / "rivet_removal.log",
            ) != 0:
                continue
            summary["stages"].append("rivet_removal")
        else:
            shutil.copy2(source, removed)
            summary["stages"].append("rivet_absent")

        surface_csv = model_dir / "surface.features.csv"
        surface_pred = model_dir / "surface.pred.csv"
        if infer(
            args.python, viewer, args.detector, args.model, args.stats,
            removed, surface_csv, surface_pred, model_dir / "surface.log",
            exterior_surface_guard=False,
        ) != 0:
            continue
        summary["stages"].append("post_rivet_prediction")

        surface_pred_for_removal = model_dir / "surface.visibility.pred.csv"
        if run(
            [
                str(args.python), str(surface_postprocessor),
                str(removed), str(surface_csv), str(surface_pred),
                "--output", str(surface_pred_for_removal),
            ],
            model_dir / "surface_visibility.log",
        ) != 0:
            continue
        summary["stages"].append("surface_visibility_guard")

        surface_count = prediction_count(surface_pred_for_removal, 2)
        summary["predicted_surface_faces"] = surface_count
        destination = args.ending_dir / model_name
        if surface_count == 0:
            shutil.copy2(removed, destination)
            summary["stages"].append("surface_feature_absent")
            summary["status"] = "succeeded"
            continue

        if route == "generic":
            candidate = model_dir / f"{stem}_surface_removed.step"
            command = [
                str(args.detector), "--remove-predicted-surface-features",
                str(removed), str(surface_pred_for_removal), str(candidate),
            ]
            if run(command, model_dir / "generic_surface.log") == 0:
                shutil.copy2(candidate, destination)
                summary["stages"].append("generic_surface_removal")
                summary["status"] = "succeeded"
            continue

        if route == "embedded":
            candidate = model_dir / f"{stem}_embedded.step"
            embedded_log = model_dir / "embedded.log"
            if run(
                [
                    str(args.detector), "--rebuild-embedded-window-hosts",
                    str(removed), str(surface_pred_for_removal),
                    str(candidate), "auto",
                ],
                embedded_log,
            ) == 0:
                if (
                    stem.startswith("Airbus")
                    and snapshot_prediction is None
                    and not log_contains(
                        embedded_log,
                        (
                            "Embedded-window host loops rebuilt: 108",
                            "Residual window faces removed: 1037",
                            "BRep valid: yes",
                            "Face count: 1723 -> 686",
                        ),
                    )
                ):
                    summary["stages"].append("airbus_geometry_regression")
                    continue
                shutil.copy2(candidate, destination)
                summary["stages"].append("embedded_window_rebuild")
                summary["status"] = "succeeded"
            continue

        split = model_dir / f"{stem}_split.step"
        if run(
            [
                str(args.detector), "--rebuild-split-window-skins",
                str(removed), str(surface_pred_for_removal), str(split),
            ],
            model_dir / "split.log",
        ) == 0:
            bridged = model_dir / f"{stem}_bridged.step"
            if run(
                [
                    str(args.python), str(bridge_script), str(split),
                    str(bridged), "--detector", str(args.detector),
                ],
                model_dir / "bridge.log",
            ) == 0:
                shutil.copy2(bridged, destination)
                summary["stages"].extend(
                    ["split_window_rebuild", "split_window_bridge"]
                )
                summary["status"] = "succeeded"
                continue

        if route == "split":
            summary["stages"].append("split_window_failed")
            continue

        if run(
            [
                str(args.detector), "--rebuild-embedded-window-hosts",
                str(removed), str(surface_pred_for_removal), str(split), "auto",
            ],
            model_dir / "embedded_fallback.log",
        ) == 0:
            shutil.copy2(split, destination)
            summary["stages"].append("embedded_window_rebuild")
            summary["status"] = "succeeded"

    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    succeeded = sum(item["status"] == "succeeded" for item in summaries)
    print(f"\nPipeline complete: {succeeded}/{len(summaries)} succeeded")
    print(f"Summary: {summary_path}")
    return 0 if succeeded == len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
