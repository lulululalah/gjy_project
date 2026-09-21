"""Apply the unified surface-visibility guard to retained pipeline predictions."""

import argparse
from pathlib import Path

from surface_visibility_guard import postprocess_prediction_csv


def intermediate_step(model_dir: Path) -> Path:
    candidates = [
        path
        for path in model_dir.glob("*_removed.step")
        if not path.name.endswith("_surface_removed.step")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one rivet-removed STEP in {model_dir}; found {candidates}."
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_work_dir", type=Path)
    parser.add_argument("air_plane_idea_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}.")
    args.output_dir.mkdir(parents=True)
    for model_dir in sorted(path for path in args.pipeline_work_dir.iterdir() if path.is_dir()):
        if model_dir.name == "Air_Plane_Idea_A":
            step_path = args.air_plane_idea_step
            feature_path = model_dir / "decal.features.csv"
            prediction_path = model_dir / "decal.pred.csv"
        else:
            step_path = intermediate_step(model_dir)
            feature_path = model_dir / "surface.features.csv"
            prediction_path = model_dir / "surface.pred.csv"
        if not step_path.is_file() or not feature_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing retained input for {model_dir.name}.")
        destination = args.output_dir / model_dir.name
        destination.mkdir()
        print(f"Processing {model_dir.name}...", flush=True)
        diagnostics = postprocess_prediction_csv(
            step_path,
            feature_path,
            prediction_path,
            destination / "surface.visibility.pred.csv",
        )
        (destination / "diagnostic_count.txt").write_text(
            str(len(diagnostics)), encoding="utf-8"
        )
        print(f"  completed: {len(diagnostics)} candidate(s) returned to background", flush=True)


if __name__ == "__main__":
    main()
