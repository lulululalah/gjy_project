"""Apply the area-free surface visibility guard to a prediction snapshot."""

import argparse
import json
from pathlib import Path

from surface_visibility_guard import postprocess_prediction_csv


def find_step_path(after_two_dir, prediction_stem):
    candidates = [
        prediction_stem,
        prediction_stem.replace("_decals", ""),
    ]
    steps = [path for path in Path(after_two_dir).iterdir() if path.is_file()]
    matches = [
        path
        for path in steps
        if path.stem in candidates and path.suffix.lower() in {".step", ".stp"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one STEP model for {prediction_stem!r}; found {matches}."
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Create an area-free surface-visibility diagnostic prediction set."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("after_two_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    diagnostics_by_model = {}
    for item in manifest["models"]:
        source_prediction = Path(item["source_prediction_file"])
        prediction_path = args.prediction_dir / source_prediction.name
        if not prediction_path.is_file():
            prediction_path = args.prediction_dir / Path(
                item["output_prediction_file"]
            ).name
        feature_path = source_prediction.with_name(
            source_prediction.name.replace(".pred.csv", ".features.csv")
        )
        step_path = find_step_path(
            args.after_two_dir,
            source_prediction.stem.replace(".pred", ""),
        )
        if not prediction_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(
                f"Missing diagnostic prediction or feature CSV for {item['model']}."
            )
        output_path = args.output_dir / prediction_path.name
        print(f"Processing {item['model']}...", flush=True)
        diagnostics = postprocess_prediction_csv(
            step_path,
            feature_path,
            prediction_path,
            output_path,
        )
        diagnostics_by_model[item["model"]] = diagnostics
        print(f"  completed: {len(diagnostics)} adjustment(s)", flush=True)

    (args.output_dir / "visibility_diagnostics.json").write_text(
        json.dumps(diagnostics_by_model, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
