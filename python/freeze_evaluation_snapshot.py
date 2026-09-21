"""Freeze prediction files against the exact face IDs in a test-label snapshot."""

import argparse
import csv
import json
from pathlib import Path


def canonical_name(filename: str) -> str:
    stem = filename
    for suffix in (".csv", ".pred", ".step", ".stp", "_decals"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth_files = {canonical_name(path.name): path for path in args.truth_dir.glob("*.csv")}
    prediction_files = {
        canonical_name(path.name): path for path in args.prediction_dir.glob("*.pred.csv")
    }
    if set(truth_files) != set(prediction_files):
        missing = sorted(set(truth_files) - set(prediction_files))
        extra = sorted(set(prediction_files) - set(truth_files))
        raise ValueError(f"Prediction-file mismatch; missing={missing}, extra={extra}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"truth_dir": str(args.truth_dir), "prediction_dir": str(args.prediction_dir), "models": []}
    for name, truth_path in sorted(truth_files.items()):
        truth_rows = read_rows(truth_path)
        prediction_rows = read_rows(prediction_files[name])
        truth_ids = [row["id"] for row in truth_rows]
        prediction_ids = [row["face_id"] for row in prediction_rows]
        if len(truth_ids) != len(set(truth_ids)) or len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(f"Duplicate face IDs in {name}")
        by_id = {row["face_id"]: row for row in prediction_rows}
        missing_ids = set(truth_ids) - set(by_id)
        if missing_ids:
            raise ValueError(f"Missing prediction IDs in {name}: {len(missing_ids)}")

        output_rows = [by_id[face_id] for face_id in truth_ids]
        output_path = args.output_dir / f"{name}.pred.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
        manifest["models"].append(
            {
                "model": name,
                "truth_file": str(truth_path),
                "source_prediction_file": str(prediction_files[name]),
                "output_prediction_file": str(output_path),
                "truth_faces": len(truth_ids),
                "source_prediction_faces": len(prediction_ids),
                "frozen_prediction_faces": len(output_rows),
                "excluded_source_prediction_faces": len(prediction_ids) - len(output_rows),
            }
        )

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
