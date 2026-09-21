"""Compute face-level metrics and per-aircraft false-positive risk from a frozen snapshot."""

import argparse
import csv
import json
from collections import Counter
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
        raise ValueError("Truth and prediction model sets do not match.")

    confusion = Counter()
    risk_rows = []
    for name, truth_path in sorted(truth_files.items()):
        truth_rows = read_rows(truth_path)
        prediction_rows = read_rows(prediction_files[name])
        truth_by_id = {row["id"]: row for row in truth_rows}
        prediction_by_id = {row["face_id"]: row for row in prediction_rows}
        if (
            len(truth_rows) != len(truth_by_id)
            or len(prediction_rows) != len(prediction_by_id)
            or set(truth_by_id) != set(prediction_by_id)
        ):
            raise ValueError(f"Face-ID alignment failed for {name}")

        per_aircraft = Counter(
            (int(truth_by_id[face_id]["label"]), int(prediction_by_id[face_id]["pred_label"]))
            for face_id in truth_by_id
        )
        confusion.update(per_aircraft)
        background_to_rivet = per_aircraft[(0, 1)]
        background_to_surface = per_aircraft[(0, 2)]
        false_positive_areas = [
            float(truth_by_id[face_id]["relativeArea"])
            for face_id in truth_by_id
            if int(truth_by_id[face_id]["label"]) == 0
            and int(prediction_by_id[face_id]["pred_label"]) in (1, 2)
        ]
        correct = sum(count for (truth, pred), count in per_aircraft.items() if truth == pred)
        risk_rows.append(
            {
                "model": name,
                "faces": len(truth_rows),
                "accuracy": correct / len(truth_rows),
                "background_to_rivet": background_to_rivet,
                "background_to_surface_feature": background_to_surface,
                "total_background_false_positives": background_to_rivet + background_to_surface,
                "max_background_false_positive_relative_area": max(false_positive_areas, default=0.0),
            }
        )

    labels = [(0, "Background"), (1, "Rivet"), (2, "Surface-feature")]
    metric_rows = []
    for label, name in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(count for (truth, pred), count in confusion.items() if pred == label and truth != label)
        false_negative = sum(count for (truth, pred), count in confusion.items() if truth == label and pred != label)
        support = sum(count for (truth, _), count in confusion.items() if truth == label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = true_positive / (true_positive + false_positive + false_negative)
        metric_rows.append(
            {"class": name, "support": support, "precision": precision, "recall": recall, "f1": f1, "iou": iou}
        )

    total_faces = sum(confusion.values())
    accuracy = sum(count for (truth, pred), count in confusion.items() if truth == pred) / total_faces
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "face_level_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    with (args.output_dir / "per_aircraft_risk.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=risk_rows[0].keys())
        writer.writeheader()
        writer.writerows(risk_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"total_faces": total_faces, "accuracy": accuracy, "confusion": sorted([*key, value] for key, value in confusion.items())},
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
