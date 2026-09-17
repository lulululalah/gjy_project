"""Summarize face-level E1 predictions against the fixed test-label CSV.

Only models whose prediction and truth face counts match are included in the
end-to-end aggregate.  This prevents accidental joins across STEP revisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CLASS_NAMES = {0: "background", 1: "rivet", 2: "surface_feature"}


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def prediction_model_name(path: Path, truth_names: set[str]) -> str | None:
    suffix = ".predictions.csv"
    stem = path.name[: -len(suffix)]
    candidates = (stem, f"{stem}.step", f"{stem}.stp", f"{stem}.STEP")
    return next((name for name in candidates if name in truth_names), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Exact model_name to include; repeat this argument to form a subset.",
    )
    args = parser.parse_args()

    with args.truth_csv.open(newline="", encoding="utf-8-sig") as handle:
        truth_rows = list(csv.DictReader(handle))
    truth_by_model: dict[str, dict[int, dict[str, str]]] = {}
    for row in truth_rows:
        truth_by_model.setdefault(row["model_name"], {})[int(row["id"])] = row
    include_models = set(args.include_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_model: list[dict[str, object]] = []
    valid_pairs: list[tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]] = []
    for prediction_path in sorted(args.prediction_dir.glob("*.predictions.csv")):
        model_name = prediction_model_name(prediction_path, set(truth_by_model))
        if not model_name:
            continue
        if include_models and model_name not in include_models:
            continue
        with prediction_path.open(newline="", encoding="utf-8-sig") as handle:
            predictions = {int(row["face_id"]): row for row in csv.DictReader(handle)}
        truth = truth_by_model[model_name]
        face_ids_match = set(predictions) == set(truth)
        if not face_ids_match:
            per_model.append(
                {
                    "model_name": model_name,
                    "status": "excluded_step_truth_face_mismatch",
                    "truth_faces": len(truth),
                    "prediction_faces": len(predictions),
                }
            )
            continue
        confusion = Counter()
        background_fp = []
        for face_id, truth_row in truth.items():
            true_label = int(truth_row["label"])
            pred_label = int(predictions[face_id]["pred_label"])
            confusion[(true_label, pred_label)] += 1
            if true_label == 0 and pred_label != 0:
                background_fp.append((truth_row, pred_label))
        row: dict[str, object] = {
            "model_name": model_name,
            "status": "included",
            "truth_faces": len(truth),
            "prediction_faces": len(predictions),
            "accuracy": sum(confusion[(k, k)] for k in CLASS_NAMES) / len(truth),
            "background_to_rivet_fp": sum(label == 1 for _, label in background_fp),
            "background_to_surface_feature_fp": sum(label == 2 for _, label in background_fp),
            "background_fp_total": len(background_fp),
            "max_background_fp_relative_area": max((float(item[0]["relativeArea"]) for item in background_fp), default=0.0),
            "max_background_fp_area": max((float(item[0]["area"]) for item in background_fp), default=0.0),
        }
        for class_id, class_name in CLASS_NAMES.items():
            tp = confusion[(class_id, class_id)]
            fp = sum(confusion[(other, class_id)] for other in CLASS_NAMES if other != class_id)
            fn = sum(confusion[(class_id, other)] for other in CLASS_NAMES if other != class_id)
            row[f"{class_name}_support"] = sum(confusion[(class_id, other)] for other in CLASS_NAMES)
            row.update({f"{class_name}_{key}": value for key, value in metrics(tp, fp, fn).items()})
        per_model.append(row)
        valid_pairs.append((truth, predictions))

    fieldnames = sorted({key for row in per_model for key in row})
    with (args.out_dir / "E1_逐飞机检测与背景误检风险.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_model)

    confusion = Counter()
    for truth, predictions in valid_pairs:
        for face_id, truth_row in truth.items():
            confusion[(int(truth_row["label"]), int(predictions[face_id]["pred_label"]))] += 1
    aggregate: dict[str, object] = {
        "included_models": len(valid_pairs),
        "excluded_models": [row for row in per_model if row["status"] != "included"],
        "support": sum(confusion.values()),
        "accuracy": sum(confusion[(k, k)] for k in CLASS_NAMES) / sum(confusion.values()),
        "confusion_matrix_rows_true_columns_pred": [
            [confusion[(true_label, pred_label)] for pred_label in CLASS_NAMES] for true_label in CLASS_NAMES
        ],
    }
    class_metrics = []
    for class_id, class_name in CLASS_NAMES.items():
        tp = confusion[(class_id, class_id)]
        fp = sum(confusion[(other, class_id)] for other in CLASS_NAMES if other != class_id)
        fn = sum(confusion[(class_id, other)] for other in CLASS_NAMES if other != class_id)
        values = metrics(tp, fp, fn)
        values.update({"class_id": class_id, "class_name": class_name, "support": sum(confusion[(class_id, other)] for other in CLASS_NAMES)})
        class_metrics.append(values)
    aggregate["class_metrics"] = class_metrics
    aggregate["macro_f1"] = sum(item["f1"] for item in class_metrics) / len(class_metrics)
    aggregate["foreground_macro_f1"] = sum(item["f1"] for item in class_metrics[1:]) / 2
    with (args.out_dir / "E1_检测指标汇总.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
