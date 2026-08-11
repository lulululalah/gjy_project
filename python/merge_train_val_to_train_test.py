"""Merge explicit train/validation CSVs into train while preserving a fixed test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def feature_flags(model_name: str, label_dir: Path) -> dict[str, bool]:
    path = label_dir / f"{Path(model_name).stem}.labels.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label JSON for {model_name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    semantics = {str(face["semantic"]) for face in payload["faces"]}
    added_decal = any(str(item.get("type")) == "decal" for item in payload.get("instances", []))
    return {
        "native_window": "window" in semantics,
        "native_decal": "decal" in semantics and not added_decal,
        "added_decal": added_decal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a train/test-only split from existing explicit CSVs.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def validate_split(name: str, frame: pd.DataFrame, label_dir: Path) -> list[dict[str, object]]:
    records = []
    for model_name in sorted(frame["model_name"].astype(str).unique()):
        flags = feature_flags(model_name, label_dir)
        records.append({"split": name, "model_name": model_name, **flags})
    coverage = {key: any(record[key] for record in records) for key in records[0] if key not in {"split", "model_name"}}
    missing = [key for key, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"{name} is missing required feature coverage: {missing}")
    return records


def main() -> None:
    args = parse_args()
    train = pd.concat([pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)], ignore_index=True)
    test = pd.read_csv(args.test_csv)
    if train.duplicated(["model_name", "id"]).any():
        raise ValueError("Merged training inputs overlap on model_name/id.")
    overlap = set(train["model_name"].astype(str)).intersection(test["model_name"].astype(str))
    if overlap:
        raise ValueError(f"Train/test model overlap is not allowed: {sorted(overlap)}")
    train_records = validate_split("train", train, args.label_dir)
    test_records = validate_split("test", test, args.label_dir)
    for path in (args.train_out, args.test_out, args.manifest_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_out, index=False, encoding="utf-8")
    test.to_csv(args.test_out, index=False, encoding="utf-8")
    pd.DataFrame(train_records + test_records).to_csv(args.manifest_out, index=False, encoding="utf-8-sig")
    print(f"train: rows={len(train)}, models={train['model_name'].nunique()}, labels={train['label'].value_counts().sort_index().to_dict()}")
    print(f"test: rows={len(test)}, models={test['model_name'].nunique()}, labels={test['label'].value_counts().sort_index().to_dict()}")
    print("Train/test model overlap: 0")
    print(f"Coverage manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
