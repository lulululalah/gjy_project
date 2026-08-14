"""Create explicit model-isolated train, validation, and test CSV splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a CAD face CSV by complete model name into three sets.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--val-model", action="append", default=[], help="Exact validation model_name; repeat.")
    parser.add_argument("--test-model", action="append", default=[], help="Exact test model_name; repeat.")
    return parser.parse_args()


def label_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {label: int((frame["label"] == label).sum()) for label in range(3)}


def validate_split(name: str, frame: pd.DataFrame, source_models: set[str]) -> None:
    models = set(frame["model_name"].astype(str).unique())
    if not models:
        raise ValueError(f"{name} split is empty")
    if not models.issubset(source_models):
        raise ValueError(f"{name} contains unknown models")
    counts = label_counts(frame)
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"{name} is missing required labels: {missing}")


def main() -> int:
    args = parse_args()
    if not args.val_model or not args.test_model:
        raise ValueError("Provide at least one --val-model and one --test-model")

    df = pd.read_csv(args.input_csv)
    required = {"graph_id", "model_name", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if not set(df["label"].astype(int).unique()).issubset({0, 1, 2}):
        raise ValueError("Input CSV must use background=0, rivet=1, surface_feature=2")

    all_models = set(df["model_name"].astype(str).unique())
    val_models = set(args.val_model)
    test_models = set(args.test_model)
    if val_models.intersection(test_models):
        raise ValueError(f"Validation/test overlap: {sorted(val_models.intersection(test_models))}")
    unknown = sorted((val_models | test_models).difference(all_models))
    if unknown:
        raise ValueError(f"Requested models absent from input CSV: {unknown}")

    val = df[df["model_name"].isin(val_models)].copy()
    test = df[df["model_name"].isin(test_models)].copy()
    train = df[~df["model_name"].isin(val_models | test_models)].copy()
    split_models = [set(frame["model_name"].astype(str).unique()) for frame in (train, val, test)]
    if any(split_models[i].intersection(split_models[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Model leakage across splits")
    if set().union(*split_models) != all_models:
        raise ValueError("Splits do not reconstruct source model membership")
    if len(train) + len(val) + len(test) != len(df):
        raise ValueError("Splits do not reconstruct source row count")
    if any(set(train["graph_id"]).intersection(other["graph_id"]) for other in (val, test)) or set(val["graph_id"]).intersection(test["graph_id"]):
        raise ValueError("Graph ID leakage across splits")

    for name, frame in (("train", train), ("val", val), ("test", test)):
        validate_split(name, frame, all_models)
    for output, frame in ((args.train_csv, train), (args.val_csv, val), (args.test_csv, test)):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing split: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)

    print(f"source: models={len(all_models)}, faces={len(df)}, labels={label_counts(df)}")
    for name, frame in (("train", train), ("val", val), ("test", test)):
        print(f"{name}: models={frame['model_name'].nunique()}, faces={len(frame)}, labels={label_counts(frame)}")
        for model_name in sorted(frame["model_name"].astype(str).unique()):
            print(f"  - {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
