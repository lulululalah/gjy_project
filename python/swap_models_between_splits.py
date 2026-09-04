"""Move complete aircraft models between train and test CSVs.

The operation is model-level: all rows belonging to each selected model are
removed from both inputs and then added to the requested destination split.
"""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--move-to-train", required=True)
    parser.add_argument("--move-to-test", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    args = parser.parse_args()

    if args.move_to_train == args.move_to_test:
        raise ValueError("The two moved models must be different.")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    for frame_name, frame in (("train", train), ("test", test)):
        if "model_name" not in frame.columns:
            raise ValueError(f"{frame_name} CSV has no model_name column")

    all_rows = pd.concat([train, test], ignore_index=True)
    moved_train = all_rows[all_rows["model_name"] == args.move_to_train].copy()
    moved_test = all_rows[all_rows["model_name"] == args.move_to_test].copy()
    if moved_train.empty:
        raise ValueError(f"Model not found in either input CSV: {args.move_to_train}")
    if moved_test.empty:
        raise ValueError(f"Model not found in either input CSV: {args.move_to_test}")

    # Remove both models from both splits before placing them in their target.
    excluded = {args.move_to_train, args.move_to_test}
    train_base = train[~train["model_name"].isin(excluded)]
    test_base = test[~test["model_name"].isin(excluded)]
    train_out = pd.concat([train_base, moved_train], ignore_index=True)
    test_out = pd.concat([test_base, moved_test], ignore_index=True)

    train_models = set(train_out["model_name"])
    test_models = set(test_out["model_name"])
    overlap = train_models & test_models
    if overlap:
        raise ValueError(f"Model leakage remains after swap: {sorted(overlap)}")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(args.train_output, index=False, encoding="utf-8-sig")
    test_out.to_csv(args.test_output, index=False, encoding="utf-8-sig")

    print(f"moved_to_train={args.move_to_train}")
    print(f"moved_to_train_rows={len(moved_train)}")
    print(f"moved_to_test={args.move_to_test}")
    print(f"moved_to_test_rows={len(moved_test)}")
    print(f"train_rows={len(train_out)} test_rows={len(test_out)}")
    print(f"train_models={len(train_models)} test_models={len(test_models)}")
    print(f"train_output={args.train_output}")
    print(f"test_output={args.test_output}")


if __name__ == "__main__":
    main()
