"""Create an explicit aircraft-level train/validation split for CAD face CSVs."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_VAL_MODELS = [
    "95- 737-800-4 CREO STP_boeing_737_cj_wing_rivets.step",
    "68- KC46 pegasus STP STL_Boeing KC-46 Peguses v4_wing_rivets.step",
    "221-cessna-model-grand-caravan STP_cessn CARAVAN v13_wing_rivets_decals.step",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a training CSV by whole aircraft model names."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/wing_rivet_training_set.csv"),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/plane_model/plane_model_face_train_13.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/plane_model/plane_model_face_val_3.csv"),
    )
    parser.add_argument(
        "--val-model",
        action="append",
        dest="val_models",
        help="Exact model_name to place in validation; repeat for every validation model.",
    )
    return parser.parse_args()


def label_counts(frame):
    return {label: int((frame["label"] == label).sum()) for label in range(3)}


def main():
    args = parse_args()
    val_models = args.val_models or DEFAULT_VAL_MODELS
    if not val_models:
        raise ValueError("Provide at least one validation model.")

    df = pd.read_csv(args.input_csv)
    required_columns = {"graph_id", "model_name", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")

    all_models = set(df["model_name"].unique())
    unknown_models = sorted(set(val_models).difference(all_models))
    if unknown_models:
        raise ValueError(f"Validation models are not in input CSV: {unknown_models}")

    val_df = df[df["model_name"].isin(val_models)].copy()
    train_df = df[~df["model_name"].isin(val_models)].copy()
    train_models = set(train_df["model_name"].unique())
    actual_val_models = set(val_df["model_name"].unique())

    if len(train_models) + len(actual_val_models) != len(all_models):
        raise ValueError("Train and validation model counts do not reconstruct the source models.")
    if train_models.intersection(actual_val_models):
        raise ValueError("Train and validation model sets overlap.")
    if set(train_df["graph_id"]).intersection(set(val_df["graph_id"])):
        raise ValueError("Train and validation graph IDs overlap.")
    if len(train_df) + len(val_df) != len(df):
        raise ValueError("Split rows do not reconstruct the source CSV.")
    if not set(df["label"].astype(int).unique()).issubset({0, 1, 2}):
        raise ValueError("Input CSV must use background=0, rivet=1, surface_feature=2.")
    if any(label_counts(val_df)[label] == 0 for label in range(3)):
        raise ValueError("Validation split must contain background, rivet, and surface_feature labels.")

    args.train_csv.parent.mkdir(parents=True, exist_ok=True)
    args.val_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)

    print(f"source: {args.input_csv} ({len(df)} faces, {len(all_models)} models)")
    print(f"train:  {args.train_csv} ({len(train_df)} faces, {len(train_models)} models, labels={label_counts(train_df)})")
    print(f"val:    {args.val_csv} ({len(val_df)} faces, {len(actual_val_models)} models, labels={label_counts(val_df)})")
    print("validation models:")
    for model_name in val_models:
        print(f"- {model_name}")


if __name__ == "__main__":
    main()
