import argparse
from pathlib import Path

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_RED
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.gp import gp_Pnt
from OCC.Display.SimpleGui import init_display

from hole_candidate_runtime import (
    DEFAULT_DETECTOR,
    DEFAULT_INPUT_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    export_candidate_csv,
    run_inference,
)


def visualize(step_path, predicted_df, sphere_scale, score_threshold):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, update=False)

    positives = predicted_df[
        (predicted_df["pred_label"] == 1) & (predicted_df["pred_score"] >= score_threshold)
    ].copy()
    print(f"Visualizing positives: {len(positives)} / total candidates: {len(predicted_df)}")

    for _, row in positives.iterrows():
        radius = max(float(row["estimatedHoleRadius"]) * sphere_scale, 0.5)
        center = gp_Pnt(float(row["wireCenterX"]), float(row["wireCenterY"]), float(row["wireCenterZ"]))
        marker = BRepPrimAPI_MakeSphere(center, radius).Shape()
        display.DisplayShape(marker, color=Quantity_NOC_RED, update=False)

    display.FitAll()
    start_display()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predicted hole candidates on a STEP model.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--sphere-scale", type=float, default=0.35)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_export:
        export_candidate_csv(args.step_model, args.detector, args.csv)

    predicted = run_inference(args.csv, args.model, args.stats, args.hidden_dim)
    if predicted.empty:
        print("No hole candidates found.")
        return

    required_cols = {"wireCenterX", "wireCenterY", "wireCenterZ"}
    if not required_cols.issubset(set(predicted.columns)):
        missing = sorted(required_cols - set(predicted.columns))
        raise ValueError(f"CSV is missing required center columns: {missing}")

    print("Prediction distribution:")
    print(predicted["pred_label"].value_counts().sort_index().to_string())
    visualize(args.step_model, predicted, args.sphere_scale, args.score_threshold)


if __name__ == "__main__":
    main()
