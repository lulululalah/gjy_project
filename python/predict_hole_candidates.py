import argparse
from pathlib import Path

from hole_candidate_runtime import (
    DEFAULT_DETECTOR,
    DEFAULT_INPUT_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    export_candidate_csv,
    run_inference,
)


DISPLAY_COLS = [
    "candidate_id",
    "host_face_id",
    "wireLength",
    "estimatedHoleRadius",
    "wireCenterX",
    "wireCenterY",
    "wireCenterZ",
    "hostFaceArea",
    "hostInnerWireCount",
    "adjacentSmallCylinderCount",
    "concaveEdgeRatio",
    "pred_score",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict small-hole candidates for a STEP model.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-negative-k", type=int, default=10)
    return parser.parse_args()


def print_candidate_table(title, df, limit):
    print(f"\n{title} (top {min(limit, len(df))}):")
    if df.empty:
        print("None")
        return
    print(df[DISPLAY_COLS].head(limit).to_string(index=False))


def main():
    args = parse_args()

    if not args.skip_export:
        export_candidate_csv(args.step_model, args.detector, args.csv)

    result = run_inference(args.csv, args.model, args.stats, args.hidden_dim)
    if result.empty:
        print("No hole candidates found.")
        return

    print(f"Candidates: {len(result)}")
    print("Prediction distribution:")
    print(result["pred_label"].value_counts().sort_index().to_string())

    positive = result[result["pred_label"] == 1].sort_values("pred_score", ascending=False)
    negative = result[result["pred_label"] == 0].sort_values("pred_score", ascending=False)

    print_candidate_table("Top positive candidates", positive, args.top_k)
    print_candidate_table("Top near-miss negative candidates", negative, args.top_negative_k)


if __name__ == "__main__":
    main()
