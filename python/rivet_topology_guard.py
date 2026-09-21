"""Conservatively suppress structurally implausible rivet predictions.

The guard is deliberately independent of face area.  It only changes a
predicted rivet to background when the face is planar, has at least four
planar neighbours, and has no smooth boundary edge.  These conditions describe
a broad planar patch rather than a local rivet face; all other predictions are
left untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_FEATURE_COLUMNS = {
    "surfaceType",
    "neighborPlaneCount",
    "smoothEdgeCount",
}
REQUIRED_PREDICTION_COLUMNS = {"face_id", "pred_label"}


def suppress_planar_patch_rivet_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """Return guarded predictions and the face IDs changed to background."""
    missing_prediction = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if "face_id" in features.columns:
        feature_face_id = "face_id"
    elif "id" in features.columns:
        feature_face_id = "id"
    else:
        raise ValueError("Feature CSV requires either face_id or id for face-id alignment.")
    missing_feature = REQUIRED_FEATURE_COLUMNS - set(features.columns)
    if missing_prediction:
        raise ValueError(f"Prediction CSV is missing: {sorted(missing_prediction)}")
    if missing_feature:
        raise ValueError(f"Feature CSV is missing: {sorted(missing_feature)}")
    if predictions["face_id"].duplicated().any() or features[feature_face_id].duplicated().any():
        raise ValueError("face-id values must be unique in both CSV files.")

    feature_view = features.loc[:, [
        feature_face_id, "surfaceType", "neighborPlaneCount", "smoothEdgeCount"
    ]].rename(columns={feature_face_id: "face_id"})
    merged = predictions.merge(
        feature_view,
        on="face_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged[["surfaceType", "neighborPlaneCount", "smoothEdgeCount"]].isna().any().any():
        raise ValueError("Prediction and feature CSV face-id values do not align.")

    suppress = (
        (merged["pred_label"].astype(int) == 1)
        & (merged["surfaceType"].astype(int) == 0)
        & (merged["neighborPlaneCount"].astype(int) >= 4)
        & (merged["smoothEdgeCount"].astype(int) == 0)
    )
    result = predictions.copy()
    result.loc[suppress.to_numpy(), "pred_label"] = 0
    if "pred_name" in result.columns:
        result.loc[suppress.to_numpy(), "pred_name"] = "background"
    if "pred_confidence" in result.columns and "prob_background" in result.columns:
        result.loc[suppress.to_numpy(), "pred_confidence"] = result.loc[
            suppress.to_numpy(), "prob_background"
        ]
    return result, result.loc[suppress.to_numpy(), "face_id"].astype(int).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    predictions = pd.read_csv(args.predictions)
    guarded, suppressed_face_ids = suppress_planar_patch_rivet_predictions(
        predictions, features
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    guarded.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Suppressed predicted rivets: {len(suppressed_face_ids)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
