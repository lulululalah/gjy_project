from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rivet_topology_guard import suppress_planar_patch_rivet_predictions


def test_guard_suppresses_only_planar_patch_rivets():
    predictions = pd.DataFrame(
        {
            "face_id": [1, 2, 3, 4],
            "pred_label": [1, 1, 1, 2],
            "pred_name": ["rivet", "rivet", "rivet", "surface_feature"],
            "pred_confidence": [0.9, 0.8, 0.7, 0.6],
            "prob_background": [0.1, 0.2, 0.3, 0.1],
        }
    )
    features = pd.DataFrame(
        {
            "face_id": [1, 2, 3, 4],
            "surfaceType": [0, 0, 1, 0],
            "neighborPlaneCount": [4, 3, 5, 4],
            "smoothEdgeCount": [0, 0, 0, 0],
        }
    )

    guarded, suppressed = suppress_planar_patch_rivet_predictions(predictions, features)

    assert suppressed == [1]
    assert guarded["pred_label"].tolist() == [0, 1, 1, 2]
    assert guarded["pred_name"].tolist() == ["background", "rivet", "rivet", "surface_feature"]
    assert guarded["pred_confidence"].tolist() == [0.1, 0.8, 0.7, 0.6]


def test_guard_accepts_detector_feature_id_column():
    predictions = pd.DataFrame({"face_id": [8], "pred_label": [1]})
    features = pd.DataFrame(
        {"id": [8], "surfaceType": [0], "neighborPlaneCount": [4], "smoothEdgeCount": [0]}
    )

    guarded, suppressed = suppress_planar_patch_rivet_predictions(predictions, features)

    assert suppressed == [8]
    assert guarded["pred_label"].tolist() == [0]
