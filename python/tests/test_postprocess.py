import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.postprocess import apply_base_guards


class AblationPostprocessTests(unittest.TestCase):
    def features(self):
        return pd.DataFrame({
            "id": [1, 2, 3], "surfaceType": [0, 0, 0],
            "innerWireCount": [0, 0, 0], "numEdges": [3, 3, 3],
            "relativeArea": [1e-5, 1e-7, 0.002], "compactness": [1.0, 1.0, 1.0],
            "neighborPlaneCount": [0, 0, 0], "neighborCurvedCount": [1, 1, 1],
            "convexEdgeCount": [0, 0, 0], "smoothEdgeCount": [1, 1, 1],
        })

    def predictions(self):
        return pd.DataFrame({"face_id": [1, 2, 3], "pred_label": [1, 1, 0],
                             "pred_name": ["rivet", "rivet", "background"],
                             "pred_confidence": [0.9, 0.95, 0.8], "prob_background": [0.1, 0.05, 0.8]})

    def test_rivet_area_guard_updates_metadata_without_modifying_raw_predictions(self):
        original = self.predictions()
        screened, count = apply_base_guards(original, self.features())
        self.assertEqual(screened.pred_label.tolist(), [0, 1, 0])
        self.assertEqual(count, 1)
        self.assertEqual(screened.loc[0, "pred_name"], "background")
        self.assertEqual(screened.loc[0, "pred_confidence"], 0.1)
        self.assertEqual(original.pred_label.tolist(), [1, 1, 0])

    def test_wing_guard_rejects_a_shell_candidate_but_keeps_other_surface_faces(self):
        predictions = self.predictions()
        predictions["pred_label"] = [2, 2, 0]
        predictions["pred_name"] = ["surface_feature", "surface_feature", "background"]
        features = self.features()
        features.loc[0, ["surfaceType", "numEdges", "relativeArea", "compactness",
                         "neighborPlaneCount", "neighborCurvedCount", "convexEdgeCount", "smoothEdgeCount"]] = [6, 4, 0.002, 2.0, 4, 0, 4, 0]
        screened, count = apply_base_guards(predictions, features)
        self.assertEqual(screened.pred_label.tolist(), [0, 2, 0])
        self.assertEqual(count, 1)

    def test_misaligned_faces_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            apply_base_guards(self.predictions(), self.features().iloc[::-1])



import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from recognition.graph import ATTACHED_SURFACE_FEATURE_COLS, SMOOTH_COMPONENT_RUNTIME_COLS, compute_attached_surface_feature_frame
from recognition.postprocess import apply_rivet_size_guard, apply_wing_shell_surface_guard
from recognition.postprocess import suppress_low_confidence_dense_surface_predictions, suppress_fully_occluded_surface_predictions


class AttachedSurfaceFeatureTests(unittest.TestCase):


    def test_pure_visibility_guard_checks_every_surface_candidate(self):
        guarded, suppressed = suppress_fully_occluded_surface_predictions(
            predictions=[2, 2, 1, 0],
            exposure_scores=[0.0, 0.2, 0.0, 0.0],
        )
        self.assertEqual(guarded.tolist(), [0, 2, 1, 0])
        self.assertEqual(suppressed, [0])

    def test_dense_surface_guard_suppresses_only_low_confidence_candidates(self):
        guarded, suppressed, density = suppress_low_confidence_dense_surface_predictions(
            predictions=[2, 2, 2, 0],
            surface_probabilities=[0.69, 0.70, 0.95, 0.99],
            density_threshold=0.5,
            minimum_surface_confidence=0.7,
        )
        self.assertAlmostEqual(density, 0.75)
        self.assertEqual(guarded.tolist(), [0, 2, 2, 0])
        self.assertEqual(suppressed, [0])

    def test_dense_surface_guard_does_not_trigger_at_normal_density(self):
        guarded, suppressed, density = suppress_low_confidence_dense_surface_predictions(
            predictions=[2, 0, 0, 0],
            surface_probabilities=[0.2, 0.1, 0.1, 0.1],
        )
        self.assertAlmostEqual(density, 0.25)
        self.assertEqual(guarded.tolist(), [2, 0, 0, 0])
        self.assertEqual(suppressed, [])

    def test_smooth_component_runtime_values_are_not_model_features(self):
        self.assertTrue(set(SMOOTH_COMPONENT_RUNTIME_COLS).isdisjoint(ATTACHED_SURFACE_FEATURE_COLS))


    def test_wing_shell_guard_rejects_wing_patch_but_keeps_large_decal(self):
        frame = pd.DataFrame([
            {
                "surfaceType": 6,
                "innerWireCount": 0,
                "numEdges": 5,
                "relativeArea": 0.00089,
                "compactness": 1.54,
                "neighborPlaneCount": 4,
                "neighborCurvedCount": 1,
                "convexEdgeCount": 4,
                "smoothEdgeCount": 1,
            },
            {
                "surfaceType": 6,
                "innerWireCount": 0,
                "numEdges": 4,
                "relativeArea": 0.0017,
                "compactness": 5.0,
                "neighborPlaneCount": 4,
                "neighborCurvedCount": 0,
                "convexEdgeCount": 4,
                "smoothEdgeCount": 0,
            },
            {
                "surfaceType": 6,
                "innerWireCount": 0,
                "numEdges": 4,
                "relativeArea": 0.00131,
                "compactness": 1.91,
                "neighborPlaneCount": 2,
                "neighborCurvedCount": 2,
                "convexEdgeCount": 3,
                "smoothEdgeCount": 1,
            },
            {
                "surfaceType": 6,
                "innerWireCount": 0,
                "numEdges": 4,
                "relativeArea": 0.0017,
                "compactness": 5.0,
                "neighborPlaneCount": 4,
                "neighborCurvedCount": 0,
                "convexEdgeCount": 4,
                "smoothEdgeCount": 0,
            },
        ])
        guarded = apply_wing_shell_surface_guard(torch.tensor([2, 2, 2, 1]), frame)
        self.assertEqual(guarded.tolist(), [0, 0, 2, 1])


    def test_rivet_size_guard_rejects_only_oversized_rivets(self):
        frame = pd.DataFrame({"relativeArea": [5e-7, 5e-6, 5.1e-6]})
        guarded = apply_rivet_size_guard(torch.tensor([1, 1, 1]), frame)
        self.assertEqual(guarded.tolist(), [1, 1, 0])

    def test_embedded_patch_features_use_smooth_same_surface_neighbor(self):
        frame = pd.DataFrame([{
            "area": 1.0,
            "perimeter": 4.0,
            "normalizationScale": 0.5,
            "areaToNeighborMean": 0.2,
            "areaToNeighborMax": 0.1,
            "surfaceType": 6,
            "edge_types": "0 1",
            "edge_neighbor_surface_types": "6 6",
            "shared_edge_lengths": "2 2",
            "edge_area_ratios": "0.1 2",
            "edge_dihedral_means": "0.01 1.0",
            "neighbors": "7 8",
            "numEdges": 4,
        }])
        features = compute_attached_surface_feature_frame(frame).iloc[0]
        self.assertEqual(features["sameSurfaceNeighborRatio"], 1.0)
        self.assertEqual(features["smoothSameSurfaceNeighborRatio"], 0.5)
        self.assertEqual(features["smoothSameSurfaceBoundaryRatio"], 0.5)
        self.assertAlmostEqual(features["smoothSameSurfaceAreaContrast"], -np.log(0.1))
        self.assertEqual(features["hasLargerSmoothSameSurfaceNeighbor"], 1.0)
        self.assertEqual(features["uniqueNeighborToEdgeRatio"], 0.5)
        self.assertEqual(features["sharedBoundaryRatio"], 1.0)

    def test_no_smooth_same_surface_neighbor_has_neutral_host_features(self):
        frame = pd.DataFrame([{
            "area": 10.0,
            "perimeter": 8.0,
            "normalizationScale": 0.5,
            "areaToNeighborMean": 10.0,
            "areaToNeighborMax": 10.0,
            "surfaceType": 6,
            "edge_types": "1",
            "edge_neighbor_surface_types": "6",
            "shared_edge_lengths": "8",
            "edge_area_ratios": "10",
            "edge_dihedral_means": "1.0",
            "neighbors": "7",
            "numEdges": 8,
        }])
        features = compute_attached_surface_feature_frame(frame).iloc[0]
        self.assertEqual(features["smoothSameSurfaceNeighborRatio"], 0.0)
        self.assertEqual(features["smoothSameSurfaceBoundaryRatio"], 0.0)
        self.assertEqual(features["smoothSameSurfaceAreaContrast"], 0.0)
        self.assertEqual(features["hasLargerSmoothSameSurfaceNeighbor"], 0.0)
        self.assertEqual(features["uniqueNeighborToEdgeRatio"], 0.125)
        self.assertEqual(features["sharedBoundaryRatio"], 1.0)



if __name__ == '__main__':
    unittest.main()
