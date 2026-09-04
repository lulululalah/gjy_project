import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from train_rivet_gcn import (
    apply_smooth_shell_surface_guard,
    compute_attached_surface_feature_frame,
)


class AttachedSurfaceFeatureTests(unittest.TestCase):
    def test_shell_guard_rejects_dominant_and_dense_shell_faces_only(self):
        predictions = torch.tensor([2, 2, 2, 1])
        guarded = apply_smooth_shell_surface_guard(
            predictions,
            component_normalized_areas=torch.tensor([0.5, 0.5, 0.1, 0.5]),
            component_face_counts=torch.tensor([2.0, 195.0, 195.0, 195.0]),
            component_face_area_ratios=torch.tensor([0.5, 0.04, 0.04, 0.04]),
        )
        self.assertEqual(guarded.tolist(), [0, 0, 2, 1])

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


if __name__ == "__main__":
    unittest.main()
