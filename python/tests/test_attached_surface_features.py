import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from train_rivet_gcn import (
    ATTACHED_SURFACE_FEATURE_COLS,
    SMOOTH_COMPONENT_RUNTIME_COLS,
    apply_decal_only_surface_guard,
    apply_rivet_size_guard,
    apply_smooth_shell_surface_guard,
    apply_wing_shell_surface_guard,
    compute_attached_surface_feature_frame,
)


class AttachedSurfaceFeatureTests(unittest.TestCase):
    def test_smooth_component_runtime_values_are_not_model_features(self):
        self.assertTrue(set(SMOOTH_COMPONENT_RUNTIME_COLS).isdisjoint(ATTACHED_SURFACE_FEATURE_COLS))

    def test_shell_guard_rejects_dominant_and_dense_shell_faces_only(self):
        predictions = torch.tensor([2, 2, 2, 1])
        guarded = apply_smooth_shell_surface_guard(
            predictions,
            component_normalized_areas=torch.tensor([0.5, 0.5, 0.1, 0.5]),
            component_face_counts=torch.tensor([2.0, 195.0, 195.0, 195.0]),
            component_face_area_ratios=torch.tensor([0.5, 0.04, 0.04, 0.04]),
        )
        self.assertEqual(guarded.tolist(), [0, 0, 2, 1])

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

    def test_decal_only_guard_ignores_rivets_and_keeps_only_decal_signature(self):
        frame = pd.DataFrame([
            {
                "surfaceType": 6,
                "numEdges": 2,
                "innerWireCount": 0,
                "neighborCurvedCount": 1,
                "smoothEdgeCount": 1,
            },
            {
                "surfaceType": 0,
                "numEdges": 6,
                "innerWireCount": 0,
                "neighborCurvedCount": 4,
                "smoothEdgeCount": 0,
            },
            {
                "surfaceType": 6,
                "numEdges": 2,
                "innerWireCount": 0,
                "neighborCurvedCount": 1,
                "smoothEdgeCount": 1,
            },
        ])
        guarded = apply_decal_only_surface_guard(torch.tensor([2, 2, 1]), frame)
        self.assertEqual(guarded.tolist(), [2, 0, 0])

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


if __name__ == "__main__":
    unittest.main()
