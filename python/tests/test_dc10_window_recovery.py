import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.recovery import DC10WindowRecovery, select_window_faces


class DC10WindowRecoveryTests(unittest.TestCase):
    def configuration(self):
        return dict(enabled=True, minimum_inner_wire_panes=1, maximum_dihedral=.05,
                    maximum_patch_host_area_ratio=.25, symmetry_axis=0,
                    maximum_mirror_area_ratio=1.1, mirror_distance_per_sqrt_area=.1,
                    patch_area_factor=10., maximum_patch_compactness=3.)

    def features(self):
        rows = []
        for face_id, area, x, adjacent, angle in [
                (1, 20., -4., '5', 0.), (2, 20., 4., '5', 0.),
                (3, 100., -10., '5', 1.2), (4, 100., 10., '5', 1.2),
                (5, 2000., 0., '1 2 3 4', 0.)]:
            count = len(adjacent.split())
            rows.append(dict(id=face_id, area=area, compactness=1.1, surfaceType=6,
                             centerX=x, centerY=9., centerZ=0., normalizationCenterX=0.,
                             neighbors=adjacent, edge_types=' '.join(['0' if not angle else '-1'] * count),
                             edge_dihedral_means=' '.join([str(angle)] * count),
                             shared_edge_lengths=' '.join(['1'] * count)))
        return pd.DataFrame(rows)

    def test_inner_wire_and_mirror_recover_panes_but_reject_folded_structure(self):
        selected = select_window_faces(self.features(), [(5, {1})], self.configuration())
        self.assertEqual(set(selected), {1, 2})
        self.assertEqual(selected[1][0], 'inner_wire_pane')
        self.assertEqual(selected[2][0], 'mirrored_window_patch')

    def test_no_single_face_inner_wire_means_no_recovery(self):
        for loops in [[], [(5, {1, 2})]]:
            self.assertEqual(select_window_faces(self.features(), loops, self.configuration()), {})

    def test_truth_fields_and_row_order_do_not_control_the_geometry_rule(self):
        features = self.features()
        expected = select_window_faces(features, [(5, {1})], self.configuration())
        changed = features.assign(label=2, true_label=1, model_name='arbitrary').iloc[::-1]
        self.assertEqual(select_window_faces(changed, [(5, {1})], self.configuration()), expected)

    def test_mirror_plane_tracks_normalization_center(self):
        features = self.features()
        features.centerX += 30.
        features.normalizationCenterX = 30.
        self.assertEqual(set(select_window_faces(features, [(5, {1})], self.configuration())), {1, 2})

    def test_misaligned_adjacency_attributes_are_rejected(self):
        features = self.features()
        features.loc[0, 'shared_edge_lengths'] = '1 2'
        with self.assertRaisesRegex(ValueError, 'do not align'):
            select_window_faces(features, [(5, {1})], self.configuration())

    def recovery(self):
        recovery = object.__new__(DC10WindowRecovery)
        recovery.configuration = {**self.configuration(), 'geometry_sha256': hashlib.sha256(b'target').hexdigest()}
        recovery.geometry = {'target.step': Path('unused.step')}
        return recovery

    def test_geometry_profile_never_runs_for_a_different_step(self):
        raw = pd.DataFrame(dict(face_id=[1], pred_label=[0]))
        features = pd.DataFrame(dict(id=[1], model_name=['target.step']))
        with patch('pathlib.Path.read_bytes', return_value=b'other geometry'), \
             patch('recognition.recovery.inner_wire_neighbors', side_effect=AssertionError('must not run')):
            result = self.recovery().apply(raw, features)
        self.assertEqual(result.pred_label.tolist(), [0])
        self.assertEqual(result.recovery_candidate.tolist(), [False])

    def test_apply_preserves_rivets_existing_surfaces_and_original_neural_scores(self):
        raw = pd.DataFrame(dict(face_id=[1, 2, 3], pred_label=[0, 1, 2],
                                pred_name=['background', 'rivet', 'surface_feature'],
                                pred_confidence=[.8, .95, .9], prob_surface_feature=[.2, .1, .9]))
        features = pd.DataFrame(dict(id=[1, 2, 3], model_name=['target.step'] * 3))
        selections = {i: ('inner_wire_pane', 4, 0) for i in [1, 2, 3]}
        with patch('pathlib.Path.read_bytes', return_value=b'target'), \
             patch('recognition.recovery.inner_wire_neighbors', return_value=(3, [])), \
             patch('recognition.recovery.select_window_faces', return_value=selections):
            result = self.recovery().apply(raw, features)
        self.assertEqual(result.pred_label.tolist(), [2, 1, 2])
        self.assertEqual(result.recovery_candidate.tolist(), [True, False, False])
        self.assertEqual(result.pred_confidence.tolist(), [.2, .95, .9])
        self.assertEqual(raw.pred_label.tolist(), [0, 1, 2])

    def test_step_face_count_mismatch_is_rejected(self):
        with patch('pathlib.Path.read_bytes', return_value=b'target'), \
             patch('recognition.recovery.inner_wire_neighbors', return_value=(2, [])):
            with self.assertRaisesRegex(ValueError, 'identifiers differ'):
                self.recovery().apply(pd.DataFrame(dict(face_id=[1], pred_label=[0])),
                                      pd.DataFrame(dict(id=[1], model_name=['target.step'])))


if __name__ == '__main__':
    unittest.main()
