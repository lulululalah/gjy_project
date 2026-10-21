import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.predict import compute_metrics, recovery_threshold, screen_predictions, validate_datasets, validate_predictions
from recognition.predict import canonical_model as canonical_name


class RecognitionBundleContractTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"graph_id": [1, 1], "model_name": ["train.step"] * 2,
                                   "id": [1, 2], "label": [0, 1]})
        self.test = pd.DataFrame({"graph_id": [0, 0, 0], "model_name": ["test.step"] * 3,
                                  "id": [1, 2, 3], "label": [0, 1, 2]})
        self.predictions = pd.DataFrame({"model_name": ["test.step"] * 3, "face_id": [3, 1, 2],
                                        "true_label": [2, 0, 1], "raw_label": [2, 0, 1],
                                        "pred_label": [2, 0, 0]})

    def test_prediction_alignment_uses_aircraft_and_face_identity_not_row_order(self):
        validate_datasets(self.train, self.test)
        validate_predictions(self.test, self.predictions)

    def test_duplicate_or_missing_prediction_faces_are_rejected(self):
        for frame in (pd.concat([self.predictions, self.predictions.iloc[:1]]), self.predictions.iloc[:2]):
            with self.assertRaises(ValueError):
                validate_predictions(self.test, frame)

    def test_wrong_truth_and_cross_target_relabelling_are_rejected(self):
        wrong_truth = self.predictions.copy()
        wrong_truth.loc[0, "true_label"] = 0
        with self.assertRaisesRegex(ValueError, "truth labels differ"):
            validate_predictions(self.test, wrong_truth)
        wrong_target = self.predictions.copy()
        wrong_target.loc[0, "pred_label"] = 1
        with self.assertRaisesRegex(ValueError, "another target"):
            validate_predictions(self.test, wrong_target)

    def test_aircraft_overlap_is_rejected_even_with_a_decal_suffix(self):
        overlapping = self.test.assign(model_name="train_decals.step")
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_datasets(self.train, overlapping)

    def test_duplicate_faces_and_multiple_aircraft_per_graph_are_rejected(self):
        duplicate = pd.concat([self.test, self.test.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_datasets(self.train, duplicate)
        mixed = self.test.copy()
        mixed.loc[0, "model_name"] = "another.step"
        with self.assertRaisesRegex(ValueError, "one aircraft"):
            validate_datasets(self.train, mixed)

    def test_three_class_metrics_match_hand_counted_confusion(self):
        rows, summary = compute_metrics([0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 1, 0, 2, 2])
        self.assertEqual(summary["confusion"], [[1, 1, 1], [1, 1, 0], [0, 0, 2]])
        self.assertEqual([r["support"] for r in rows], [3, 2, 2])
        self.assertEqual(summary["background_fp"], 2)
        self.assertAlmostEqual(summary["rivet_f1"], 0.5)
        self.assertAlmostEqual(summary["surface_feature_f1"], 0.8)
        self.assertAlmostEqual(summary["foreground_macro_f1"], 0.65)

    def test_segmented_inputs_and_predictions_share_one_aircraft_name(self):
        self.assertEqual(canonical_name("aircraft_decals.features.csv"), "aircraft")
        self.assertEqual(canonical_name("aircraft.pred.csv"), "aircraft")
        self.assertEqual(canonical_name("aircraft_decals.step.csv"), "aircraft")

    def test_recovery_requires_background_origin_score_and_explicit_trace(self):
        recovered = self.predictions.copy()
        recovered.loc[1, "pred_label"] = 2
        recovered["recovery_candidate"] = [False, True, False]
        recovered["surface_recovered"] = [False, True, False]
        recovered["recovery_score"] = [0., .97, 0.]
        validate_predictions(self.test, recovered, recovery_threshold=.95)
        with self.assertRaises(ValueError):
            validate_predictions(self.test, recovered)
        recovered.loc[1, "recovery_score"] = .94
        with self.assertRaises(ValueError):
            validate_predictions(self.test, recovered, recovery_threshold=.95)

    def test_recovery_configuration_requires_matching_training_and_checkpoint(self):
        with patch("pathlib.Path.read_text") as read_configuration:
            bundle = Path("unused")
            configuration = dict(enabled=True, train_sha256="training", checkpoint_sha256="weights", minimum_score=.95)
            manifest = {"surface_recovery": {"enabled": True, "configuration": "recovery.json"},
                        "train_csv": "train.csv", "checkpoint": "model.pth",
                        "input_sha256": {"train.csv": "training", "model.pth": "weights"}}
            read_configuration.return_value = json.dumps(configuration)
            self.assertEqual(recovery_threshold(bundle, manifest), .95)
            for key in ("train_sha256", "checkpoint_sha256"):
                read_configuration.return_value = json.dumps({**configuration, key: "different"})
                with self.assertRaises(ValueError):
                    recovery_threshold(bundle, manifest)
            read_configuration.return_value = json.dumps({**configuration, "enabled": False})
            with self.assertRaisesRegex(ValueError, "failed calibration"):
                recovery_threshold(bundle, manifest)

    def test_later_guards_can_revoke_recovery_and_restore_background_confidence(self):
        raw = pd.DataFrame({"face_id": [1], "pred_label": [0], "pred_name": ["background"],
                            "pred_confidence": [.7], "prob_background": [.7], "prob_surface_feature": [.25]})
        class Recovery:
            def apply(self, raw, features):
                return raw.assign(pred_label=2, pred_name="surface_feature", pred_confidence=.98,
                                  recovery_candidate=True, recovery_score=.98)
        with patch("recognition.postprocess.apply_base_guards", side_effect=lambda frame, _: (frame.copy(), 0)), \
             patch("recognition.postprocess.suppress_planar_patch_rivet_predictions", side_effect=lambda frame, _: (frame, 0)), \
             patch("recognition.postprocess.apply_exterior_visibility_surface_guard", return_value=([0], {})), \
             patch("recognition.postprocess.suppress_low_confidence_dense_surface_predictions", side_effect=lambda labels, _: (labels, 0, {})):
            result = screen_predictions(raw, pd.DataFrame(), Path("unused.step"), Path("unused"), Recovery())
        self.assertEqual(result.pred_label.tolist(), [0])
        self.assertEqual(result.pred_name.tolist(), ["background"])
        self.assertEqual(result.pred_confidence.tolist(), [.7])
        self.assertEqual(result.surface_recovered.tolist(), [False])





class WorkflowCommandTests(unittest.TestCase):
    def test_dispatch_preserves_predict_options_and_exit_status(self):
        import main as workflow
        from types import SimpleNamespace
        from unittest.mock import Mock
        entry = Mock(return_value=7)
        with patch.object(sys, 'argv', ['main.py', 'predict', '--verify-only']), patch.object(workflow.importlib, 'import_module', return_value=SimpleNamespace(main=entry)) as load:
            self.assertEqual(workflow.main(), 7)
            self.assertEqual(sys.argv, ['main.py predict', '--verify-only'])
        load.assert_called_once_with('recognition.predict')
        entry.assert_called_once_with()

    def test_unknown_command_fails_before_loading_a_model(self):
        import main as workflow
        import contextlib,io
        with patch.object(sys, 'argv', ['main.py', 'old-dual-head']), patch.object(workflow.importlib, 'import_module') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                workflow.main()
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
