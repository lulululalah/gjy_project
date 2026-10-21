"""Prevent resumed experiments from mixing checkpoints, topology or predictions."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad.pipeline as experiment


class EvidenceIntegrityTests(unittest.TestCase):
    def test_cad_recovery_uses_the_configuration_from_the_selected_bundle(self):
        import json
        step = Path('current-aircraft.step')
        manifest = json.loads((experiment.BUNDLE/'manifest.json').read_text(encoding='utf-8'))
        expected = json.loads((experiment.BUNDLE/manifest['surface_recovery']['configuration']).read_text(encoding='utf-8'))
        recovery = experiment.cad_recovery(step)
        self.assertEqual(recovery.configuration, expected)
        self.assertEqual(recovery.geometry, {step.name: step})

    def test_disabled_recovery_does_not_construct_a_recovery_model(self):
        with patch.object(experiment.json, 'loads', return_value={'surface_recovery': {'enabled': False}}):
            with patch('recognition.recovery.DC10WindowRecovery') as constructor:
                self.assertIsNone(experiment.cad_recovery(Path('current-aircraft.step')))
                constructor.assert_not_called()

    def record(self):
        return dict(checkpoint_sha256='selected',
            original=dict(path='source.step', sha256='source'),
            rivet_endpoint=dict(path='case/rivet.step', sha256='rivet'),
            final=dict(path='case/final.step', sha256='final'),
            original_predictions_sha256='pred0', intermediate_predictions_sha256='pred1',
            intermediate_features_sha256='features1')

    def test_another_checkpoint_is_rejected_before_reading_cache(self):
        with patch.object(experiment, 'sha256') as digest:
            with self.assertRaisesRegex(ValueError, 'another checkpoint'):
                experiment.verify_completed_record(self.record(), 'different')
        digest.assert_not_called()

    def test_changed_intermediate_topology_is_rejected(self):
        with patch.object(experiment, 'sha256', side_effect=['source', 'changed']):
            with self.assertRaisesRegex(ValueError, 'rivet_endpoint'):
                experiment.verify_completed_record(self.record(), 'selected')

    def test_changed_predictions_are_rejected_even_when_step_files_match(self):
        with patch.object(experiment, 'sha256', side_effect=['source','rivet','final','pred0','changed']):
            with self.assertRaisesRegex(ValueError, 'intermediate.pred.csv'):
                experiment.verify_completed_record(self.record(), 'selected')

    def test_unchanged_complete_chain_is_accepted(self):
        with patch.object(experiment, 'sha256', side_effect=['source','rivet','final','pred0','pred1','features1']):
            experiment.verify_completed_record(self.record(), 'selected')


if __name__ == '__main__':
    unittest.main()
