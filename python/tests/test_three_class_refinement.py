from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.train import CONFIG, student_loss, validate_training_scope


class ThreeClassRefinementTests(unittest.TestCase):
    def graph(self, targets):
        count = len(targets)
        return SimpleNamespace(y=torch.tensor(targets), teacher_logits=torch.randn(count, 3, requires_grad=True),
            normalized_area=torch.zeros(count), shared_boundary_ratio=torch.ones(count),
            smooth_component_normalized_area=torch.zeros(count), smooth_component_face_count=torch.ones(count),
            target_mask=torch.ones(count, dtype=torch.bool))

    def test_cached_supervision_requires_exact_training_scope(self):
        graphs = [SimpleNamespace(model_name='train.step')]
        validate_training_scope(graphs, ['train.step'], {'test.step'})
        with self.assertRaises(ValueError):
            validate_training_scope(graphs, ['other.step'], {'test.step'})
        with self.assertRaises(ValueError):
            validate_training_scope(graphs, ['train.step'], {'train.step'})

    def test_distillation_does_not_backpropagate_into_teacher(self):
        graph = self.graph([0, 1, 2])
        logits = torch.randn(3, 3, requires_grad=True)
        loss = student_loss(logits, graph, torch.ones(3), CONFIG)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(graph.teacher_logits.grad)

    def test_matching_soft_targets_have_zero_kl_without_gold_loss(self):
        graph = self.graph([0, 1])
        loss = student_loss(graph.teacher_logits.detach().clone(), graph, torch.ones(3),
                            {**CONFIG, 'supervised_loss_weight': 0.})
        self.assertAlmostEqual(float(loss), 0., places=6)

    def test_invalid_teacher_targets_are_rejected(self):
        graph = self.graph([0, 1])
        graph.teacher_logits = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            student_loss(torch.zeros(2, 3), graph, torch.ones(3), CONFIG)


if __name__ == '__main__':
    unittest.main()
