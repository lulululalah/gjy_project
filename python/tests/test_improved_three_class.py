import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.model import BASE_CONFIG as CONFIG, categorical_loss, create_model
from recognition.predict import bundle_decision, create_bundle_model


class ImprovedThreeClassTests(unittest.TestCase):
    def graph(self, labels):
        count = len(labels)
        return SimpleNamespace(y=torch.tensor(labels), normalized_area=torch.zeros(count),
            shared_boundary_ratio=torch.ones(count), smooth_component_normalized_area=torch.zeros(count),
            smooth_component_face_count=torch.ones(count), target_mask=torch.ones(count, dtype=torch.bool))

    def test_background_rivet_cost_penalizes_false_positive(self):
        graph = self.graph([0])
        logits = torch.tensor([[0., 3., 0.]], requires_grad=True)
        base = categorical_loss(logits, graph, torch.ones(3), {**CONFIG, 'background_rivet_penalty': 0.})
        risk = categorical_loss(logits, graph, torch.ones(3))
        self.assertGreater(float(risk.detach()), float(base.detach()))
        risk.backward()
        self.assertGreater(float(logits.grad[0, 1]), 0.)

    def test_background_penalties_do_not_penalize_true_targets(self):
        logits = torch.tensor([[0., 3., 0.], [0., 0., 3.]])
        loss = categorical_loss(logits, self.graph([1, 2]), torch.ones(3))
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])))

    def test_mask_ignores_unsupervised_faces(self):
        graph = self.graph([1, 0])
        graph.target_mask[1] = False
        logits = torch.tensor([[0., 3., 0.], [0., 20., 0.]])
        torch.testing.assert_close(categorical_loss(logits, graph, torch.ones(3)),
            categorical_loss(logits[:1], self.graph([1]), torch.ones(3)))

    def test_single_head_has_three_jointly_normalized_outputs(self):
        model = create_model(dict(feature_cols=np.arange(48), edge_attr_cols=np.arange(6)),
                             {**CONFIG, 'hidden_dim': 8, 'adapter_dim': 12})
        graph = Data(x=torch.randn(3, 48), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2, 6))
        probability = model(graph).softmax(1)
        self.assertEqual(tuple(probability.shape), (3, 3))
        torch.testing.assert_close(probability.sum(1), torch.ones(3))
        self.assertFalse(any('rivet_' in key or 'surface_' in key for key in model.state_dict()))

    def test_deployed_categorical_decision_uses_argmax_without_binary_thresholds(self):
        stats = dict(model_architecture=np.array('three-class-mlp'))
        logits = torch.log(torch.tensor([[.2, .6, .2], [.1, .1, .8]]))
        labels, probability = bundle_decision(logits, stats)
        self.assertEqual(labels.tolist(), [1, 2])
        torch.testing.assert_close(probability.sum(1), torch.ones(2))

    def test_deployed_loader_preserves_student_keys_and_encoder_width(self):
        normalization = dict(feature_cols=np.arange(48), edge_attr_cols=np.arange(6))
        config = {**CONFIG, 'hidden_dim': 8, 'adapter_dim': 12}
        original = create_model(normalization, config)
        stats = {**normalization, 'model_architecture': np.array('three-class-mlp'),
            'encoder_sharing': np.array('shared'), 'inference_mode': np.array('full'),
            'hidden_dim': np.array(8), 'num_layers': np.array(4), 'adapter_dim': np.array(12), 'dropout': np.array(.1)}
        manifest = dict(model_architecture='three-class-mlp', encoder_sharing='shared', decision='argmax')
        restored = create_bundle_model(stats, manifest)
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(restored.encoder.convs[0].out_channels, 8)
        with self.assertRaises(ValueError):
            create_bundle_model(stats, {**manifest, 'model_architecture': 'dual-head'})


if __name__ == '__main__':
    unittest.main()
