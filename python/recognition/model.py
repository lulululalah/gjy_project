"""Single three-class network, supervised loss and fixed-bundle evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from recognition.recovery import DC10WindowRecovery
from recognition.graph import FaceGraphEncoder, build_graph
from recognition.predict import LABEL_NAMES, canonical_model, compute_metrics, screen_predictions, validate_predictions, sha256

ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT/'model/final_detection_model'

BASE_CONFIG = dict(hidden_dim=64, num_layers=4, adapter_dim=192, dropout=.1,
                   background_rivet_penalty=3., host_surface_penalty=2.)

def save_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

class ThreeClassGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=64, num_layers=4, adapter_dim=192, dropout=.1):
        super().__init__()
        self.encoder = FaceGraphEncoder(node_features, edge_features, hidden_dim, 3, num_layers, dropout)
        self.encoder.classifier = torch.nn.Identity()
        width = hidden_dim if num_layers == 3 else hidden_dim*3
        self.adapter = torch.nn.Sequential(torch.nn.Linear(width, adapter_dim),
            torch.nn.LayerNorm(adapter_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.output = torch.nn.Linear(adapter_dim, 3)
    def forward(self, graph):
        return self.output(self.adapter(self.encoder.encode(graph)))

def create_model(normalization, config=BASE_CONFIG):
    return ThreeClassGNN(len(normalization['feature_cols']), len(normalization['edge_attr_cols']),
        config['hidden_dim'], config['num_layers'], config['adapter_dim'], config['dropout'])

def load_graphs(train, test, stats_path):
    with np.load(stats_path, allow_pickle=True) as stats:
        normalization = {key:stats[key].copy() for key in ['mean','std','edge_mean','edge_std','feature_cols','edge_attr_cols']}
    def build(frame):
        return [build_graph(group.reset_index(drop=True), normalization['mean'], normalization['std'],
            normalization['edge_mean'], normalization['edge_std'], normalization['feature_cols'].tolist())
            for _,group in frame.groupby('graph_id', sort=True)]
    return build(train),build(test),normalization

def categorical_loss(logits, graph, class_weights, config=BASE_CONFIG):
    """Joint softmax CE plus class-specific costs on true background faces."""
    if logits.ndim != 2 or logits.shape != (len(graph.y), 3):
        raise ValueError('Expected one three-class logit vector per face')
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_probs, graph.y, weight=class_weights, reduction='none')
    background = graph.y == 0
    probabilities = log_probs.exp()
    loss = loss + background * config['background_rivet_penalty'] * (
        -torch.log1p(-probabilities[:, 1].clamp(max=1.-1e-6)))
    host = background & (((graph.normalized_area >= .01) & (graph.shared_boundary_ratio <= .2)) |
        ((graph.smooth_component_normalized_area >= .25) & (graph.smooth_component_face_count >= 2)))
    loss = loss + host * config['host_surface_penalty'] * (
        -torch.log1p(-probabilities[:, 2].clamp(max=1.-1e-6)))
    mask = getattr(graph, 'target_mask', torch.ones_like(background))
    if not mask.any():
        raise ValueError('No supervised faces')
    return loss[mask].mean()

def evaluate(output, graphs, normalization, manifest, test, config, device, donor_bundle=BUNDLE):
    model = create_model(normalization, config).to(device)
    model.load_state_dict(torch.load(output/'model.pth', map_location=device, weights_only=True), strict=True)
    model.eval()
    geometry = {name: (donor_bundle/item['path']).resolve() for name, item in manifest['geometry'].items()}
    recovery = DC10WindowRecovery(donor_bundle/manifest['surface_recovery']['configuration'], geometry)
    folder = output/'evaluation'
    folder.mkdir()
    frames = []
    for graph in graphs:
        name = graph.model_name
        features = test[test.model_name == name].reset_index(drop=True)
        print(f'Evaluate argmax: {name}', flush=True)
        with torch.no_grad():
            scores = model(graph.clone().to(device)).softmax(1).cpu().numpy()
        labels = scores.argmax(1)
        raw = pd.DataFrame(dict(face_id=features.id.astype(int), pred_label=labels,
            pred_name=[LABEL_NAMES[int(label)] for label in labels], pred_confidence=scores[np.arange(len(scores)), labels],
            prob_background=scores[:, 0], prob_rivet=scores[:, 1], prob_surface_feature=scores[:, 2]))
        final = screen_predictions(raw, features, geometry[name],
            ROOT/'work/recognition_geometry_cache', recovery=recovery)
        final.insert(0, 'model_name', name)
        final['true_label'], final['raw_label'] = features.label.to_numpy(), labels
        frames.append(final)
    predictions = pd.concat(frames, ignore_index=True)
    validate_predictions(test, predictions, recovery_threshold=0.)
    predictions.to_csv(folder/'predictions.csv', index=False)
    for stage, column in [('raw', 'raw_label'), ('screened', 'pred_label')]:
        rows, summary = compute_metrics(predictions.true_label, predictions[column])
        pd.DataFrame(rows).to_csv(folder/f'{stage}_class_metrics.csv', index=False)
        save_json(folder/f'{stage}_summary.json', summary)
    (folder/'predictions').mkdir()
    audits = []
    for name, group in predictions.groupby('model_name', sort=True):
        group.drop(columns='model_name').to_csv(folder/'predictions'/f'{canonical_model(name)}.pred.csv', index=False)
        _, summary = compute_metrics(group.true_label, group.pred_label)
        audits.append(dict(model_name=name, **{k: v for k, v in summary.items() if k != 'confusion'}))
    pd.DataFrame(audits).to_csv(folder/'per_aircraft_audit.csv', index=False)
    save_json(folder/'complete.json', dict(status='PASS', checkpoint_sha256=sha256(output/'model.pth'),
        stats_sha256=sha256(output/'stats.npz'), output_sha256={str(p.relative_to(folder)): sha256(p)
            for p in folder.rglob('*') if p.is_file()}))
