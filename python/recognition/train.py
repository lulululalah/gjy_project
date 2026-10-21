"""Retrain the retained three-class model with its fixed train-only cached supervision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from recognition.model import categorical_loss, create_model, evaluate, load_graphs, save_json, sha256
from recognition.predict import load_bundle
from recognition.graph import backward_graph_batch, compute_class_weights

ROOT = Path(__file__).resolve().parents[2]

ACTIVE = ROOT/'model/final_detection_model'

OUTPUT = ROOT/'work/three_class_cached_training_20261019'

CONFIG = dict(seed=123, epochs=200, hidden_dim=64, num_layers=4, adapter_dim=192,
    dropout=.1, batch_size=8, head_lr=.001, encoder_lr=.00002, weight_decay=.0001,
    background_rivet_penalty=3., host_surface_penalty=2., gradient_clip=2.,
    temperature=2., supervised_loss_weight=.5, distillation_loss_weight=1.)

VARIANTS = dict(distilled_finetune=40)

def initialize_encoder(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if any('classifier' in key or 'adapter' in key or 'output' in key for key in state):
        raise ValueError('Initialization must contain encoder tensors only')
    model.encoder.load_state_dict(state, strict=True)
    return len(state)

def student_loss(logits, graph, weights, config):
    target = graph.teacher_logits.detach()
    if target.shape != logits.shape or logits.shape[1] != 3:
        raise ValueError('Teacher categorical targets must align with student outputs')
    temperature = config['temperature']
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    mask = graph.target_mask
    if not mask.any():
        raise ValueError('No supervised faces')
    kl = F.kl_div(F.log_softmax(logits[mask]/temperature, dim=1),
                  F.softmax(target[mask]/temperature, dim=1), reduction='none').sum(1).mean()*temperature**2
    gold = categorical_loss(logits, graph, weights, config)
    return config['supervised_loss_weight']*gold + config['distillation_loss_weight']*kl

def validate_training_scope(graphs, train_names, test_names):
    if set(train_names) & set(test_names):
        raise ValueError('Training/test aircraft overlap')
    if {graph.model_name for graph in graphs} != set(train_names):
        raise ValueError('Teacher targets must use exactly the training aircraft')

def attach_training_targets(graphs, train_names, test_names, assets, output):
    import hashlib

    validate_training_scope(graphs, train_names, test_names)
    index = pd.read_csv(assets/'index.csv').set_index('model_name', verify_integrity=True)
    if set(index.index) != set(train_names):
        raise ValueError('Cached target aircraft differ from training data')
    records = []
    with np.load(assets/'auxiliary_three_class_logits.npz') as arrays:
        for graph in graphs:
            row = index.loc[graph.model_name]
            ids_hash = hashlib.sha256(graph.face_id.numpy().astype(np.int64).tobytes()).hexdigest()
            truth_hash = hashlib.sha256(graph.y.numpy().astype(np.int64).tobytes()).hexdigest()
            if ids_hash != row.face_ids_sha256 or truth_hash != row.truth_sha256:
                raise ValueError('Cached supervision face identifiers or truth differ')
            values = arrays[row.array_key]
            if values.shape != (len(graph.y), 3) or not np.isfinite(values).all():
                raise ValueError('Invalid cached categorical supervision')
            graph.teacher_logits = torch.from_numpy(values.copy())
            records.append(dict(model_name=graph.model_name, graph_id=int(graph.graph_id), faces=len(graph.y)))
    pd.DataFrame(records).to_csv(output/'teacher_target_audit.csv', index=False)
    return records

def train_variant(output, name, graphs, normalization, device, assets, epochs=None):
    config = {**CONFIG, 'frozen_encoder_epochs': VARIANTS[name]}
    if epochs is not None:
        config['epochs'] = epochs
    folder = output/name
    folder.mkdir(exist_ok=True)
    done = folder/'training_complete.json'
    if done.exists():
        saved = json.loads(done.read_text(encoding='utf-8'))
        for path, expected in saved['sha256'].items():
            if sha256(folder/path) != expected:
                raise ValueError('Completed checkpoint or statistics changed')
        return config
    if (folder/'model.pth').exists() or (folder/'stats.npz').exists():
        raise FileExistsError('Partial checkpoint exists')
    torch.manual_seed(config['seed'])
    model = create_model(normalization, config).to(device)
    count = initialize_encoder(model, assets/'initial_encoder.pth')
    alpha = compute_class_weights(graphs, 3, 5.).to(device)
    loader = DataLoader(graphs, batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(config['seed']))
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=config['encoder_lr']),
        dict(params=list(model.adapter.parameters())+list(model.output.parameters()), lr=config['head_lr'])],
        weight_decay=config['weight_decay'])
    history = []
    start = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        frozen = epoch <= config['frozen_encoder_epochs']
        model.train()
        model.encoder.requires_grad_(not frozen)
        if frozen:
            model.encoder.eval()
        progress = (epoch-1)/max(1, config['epochs']-1)
        factor = .1+.9*(1+math.cos(math.pi*progress))/2
        optimizer.param_groups[0]['lr'] = 0. if frozen else config['encoder_lr']*factor
        optimizer.param_groups[1]['lr'] = config['head_lr']*factor
        total, faces = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            value, count_faces = backward_graph_batch(model, batch,
                lambda logits, graph: student_loss(logits, graph, alpha, config), device, 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += value
            faces += count_faces
        history.append(dict(epoch=epoch, loss=total/faces, supervised_faces=faces, encoder_frozen=frozen,
                            encoder_lr=optimizer.param_groups[0]['lr'], head_lr=optimizer.param_groups[1]['lr']))
        if epoch == 1 or epoch % 20 == 0:
            print(f'{name} epoch={epoch}/{config["epochs"]} loss={total/faces:.6f} frozen={frozen}', flush=True)
    torch.save(model.state_dict(), folder/'model.pth')
    np.savez_compressed(folder/'stats.npz', **normalization, configuration_json=json.dumps(config),
        model_architecture='three-class-mlp', encoder_count=1, classifier_outputs=3)
    pd.DataFrame(history).to_csv(folder/'training_history.csv', index=False)
    save_json(done, dict(status='PASS', variant=name, configuration=config,
        parameters=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.perf_counter()-start,
        encoder_tensors_transferred=count, encoder_count=1, classifier_heads=1, binary_heads=0,
        sha256={path: sha256(folder/path) for path in ['model.pth', 'stats.npz']}))
    return config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--variant', choices=list(VARIANTS), default='distilled_finetune')
    parser.add_argument('--smoke', action='store_true', help='Two training epochs, no test evaluation')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest, train_frame, test = load_bundle(ACTIVE)
    assets = ACTIVE/'training_assets'
    asset_manifest = json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    if sha256(ACTIVE/manifest['train_csv']) != asset_manifest['train_csv_sha256']:
        raise ValueError('Training data differ from cached supervision')
    for name, expected in asset_manifest['files'].items():
        if sha256(assets/name) != expected:
            raise ValueError(f'Training asset changed: {name}')
    variants = [args.variant]
    sources = ['recognition/train.py', 'recognition/model.py', 'recognition/graph.py',
        'recognition/predict.py', 'recognition/recovery.py', 'recognition/postprocess.py']
    protocol = dict(configuration=CONFIG, variants=variants, smoke=args.smoke, seed=123,
        input_sha256=manifest['input_sha256'], training_assets=asset_manifest,
        source_sha256={name: sha256(ROOT/'python'/name) for name in sources},
        architecture='one shared GNN, one shared 192-to-192 MLP, one 192-to-3 classifier, softmax argmax',
        training='Train-17 real labels plus cached train-only categorical auxiliary logits; no old recognition checkpoint loaded.',
        checkpoint='Fixed epoch 200 for both; no test threshold, checkpoint, or epoch selection.',
        promotion='Both screened class F1 >= current categorical F1 and foreground mean strictly greater; among eligible candidates choose greatest foreground mean, then fewer background FP.',
        target='Preserve the selected single-three-class formulation and cached-training protocol.',
        scope='Explicit development selection on the known nine-aircraft comparison set; not independent confirmatory evaluation.',
        postprocessing='Existing fixed recovery and all geometry guards, uniformly applied to all nine aircraft.', official_bundle_replaced=False)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol changed; use a new output directory')
    else:
        save_json(output/'protocol.json', protocol)
    snapshot = output/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name, expected in protocol['source_sha256'].items():
        target = snapshot/name
        if target.exists() and sha256(target) != expected:
            raise ValueError('Frozen source changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'python'/name, target)
    graphs, test_graphs, normalization = load_graphs(train_frame, test, assets/'normalization.npz')
    with np.load(ACTIVE/manifest['stats'], allow_pickle=True) as stored:
        for name, values in normalization.items():
            if not np.array_equal(values, stored[name]):
                raise ValueError('Training asset normalization differs from active model')
    records = attach_training_targets(graphs, train_frame.model_name.unique(), set(test.model_name.unique()),
                                      assets, output)
    save_json(output/'training_scope_audit.json', dict(status='PASS', training_aircraft=len(records),
        training_faces=sum(row['faces'] for row in records), test_aircraft_used_for_teacher_supervision=0,
        target_sha256=sha256(assets/'auxiliary_three_class_logits.npz')))
    results = []
    for name in variants:
        config = train_variant(output, name, graphs, normalization, device, assets, epochs=2 if args.smoke else None)
        if args.smoke:
            continue
        folder = output/name
        if not (folder/'evaluation/complete.json').exists():
            evaluate(folder, test_graphs, normalization, manifest, test, config, device, ACTIVE)
        done = json.loads((folder/'evaluation/complete.json').read_text(encoding='utf-8'))
        if done['checkpoint_sha256'] != sha256(folder/'model.pth') or done['stats_sha256'] != sha256(folder/'stats.npz'):
            raise ValueError('Evaluation differs from checkpoint/statistics')
        for relative, expected in done['output_sha256'].items():
            if sha256(folder/'evaluation'/relative) != expected:
                raise ValueError('Evaluation output changed')
        for stage in ['raw', 'screened']:
            summary = json.loads((folder/'evaluation'/f'{stage}_summary.json').read_text(encoding='utf-8'))
            results.append(dict(variant=name, stage=stage, **{k: v for k, v in summary.items() if k != 'confusion'}))
        print(pd.DataFrame(results[-2:]).to_string(index=False), flush=True)
    pd.DataFrame(results).to_csv(output/'comparison.csv', index=False)
    for name, expected in protocol['input_sha256'].items():
        if sha256(ACTIVE/name) != expected:
            raise ValueError('Active bundle changed during training')
    save_json(output/'complete.json', dict(status='PASS', variants=variants, smoke=args.smoke,
        output_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'complete.json'}))
