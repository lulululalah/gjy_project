"""Run the current staged CAD processing and paired mesh experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recognition.predict import LABEL_NAMES, create_bundle_model, bundle_decision, screen_predictions, sha256
from cad.candidates import stage_stem, verified_route, reconstruction_profile, intermediate_step_name
from cad.candidates import transfer_surface_candidates, complete_geometric_candidates

ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / 'model/final_detection_model'

EVIDENCE = ROOT / 'results/current_three_class_complete_20261020'

DETECTOR = ROOT / 'build/Release/Detector.exe'

PROTOCOL_ID = 'complete-three-class-cad-v3'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def command(args, log):
    started = time.perf_counter()
    print(subprocess.list2cmdline(list(map(str, args))), flush=True)
    with log.open('w',encoding='utf-8') as output:
        result = subprocess.run(list(map(str, args)), cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
    return {'returncode': result.returncode, 'seconds': time.perf_counter()-started,
            'command': list(map(str, args)), 'log': str(log.relative_to(ROOT))}

def geometry(path):
    from geometry.measure import load_step
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    shape = load_step(path)
    counts = {}
    for name, kind in [('faces', TopAbs_FACE), ('edges', TopAbs_EDGE)]:
        items = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, kind, items)
        counts[name] = items.Size()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), **counts)

def verify_completed_record(record, checkpoint_sha256):
    if record['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Completed CAD evidence belongs to another checkpoint')
    for key in ['original', 'rivet_endpoint', 'final']:
        item = record[key]
        if sha256(ROOT/item['path']) != item['sha256']:
            raise ValueError('A completed endpoint was modified: '+key)
    folder = (ROOT/record['final']['path']).parent
    for name, key in [('original.pred.csv', 'original_predictions_sha256'),
                      ('intermediate.pred.csv', 'intermediate_predictions_sha256'),
                      ('intermediate_features.csv', 'intermediate_features_sha256')]:
        if sha256(folder/name) != record[key]:
            raise ValueError('Completed CAD evidence was modified: '+name)

def infer(step, csv_path, out, model, stats, device):
    from recognition.graph import build_graph
    from recognition.recovery import DC10WindowRecovery
    features = pd.read_csv(csv_path)
    features['model_name'] = step.name
    graph = build_graph(features, stats['mean'], stats['std'], stats['edge_mean'],
                        stats['edge_std'], stats['feature_cols'].tolist()).to(device)
    with torch.no_grad():
        labels, scores = bundle_decision(model(graph), stats)
    labels, scores = labels.cpu().numpy(), scores.cpu().numpy()
    raw = pd.DataFrame(dict(face_id=features.id.astype(int), pred_label=labels,
        pred_name=[LABEL_NAMES[x] for x in labels],
        pred_confidence=scores[np.arange(len(labels)), labels],
        prob_background=scores[:, 0], prob_rivet=scores[:, 1], prob_surface_feature=scores[:, 2]))
    recovery = DC10WindowRecovery(BUNDLE/'recognition.recovery.json', {step.name: step})
    result = screen_predictions(raw, features, step, EVIDENCE/'visibility_cache', recovery)
    result['raw_label'] = raw.pred_label
    raw.to_csv(out.with_suffix('.raw.csv'), index=False)
    result.to_csv(out, index=False)
    return result

def run_cad(only=None, cad_subdir='cad', force=False):
    manifest = json.loads((BUNDLE/'manifest.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(EVIDENCE/'recognition/predictions.csv')
    official = pd.read_csv(BUNDLE/'evaluation/predictions.csv')
    keys = ['model_name', 'face_id']
    a = predictions.set_index(keys).sort_index()
    b = official.set_index(keys).sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.pred_label, b.pred_label):
        raise ValueError('Fresh predictions differ from the selected formal bundle')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = np.load(BUNDLE/manifest['stats'], allow_pickle=True)
    model = create_bundle_model(stats, manifest).to(device)
    model.load_state_dict(torch.load(BUNDLE/manifest['checkpoint'], map_location=device, weights_only=True))
    model.eval()
    summaries = []
    original_features = pd.read_csv(BUNDLE/manifest['test_csv']).drop(columns=['label'], errors='ignore')
    selected = [name for name in sorted(predictions.model_name.unique()) if only is None or stage_stem(name)==only]
    if not selected:
        raise ValueError('No matching aircraft stem')
    output_root=(EVIDENCE/cad_subdir).resolve()
    if not output_root.is_relative_to(EVIDENCE.resolve()):
        raise ValueError('CAD output must remain inside this experiment')
    for name in selected:
        folder = output_root/stage_stem(name)
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder/'record.json'
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record.get('complete') and not force:
                if record.get('protocol_id') != PROTOCOL_ID and record.get('accepted_protocol_id') != PROTOCOL_ID:
                    raise ValueError('CAD cache uses an obsolete processing protocol')
                verify_completed_record(record, sha256(BUNDLE/manifest['checkpoint']))
                summaries.append(record)
                continue
        source = (BUNDLE/manifest['geometry'][name]['path']).resolve()
        initial = predictions[predictions.model_name == name].copy()
        pred = folder/'original.pred.csv'
        initial.to_csv(pred, index=False)
        source_features=original_features[original_features.model_name==name].reset_index(drop=True)
        initial=initial.reset_index(drop=True)
        original_cad, original_completion=complete_geometric_candidates(source_features,initial)
        original_cad.to_csv(folder/'original.cad_candidates.csv',index=False)
        record = dict(model_name=name, model_id=manifest['model_id'],
            protocol_id=PROTOCOL_ID,
            checkpoint_sha256=sha256(BUNDLE/manifest['checkpoint']), original=geometry(source),
            original_predictions_sha256=sha256(pred),
            rivet_candidates=int((initial.pred_label == 1).sum()),
            original_surface_candidates=int((initial.pred_label == 2).sum()),
            route=verified_route(stage_stem(name)), reconstruction_profile=reconstruction_profile(stage_stem(name)),
            original_geometric_completion=original_completion, detector_sha256=sha256(DETECTOR), operations=[])
        save(record_path, record)
        removed = folder/intermediate_step_name(stage_stem(name))
        if record['rivet_candidates']:
            op = command([DETECTOR, '--remove-predicted-rivets', source, pred, removed], folder/'rivet.log')
            record['operations'].append(op)
            if op['returncode'] or not removed.exists():
                record['rivet_failure'] = True
                shutil.copy2(source, removed)
        else:
            shutil.copy2(source, removed)
        record['rivet_endpoint'] = geometry(removed)
        features = folder/'intermediate_features.csv'
        op = command([DETECTOR, '--predict', removed, features], folder/'export.log')
        record['operations'].append(op)
        if op['returncode']:
            save(record_path, record)
            raise RuntimeError(f'Intermediate export failed: {name}')
        surface_pred = folder/'intermediate.pred.csv'
        surface = infer(removed, features, surface_pred, model, stats, device)
        record['intermediate_predictions_sha256'] = sha256(surface_pred)
        record['intermediate_features_sha256'] = sha256(features)
        record['surface_candidates'] = int((surface.pred_label == 2).sum())
        operator_pred = folder/'surface_candidates.csv'
        geometry_features = pd.read_csv(features).drop(columns=['label'], errors='ignore')
        geometric_candidates, completion=complete_geometric_candidates(geometry_features,surface)
        record['intermediate_geometric_completion']=completion
        candidates, correspondence = transfer_surface_candidates(
            source_features, original_cad, geometry_features, geometric_candidates)
        candidates.to_csv(operator_pred,index=False)
        save(folder/'candidate_correspondence.json',correspondence)
        record['candidate_correspondence'] = correspondence
        record['operator_candidates'] = int((candidates.pred_label == 2).sum())
        record['operator_predictions_sha256'] = sha256(operator_pred)
        candidate, final = folder/'surface_candidate.step', folder/'final.step'
        if record['operator_candidates']:
            flag = {'embedded': '--rebuild-embedded-window-hosts',
                    'invalid-host': '--rebuild-invalid-surface-hosts',
                    'split': '--rebuild-split-window-skins',
                    'generic': '--remove-predicted-surface-features'}[record['route']]
            args = [DETECTOR, flag, removed, operator_pred, candidate]
            if record['route'] == 'embedded':
                args.append(record['reconstruction_profile'])
            op = command(args, folder/'surface.log')
            record['operations'].append(op)
            succeeded = op['returncode'] == 0 and candidate.exists()
            if succeeded and record['route'] == 'split':
                bridged = folder/'bridged.step'
                op = command([sys.executable, ROOT/'python/main.py', 'complete-windows',
                              candidate, bridged, '--detector', DETECTOR,
                              '--source-features',features,'--source-predictions',operator_pred],
                             folder/'bridge.log')
                record['operations'].append(op)
                succeeded = op['returncode'] == 0 and bridged.exists()
                candidate = bridged
            record['surface_operation_succeeded'] = succeeded
            shutil.copy2(candidate if succeeded else removed, final)
        else:
            record['surface_operation_succeeded'] = None
            shutil.copy2(removed, final)
        record['final'] = geometry(final)
        record['complete'] = True
        save(record_path, record)
        summaries.append(record)
        if only is None:
            save(output_root/'summary.json', summaries)
        print('COMPLETE', name, record['original']['faces'], record['final']['faces'], flush=True)
    if only is None:
        save(output_root/'summary.json', summaries)

def run_mesh():
    records = json.loads((EVIDENCE/'cad/summary.json').read_text(encoding='utf-8'))
    for record in records:
        for stage, key in [('original', 'original'), ('simplified', 'final')]:
            output = EVIDENCE/'mesh'/f'{stage_stem(record["model_name"])}_{stage}.json'
            if output.exists():
                previous = json.loads(output.read_text())
                if previous.get('step_sha256') != record[key]['sha256']:
                    raise ValueError('Mesh cache does not match current endpoint')
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            op = command([sys.executable, ROOT/'python/main.py', 'mesh', ROOT/record[key]['path'],
                          '--relative-deflection', '.0005', '--angular-deflection-rad', '.5',
                          '--repeats', '5', '--output', output], output.with_suffix('.log'))
            if op['returncode']:
                raise RuntimeError(f'Mesh failed: {output}')
            data = json.loads(output.read_text(encoding='utf-8'))
            data.update(step_sha256=record[key]['sha256'], checkpoint_sha256=record['checkpoint_sha256'])
            save(output, data)


def main():
    global EVIDENCE, DETECTOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['cad', 'mesh'], required=True)
    parser.add_argument('--only', help='Run one aircraft stem into an isolated CAD subdirectory')
    parser.add_argument('--cad-subdir', default='cad')
    parser.add_argument('--evidence-dir',type=Path,default=EVIDENCE)
    parser.add_argument('--detector',type=Path,default=DETECTOR)
    parser.add_argument('--force',action='store_true',help='Regenerate an explicitly selected CAD case')
    args = parser.parse_args()
    EVIDENCE = args.evidence_dir.resolve()
    DETECTOR = args.detector.resolve()
    if not EVIDENCE.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Evidence directory must remain under results')
    if args.force and not args.only:
        raise ValueError('--force requires one explicit aircraft stem')
    run_cad(args.only, args.cad_subdir,args.force) if args.stage == 'cad' else run_mesh()
