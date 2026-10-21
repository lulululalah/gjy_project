"""CAD case profiles, geometric correspondence and cosmetic candidate construction."""

from __future__ import annotations

import re

from pathlib import Path

def stage_stem(model_name):
    stem=Path(model_name).stem
    stem=re.sub(r'_wing_rivets(?:_decals)?$', '', stem)
    return re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('_')

def verified_route(stem):
    if stem.startswith('Air_Plane_Idea_A'):
        return 'invalid-host'
    if stem.startswith('27-_DC-10'):
        return 'split'
    if stem.startswith(('Airbus','Airplane_body','AULIRA_2','Gulfstream_G280')):
        return 'embedded'
    return 'generic'

def reconstruction_profile(stem):
    """Resolve the operator profile from the source identity, before renaming."""
    for prefix, profile in [('Gulfstream_G280', 'gulfstream-g280'),
                            ('Airplane_body', 'airplane-body'),
                            ('AULIRA_2', 'aulira'), ('Airbus', 'airbus')]:
        if stem.startswith(prefix):
            return profile
    return 'generic'

def intermediate_step_name(stem):
    if Path(stem).name != stem or not stem:
        raise ValueError('An intermediate STEP requires a single source stem')
    return f'{stem}_rivet_removed.step'

import numpy as np


def complete_geometric_candidates(features, predictions):
    """Propose repeated two-edge panes and small single-host decorative patches."""
    if features.id.astype(int).tolist() != predictions.face_id.astype(int).tolist():
        raise ValueError('Geometry completion requires aligned face identifiers')
    frame = features.reset_index(drop=True)
    result = predictions.copy()
    result['network_pred_label'] = predictions.pred_label.astype(int)
    result['cad_candidate_reason'] = np.where(predictions.pred_label == 2,'current_network','')
    # Weak neural evidence is used only with an explicit geometric attachment.
    gate = predictions.prob_surface_feature.to_numpy(dtype=float) >= .1
    host = frame.neighborAreaMax.to_numpy(dtype=float) >= frame.area.to_numpy(dtype=float)*4
    small = frame.relativeArea.to_numpy(dtype=float) <= .001
    panes = gate & host & small & (frame.surfaceType.to_numpy()==6) & (frame.numEdges.to_numpy()==2)
    panes &= (frame.smoothEdgeCount.to_numpy()>=1) & (frame.innerWireCount.to_numpy()==0)
    ordered=sorted(np.flatnonzero(panes), key=lambda i:float(frame.area.iloc[i]))
    repeated=set();group=[]
    for index in ordered:
        if group and float(frame.area.iloc[index]) > float(frame.area.iloc[group[0]])*1.25:
            if len(group)>=8:
                repeated.update(group)
            group=[]
        group.append(int(index))
    if len(group)>=8:
        repeated.update(group)
    neighbor_counts=frame.neighbors.fillna('').astype(str).map(lambda s:len(set(s.split()))).to_numpy()
    decals=(gate & host & small & (frame.surfaceType.to_numpy()==6)
        & (frame.numEdges.to_numpy()>=6) & (frame.numWires.to_numpy()==1)
        & (frame.smoothEdgeCount.to_numpy()==1) & (neighbor_counts==1)
        & (frame.normalNeighborDotMin.to_numpy(dtype=float)>=.99))
    chosen={i:'repeated_two_edge_pane' for i in repeated}
    chosen.update({int(i):'single_host_decorative_patch' for i in np.flatnonzero(decals)})
    added=[]
    for index,reason in chosen.items():
        row=result.index[index]
        if int(result.loc[row,'pred_label'])==1:
            continue
        if int(result.loc[row,'pred_label'])!=2:
            result.loc[row,['pred_label','pred_name']]=[2,'surface_feature']
            result.loc[row,'cad_candidate_reason']=reason
            added.append(int(result.loc[row,'face_id']))
    report=dict(network_candidates=int((predictions.pred_label==2).sum()),
        cad_candidates=int((result.pred_label==2).sum()),added_face_ids=added,
        repeated_panes=len(repeated),single_host_patches=int(decals.sum()),
        scope='CAD candidates only; unchanged neural probabilities, no labels or stored target face lists.')
    return result,report

def transfer_surface_candidates(original_features, original_predictions,
                                current_features, current_predictions):
    required = ['id', 'area', 'perimeter', 'surfaceType', 'centerX', 'centerY', 'centerZ']
    first = original_features[required].set_index('id', verify_integrity=True)
    current = current_features[required].set_index('id', verify_integrity=True)
    if current.index.astype(int).tolist() != current_predictions.face_id.astype(int).tolist():
        raise ValueError('Current predictions and geometry are not aligned')
    if set(first.index.astype(int)) != set(original_predictions.face_id.astype(int)):
        raise ValueError('Original predictions and geometry are not aligned')
    source_ids = original_predictions.loc[original_predictions.pred_label == 2, 'face_id'].astype(int)
    result = current_predictions.copy()
    result['network_pred_label'] = result.get('network_pred_label',result.pred_label).astype(int)
    result['cad_source_face_ids'] = ''
    if 'cad_candidate_reason' not in result:
        result['cad_candidate_reason'] = np.where(result.pred_label == 2, 'current_network', '')
    xyz = current[['centerX','centerY','centerZ']].to_numpy(dtype=float)
    area = current.area.to_numpy(dtype=float)
    perimeter = current.perimeter.to_numpy(dtype=float)
    kind = current.surfaceType.to_numpy(dtype=int)
    values = np.column_stack([xyz,area,perimeter])
    if not np.isfinite(values).all():
        raise ValueError('Correspondence requires finite geometry')
    extent = max(float(np.ptp(xyz, axis=0).max()), 1.0) if len(xyz) else 1.0
    coordinate_tolerance = max(1e-5 * extent, 1e-5)
    matches = {}
    for face_id in source_ids:
        face = first.loc[face_id]
        center = face[['centerX','centerY','centerZ']].to_numpy(dtype=float)
        distance = np.linalg.norm(xyz-center,axis=1)
        eligible = ((distance <= coordinate_tolerance)
            & (np.abs(area-float(face.area)) <= max(abs(float(face.area))*1e-4,1e-8))
            & (np.abs(perimeter-float(face.perimeter)) <= max(abs(float(face.perimeter))*5e-4,1e-7))
            & (kind == int(face.surfaceType)))
        for index in np.flatnonzero(eligible):
            matches.setdefault(int(index), []).append(int(face_id))
    for index, source in matches.items():
        row = result.index[index]
        result.loc[row,'cad_source_face_ids'] = ' '.join(map(str,source))
        previous=str(result.loc[row,'cad_candidate_reason'])
        result.loc[row,'cad_candidate_reason'] = (previous+'+' if previous else '')+'geometry_correspondence'
        result.loc[row,['pred_label','pred_name']] = [2,'surface_feature']
    # This file is for the cosmetic stage only. Rivet labels remain preserved in
    # network_pred_label and the unmodified inference CSV.
    result.loc[result.pred_label == 1,['pred_label','pred_name']] = [0,'background']
    protected = ((current_features.innerWireCount.to_numpy(dtype=int)>=8)
        & (current_features.relativeArea.to_numpy(dtype=float)>=.01)
        & (result.pred_label.to_numpy(dtype=int)==2)) if {'innerWireCount','relativeArea'}.issubset(current_features.columns) else np.zeros(len(result),dtype=bool)
    # A perforated carrier is the reconstruction host, not one of its panes.
    # Keep it available so the native operator can fill only matched inner loops.
    result.loc[protected,['pred_label','pred_name']] = [0,'background']
    result.loc[protected,'cad_candidate_reason'] = 'protected_perforated_host'
    report = dict(original_candidates=len(source_ids), current_network_candidates=int((result.network_pred_label==2).sum()),
        geometrically_matched_original_faces=len({s for ids in matches.values() for s in ids}),
        matched_current_faces=len(matches), cad_candidates=int((result.pred_label==2).sum()),
        added_candidates=int(((result.pred_label==2)&(result.network_pred_label!=2)).sum()),
        protected_host_faces=result.loc[protected,'face_id'].astype(int).tolist(),
        coordinate_tolerance=coordinate_tolerance,
        scope='Geometry correspondence only; no truth labels or reused face identifiers.')
    return result, report
