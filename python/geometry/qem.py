"""Measure matched-target QEM components for the completed CAD endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(ROOT/'work/qem_baseline/python310_packages'))
import numpy as np
import fast_simplification
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods

from geometry.measure import load_step, bounding_box_diagonal
from cad.candidates import stage_stem
from recognition.predict import sha256

EVIDENCE = ROOT/'results/current_three_class_complete_20261020'

def extract(step):
    shape = load_step(step)
    mesh = BRepMesh_IncrementalMesh(shape, bounding_box_diagonal(shape)*.0005, False, .5, True)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError('Fresh original QEM triangulation failed')
    vertices, faces, lookup = [], [], {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, location)
        if tri:
            ids = {}
            for i in range(1, tri.NbNodes()+1):
                point = tri.Node(i).Transformed(location.Transformation())
                value = point.Coord()
                key = tuple(round(x, 9) for x in value)
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append(value)
                ids[i] = lookup[key]
            for i in range(1, tri.NbTriangles()+1):
                item = [ids[x] for x in tri.Triangle(i).Get()]
                if face.Orientation() == TopAbs_REVERSED:
                    item.reverse()
                if len(set(item)) == 3:
                    faces.append(item)
        explorer.Next()
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)

def measure_qem():
    records = json.loads((EVIDENCE/'cad/summary.json').read_text(encoding='utf-8'))
    for record in records:
        stem = stage_stem(record['model_name'])
        folder = EVIDENCE/'qem'/stem
        folder.mkdir(parents=True, exist_ok=True)
        output = folder/'measurement.json'
        if output.exists():
            previous=json.loads(output.read_text(encoding='utf-8'))
            if previous['input_step_sha256']!=record['original']['sha256'] or previous['final_step_sha256']!=record['final']['sha256']:
                raise ValueError('QEM cache belongs to another CAD endpoint')
            continue
        original = json.loads((EVIDENCE/'mesh'/f'{stem}_original.json').read_text())
        final = json.loads((EVIDENCE/'mesh'/f'{stem}_simplified.json').read_text())
        step = ROOT/record['original']['path']
        if sha256(step) != original['step_sha256']:
            raise ValueError('Original QEM STEP differs from meshing input')
        vertices, triangles = extract(step)
        np.savez_compressed(folder/'source_mesh.npz', vertices=vertices, triangles=triangles)
        target = final['mesh_triangle_count']
        samples = []
        # Edge collapse cannot increase a mesh. Record an unmatched control
        # rather than inventing a matched target when CAD processing adds faces.
        matched = target < len(triangles)
        if matched:
            for _ in range(5):
                start = time.perf_counter()
                points, faces = fast_simplification.simplify(vertices, triangles,
                    target_count=target, agg=7.0, preserve_border=False)
                samples.append(time.perf_counter()-start)
        else:
            points, faces = vertices, triangles
        np.savez_compressed(folder/'qem_mesh.npz', vertices=points, triangles=faces)
        median = statistics.median(samples) if samples else 0.0
        baseline = original['mesh_elapsed_seconds_median']+median
        data = dict(model_name=record['model_name'], checkpoint_sha256=record['checkpoint_sha256'],
            input_step_sha256=record['original']['sha256'], final_step_sha256=record['final']['sha256'],
            source_triangles=len(triangles), requested_target=target, qem_triangles=len(faces),
            matched_target=matched and abs(len(faces)-target)<=1,
            qem_version=fast_simplification.__version__, aggregation=7., preserve_border=False,
            stitching_decimal_places=9, repeats=len(samples), samples=samples, qem_median_seconds=median,
            original_meshing_seconds=original['mesh_elapsed_seconds_median'],
            final_meshing_seconds=final['mesh_elapsed_seconds_median'], baseline_seconds=baseline,
            relative_difference_percent=100*(baseline-final['mesh_elapsed_seconds_median'])/baseline,
            source_mesh_sha256=sha256(folder/'source_mesh.npz'),
            scope='Meshing and QEM kernel components only; excludes extraction, I/O, inference, CAD operations.')
        output.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(stem, len(triangles), '->', len(faces), 'target', target, flush=True)


def main():
    global EVIDENCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,default=EVIDENCE)
    args=parser.parse_args()
    EVIDENCE=args.evidence_dir.resolve()
    if not EVIDENCE.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Evidence directory must remain under results')
    measure_qem()
