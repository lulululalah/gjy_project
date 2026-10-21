"""Complete supported window patches and optional boundary seams."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
    TopTools_ListIteratorOfListOfShape,
)
from OCC.Core.TopoDS import topods

def _edges(shape):
    result = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result

def supported_window_location(area, center, candidate_geometry):
    if not candidate_geometry or area<=0:
        return False
    radius=.75*math.sqrt(area)
    return any(area*.01<=seed_area<=area*2 and math.dist(center,seed_center)<=radius
               for seed_area,seed_center in candidate_geometry)

def find_next_window(step_path: Path, candidate_geometry=None) -> tuple[int, float] | None:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    faces = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, faces)
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)

    for face_id in range(1, faces.Size() + 1):
        face = topods.Face(faces.FindKey(face_id))
        face_edges = _edges(face)
        if len(face_edges) != 2:
            continue

        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        area = properties.Mass()
        if not 15.0 <= area <= 25.0:
            continue

        hosts = []
        for edge in face_edges:
            ancestor_index = edge_faces.FindIndex(edge)
            if ancestor_index <= 0:
                continue
            iterator = TopTools_ListIteratorOfListOfShape(
                edge_faces.FindFromIndex(ancestor_index)
            )
            while iterator.More():
                ancestor = iterator.Value()
                if not ancestor.IsSame(face) and not any(
                    ancestor.IsSame(host) for host in hosts
                ):
                    hosts.append(ancestor)
                iterator.Next()
        if len(hosts) != 2 or hosts[0].IsSame(hosts[1]):
            continue

        second_edges = _edges(hosts[1])
        shared_count = sum(
            any(edge.IsSame(other) for other in second_edges)
            for edge in _edges(hosts[0])
        )
        if shared_count >= 4 or supported_window_location(area,properties.CentreOfMass().Coord(),candidate_geometry):
            return face_id, area
    return None

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_step", type=Path)
    parser.add_argument("output_step", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument('--source-features',type=Path)
    parser.add_argument('--source-predictions',type=Path)
    parser.add_argument('--source-step',type=Path)
    args = parser.parse_args()
    if bool(args.source_features)!=bool(args.source_predictions):
        parser.error('Source features and predictions must be supplied together')
    candidate_geometry=[]
    if args.source_features:
        import pandas as pd
        features=pd.read_csv(args.source_features).set_index('id',verify_integrity=True)
        predictions=pd.read_csv(args.source_predictions)
        for face_id in predictions.loc[predictions.pred_label==2,'face_id'].astype(int):
            row=features.loc[face_id]
            candidate_geometry.append((float(row.area),(float(row.centerX),float(row.centerY),float(row.centerZ))))

    args.output_step.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [
        args.output_step.with_suffix(".bridge-a.tmp.step"),
        args.output_step.with_suffix(".bridge-b.tmp.step"),
    ]
    current = args.input_step.resolve()
    completed = 0
    try:
        while True:
            candidate = find_next_window(current,candidate_geometry)
            if candidate is None:
                break
            face_id, area = candidate
            target = temporary_paths[completed % 2].resolve()
            if target.exists():
                target.unlink()
            print(
                f"[{completed + 1}] bridge F{face_id}, area={area:.3f}",
                flush=True,
            )
            result = subprocess.run(
                [
                    str(args.detector.resolve()),
                    "--bridge-split-window-face",
                    str(current),
                    str(face_id),
                    str(target),
                ],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Bridge failed at F{face_id}")
            current = target
            completed += 1

        if args.source_step:
            if not args.source_predictions:
                raise ValueError('Seam completion requires source predictions')
            result=subprocess.run([str(args.detector.resolve()),'--complete-split-window-seams',
                str(current),str(args.source_step.resolve()),str(args.source_predictions.resolve()),str(args.output_step.resolve())],check=False)
            if result.returncode:
                raise RuntimeError('Candidate-supported skin seam completion failed')
        else:
            shutil.copy2(current, args.output_step)
        print(f"Completed bridges: {completed}", flush=True)
        print(f"Output: {args.output_step}", flush=True)
        return 0
    finally:
        for path in temporary_paths:
            if path.exists() and path.resolve() != args.output_step.resolve():
                path.unlink()
