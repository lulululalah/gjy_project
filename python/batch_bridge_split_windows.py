"""Batch the validated two-edge window bridge operation.

Only small two-edge faces between two already-adjacent skin faces are selected.
The C++ remover validates BRep legality and free-edge count after every bridge.
"""

from __future__ import annotations

import argparse
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


def find_next_window(step_path: Path) -> tuple[int, float] | None:
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
        if shared_count >= 4:
            return face_id, area
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_step", type=Path)
    parser.add_argument("output_step", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    args = parser.parse_args()

    args.output_step.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [
        args.output_step.with_suffix(".bridge-a.tmp.step"),
        args.output_step.with_suffix(".bridge-b.tmp.step"),
    ]
    current = args.input_step.resolve()
    completed = 0
    try:
        while True:
            candidate = find_next_window(current)
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

        shutil.copy2(current, args.output_step)
        print(f"Completed bridges: {completed}", flush=True)
        print(f"Output: {args.output_step}", flush=True)
        return 0
    finally:
        for path in temporary_paths:
            if path.exists() and path.resolve() != args.output_step.resolve():
                path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
