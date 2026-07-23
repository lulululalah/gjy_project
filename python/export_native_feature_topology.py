"""Export inner-loop adjacency seeds for a manually selected native-feature host face."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export faces adjacent to each inner loop of one host face."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--host-face", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )
    from OCC.Core.TopoDS import topods

    args = parse_args()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    if args.host_face < 1 or args.host_face > face_map.Size():
        raise ValueError(f"--host-face must be within 1..{face_map.Size()}")

    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    host = topods.Face(face_map.FindKey(args.host_face))
    outer_wire = breptools.OuterWire(host)
    rows = []
    wire_explorer = TopExp_Explorer(host, TopAbs_WIRE)
    inner_loop_index = 0
    while wire_explorer.More():
        wire = wire_explorer.Current()
        if not wire.IsSame(outer_wire):
            inner_loop_index += 1
            adjacent_face_ids: set[int] = set()
            edge_count = 0
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            while edge_explorer.More():
                edge_count += 1
                adjacent = edge_faces.FindFromKey(edge_explorer.Current())
                iterator = TopTools_ListIteratorOfListOfShape(adjacent)
                while iterator.More():
                    face_id = face_map.FindIndex(iterator.Value())
                    if face_id and face_id != args.host_face:
                        adjacent_face_ids.add(face_id)
                    iterator.Next()
                edge_explorer.Next()
            areas = []
            for face_id in sorted(adjacent_face_ids):
                properties = GProp_GProps()
                brepgprop.SurfaceProperties(topods.Face(face_map.FindKey(face_id)), properties)
                areas.append(f"F{face_id}:{properties.Mass():.6f}")
            rows.append(
                {
                    "model_name": args.step_model.name,
                    "host_face_id": args.host_face,
                    "inner_loop_index": inner_loop_index,
                    "loop_edge_count": edge_count,
                    "adjacent_face_ids": " ".join(f"F{face_id}" for face_id in sorted(adjacent_face_ids)),
                    "adjacent_face_areas": " ".join(areas),
                }
            )
        wire_explorer.Next()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[
            "model_name", "host_face_id", "inner_loop_index", "loop_edge_count",
            "adjacent_face_ids", "adjacent_face_areas",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} inner-loop rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
