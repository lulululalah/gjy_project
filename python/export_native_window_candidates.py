"""Export topology-based native-window candidates without changing labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export inner-loop window candidates and their adjacent face groups."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-host-area", type=float, default=0.0)
    parser.add_argument("--max-loop-edges", type=int, default=16)
    parser.add_argument("--max-adjacent-faces", type=int, default=16)
    parser.add_argument("--max-relative-area", type=float, default=0.35)
    parser.add_argument("--max-single-relative-area", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument(
        "--seed-topology-mode",
        action="store_true",
        help="Keep all seed-like inner-loop window topologies instead of ranking only the top candidates.",
    )
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
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
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)

    def face_area(face_id: int) -> float:
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(topods.Face(face_map.FindKey(face_id)), properties)
        return properties.Mass()

    def face_center(face_id: int) -> tuple[float, float, float]:
        box = Bnd_Box()
        brepbndlib.Add(topods.Face(face_map.FindKey(face_id)), box)
        x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
        return ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)

    rows: list[dict[str, object]] = []
    candidate_id = 0
    for host_face_id in range(1, face_map.Size() + 1):
        host_face = topods.Face(face_map.FindKey(host_face_id))
        host_area = face_area(host_face_id)
        if host_area < args.min_host_area:
            continue
        outer_wire = breptools.OuterWire(host_face)
        wire_explorer = TopExp_Explorer(host_face, TopAbs_WIRE)
        inner_loop_index = 0
        while wire_explorer.More():
            wire = wire_explorer.Current()
            if wire.IsSame(outer_wire):
                wire_explorer.Next()
                continue
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
                    if face_id and face_id != host_face_id:
                        adjacent_face_ids.add(face_id)
                    iterator.Next()
                edge_explorer.Next()
            if not adjacent_face_ids:
                wire_explorer.Next()
                continue
            candidate_id += 1
            adjacent_areas = [face_area(face_id) for face_id in sorted(adjacent_face_ids)]
            adjacent_centers = [face_center(face_id) for face_id in sorted(adjacent_face_ids)]
            total_adjacent_area = sum(adjacent_areas)
            relative_area = total_adjacent_area / host_area
            max_single_relative_area = max(adjacent_areas) / host_area
            if (
                edge_count > args.max_loop_edges
                or len(adjacent_face_ids) > args.max_adjacent_faces
                or total_adjacent_area > host_area * args.max_relative_area
                or max(adjacent_areas) > host_area * args.max_single_relative_area
            ):
                wire_explorer.Next()
                continue
            adjacent_types = [
                BRepAdaptor_Surface(topods.Face(face_map.FindKey(face_id))).GetType()
                for face_id in sorted(adjacent_face_ids)
            ]
            # Openings with several small adjacent faces are more window-like than one large panel.
            score = edge_count * 2.0 + len(adjacent_face_ids) * 3.0 + (1.0 - relative_area) * 20.0
            if args.seed_topology_mode:
                # Approved 737 cabin windows have a four-edge inner loop with four
                # tiny adjacent trim faces.  Front/rear cabin windows use the same
                # topology but were previously lost by the global top-60 cutoff.
                standard_window = (
                    edge_count == 4
                    and len(adjacent_face_ids) == 4
                    and 20.0 <= host_area <= 70.0
                    and relative_area <= 0.15
                    and max_single_relative_area <= 0.05
                )
                if not standard_window:
                    wire_explorer.Next()
                    continue
            rows.append(
                {
                    "model_name": args.step_model.name,
                    "candidate_id": candidate_id,
                    "host_face_id": host_face_id,
                    "host_area": f"{host_area:.6f}",
                    "host_center": " ".join(f"{value:.6f}" for value in face_center(host_face_id)),
                    "inner_loop_index": inner_loop_index,
                    "loop_edge_count": edge_count,
                    "candidate_face_ids": " ".join(f"F{face_id}" for face_id in sorted(adjacent_face_ids)),
                    "candidate_face_areas": " ".join(f"{area:.6f}" for area in adjacent_areas),
                    "candidate_face_centers": ";".join(
                        " ".join(f"{value:.6f}" for value in center) for center in adjacent_centers
                    ),
                    "relative_area": f"{relative_area:.6f}",
                    "candidate_surface_types": " ".join(str(surface_type) for surface_type in adjacent_types),
                    "heuristic_score": f"{score:.3f}",
                }
            )
            wire_explorer.Next()

    if args.seed_topology_mode:
        rows.sort(key=lambda row: int(row["host_face_id"]))
    else:
        rows.sort(key=lambda row: float(row["heuristic_score"]), reverse=True)
        rows = rows[: args.max_candidates]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name", "candidate_id", "host_face_id", "host_area", "host_center", "inner_loop_index",
        "loop_edge_count", "candidate_face_ids", "candidate_face_areas", "candidate_face_centers",
        "relative_area", "candidate_surface_types", "heuristic_score",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} window candidates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
