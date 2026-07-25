"""Export repeated isolated-face window candidates from confirmed seed faces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find repeated isolated window faces matching one or more confirmed seeds."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument(
        "--reference-face-ids",
        required=True,
        help="Comma-separated confirmed isolated window face IDs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.65)
    parser.add_argument("--max-area-ratio", type=float, default=1.55)
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )
    from OCC.Core.TopoDS import topods

    args = parse_args()
    seed_ids = sorted({int(value.strip()) for value in args.reference_face_ids.split(",") if value.strip()})
    if not seed_ids:
        raise ValueError("Provide at least one --reference-face-ids value")
    if not 0.0 < args.min_area_ratio <= args.max_area_ratio:
        raise ValueError("Area-ratio bounds must satisfy 0 < min <= max")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    if any(face_id < 1 or face_id > face_map.Size() for face_id in seed_ids):
        raise ValueError(f"Seed IDs must be within 1..{face_map.Size()}")

    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)

    def face_area(face_id: int) -> float:
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(topods.Face(face_map.FindKey(face_id)), properties)
        return properties.Mass()

    def edge_count(face_id: int) -> int:
        explorer = TopExp_Explorer(topods.Face(face_map.FindKey(face_id)), TopAbs_EDGE)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    def surface_type(face_id: int) -> int:
        return int(BRepAdaptor_Surface(topods.Face(face_map.FindKey(face_id))).GetType())

    def has_adjacent_faces(face_id: int) -> bool:
        face = topods.Face(face_map.FindKey(face_id))
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            adjacent = edge_faces.FindFromKey(explorer.Current())
            iterator = TopTools_ListIteratorOfListOfShape(adjacent)
            while iterator.More():
                if face_map.FindIndex(iterator.Value()) not in (0, face_id):
                    return True
                iterator.Next()
            explorer.Next()
        return False

    seed_signatures = []
    for seed_id in seed_ids:
        if has_adjacent_faces(seed_id):
            raise RuntimeError(f"Seed face F{seed_id} is not isolated")
        seed_signatures.append((seed_id, surface_type(seed_id), edge_count(seed_id), face_area(seed_id)))

    rows: list[dict[str, object]] = []
    for face_id in range(1, face_map.Size() + 1):
        if has_adjacent_faces(face_id):
            continue
        candidate_type = surface_type(face_id)
        candidate_edges = edge_count(face_id)
        candidate_area = face_area(face_id)
        matching_seed = next(
            (
                seed_id
                for seed_id, seed_type, seed_edges, seed_area in seed_signatures
                if candidate_type == seed_type
                and candidate_edges == seed_edges
                and args.min_area_ratio <= candidate_area / seed_area <= args.max_area_ratio
            ),
            None,
        )
        if matching_seed is None:
            continue
        rows.append(
            {
                "model_name": args.step_model.name,
                "candidate_id": len(rows) + 1,
                "host_face_id": face_id,
                "host_area": f"{candidate_area:.6f}",
                "host_center": "",
                "inner_loop_index": f"isolated_seed_F{matching_seed}",
                "loop_edge_count": candidate_edges,
                "candidate_face_ids": f"F{face_id}",
                "candidate_face_areas": f"{candidate_area:.6f}",
                "candidate_face_centers": "",
                "relative_area": "1.000000",
                "candidate_surface_types": str(candidate_type),
                "heuristic_score": "isolated_seed",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "model_name", "candidate_id", "host_face_id", "host_area", "host_center",
        "inner_loop_index", "loop_edge_count", "candidate_face_ids", "candidate_face_areas",
        "candidate_face_centers", "relative_area", "candidate_surface_types", "heuristic_score",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} isolated-window candidates from seeds {seed_ids}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
