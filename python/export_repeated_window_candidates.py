"""Export repeated window candidates from one manually confirmed seed window face."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find repeated window components matching a confirmed seed face."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--reference-face-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.65)
    parser.add_argument("--max-area-ratio", type=float, default=1.55)
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
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
    if not 0.0 < args.min_area_ratio <= args.max_area_ratio:
        raise ValueError("Area-ratio bounds must satisfy 0 < min <= max")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    if not 1 <= args.reference_face_id <= face_map.Size():
        raise ValueError(f"--reference-face-id must be within 1..{face_map.Size()}")

    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)

    def face_area(face_id: int) -> float:
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(topods.Face(face_map.FindKey(face_id)), properties)
        return properties.Mass()

    def face_center(face_id: int) -> tuple[float, float, float]:
        box = Bnd_Box()
        brepbndlib.Add(topods.Face(face_map.FindKey(face_id)), box)
        bounds = box.Get()
        return tuple((bounds[index] + bounds[index + 3]) * 0.5 for index in range(3))

    def edge_count(face_id: int) -> int:
        explorer = TopExp_Explorer(topods.Face(face_map.FindKey(face_id)), TopAbs_EDGE)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    def surface_type(face_id: int) -> int:
        return int(BRepAdaptor_Surface(topods.Face(face_map.FindKey(face_id))).GetType())

    def adjacent_face_ids(face_id: int) -> tuple[int, ...]:
        face = topods.Face(face_map.FindKey(face_id))
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        neighbors: set[int] = set()
        while explorer.More():
            adjacent = edge_faces.FindFromKey(explorer.Current())
            iterator = TopTools_ListIteratorOfListOfShape(adjacent)
            while iterator.More():
                neighbor_id = face_map.FindIndex(iterator.Value())
                if neighbor_id and neighbor_id != face_id:
                    neighbors.add(neighbor_id)
                iterator.Next()
            explorer.Next()
        return tuple(sorted(neighbors))

    seed_id = args.reference_face_id
    seed_area = face_area(seed_id)
    seed_type = surface_type(seed_id)
    seed_edge_count = edge_count(seed_id)
    seed_neighbors = adjacent_face_ids(seed_id)
    if not seed_neighbors:
        raise RuntimeError(f"Reference face F{seed_id} has no adjacent component faces")

    groups: dict[tuple[int, ...], int] = {}
    for face_id in range(1, face_map.Size() + 1):
        if surface_type(face_id) != seed_type or edge_count(face_id) != seed_edge_count:
            continue
        area_ratio = face_area(face_id) / seed_area
        if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
            continue
        neighbors = adjacent_face_ids(face_id)
        if len(neighbors) != len(seed_neighbors):
            continue
        groups.setdefault(neighbors, face_id)

    rows: list[dict[str, object]] = []
    for candidate_id, (neighbor_ids, host_face_id) in enumerate(sorted(groups.items()), start=1):
        neighbor_areas = [face_area(face_id) for face_id in neighbor_ids]
        neighbor_centers = [face_center(face_id) for face_id in neighbor_ids]
        rows.append(
            {
                "model_name": args.step_model.name,
                "candidate_id": candidate_id,
                "host_face_id": host_face_id,
                "host_area": f"{face_area(host_face_id):.6f}",
                "host_center": " ".join(f"{value:.6f}" for value in face_center(host_face_id)),
                "inner_loop_index": "seed_component",
                "loop_edge_count": seed_edge_count,
                "candidate_face_ids": " ".join(f"F{face_id}" for face_id in neighbor_ids),
                "candidate_face_areas": " ".join(f"{area:.6f}" for area in neighbor_areas),
                "candidate_face_centers": ";".join(
                    " ".join(f"{value:.6f}" for value in center) for center in neighbor_centers
                ),
                "relative_area": f"{sum(neighbor_areas) / face_area(host_face_id):.6f}",
                "candidate_surface_types": " ".join(str(surface_type(face_id)) for face_id in neighbor_ids),
                "heuristic_score": "seed_topology",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "candidate_id", "host_face_id", "host_area", "host_center", "inner_loop_index",
        "loop_edge_count", "candidate_face_ids", "candidate_face_areas", "candidate_face_centers",
        "relative_area", "candidate_surface_types", "heuristic_score",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(rows)} repeated-window candidates from F{seed_id}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
