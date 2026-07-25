"""Export window candidates formed by two co-located, repeated face templates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find repeated two-face window pairs from confirmed seed faces.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--primary-face-id", type=int, required=True)
    parser.add_argument("--secondary-face-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--center-tolerance", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods

    args = parse_args()
    if not 0.0 <= args.area_ratio_tolerance < 1.0 or args.center_tolerance <= 0.0:
        raise ValueError("Invalid area-ratio or center tolerance")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    def describe(face_id: int) -> tuple[float, int, int, tuple[float, float, float]]:
        face = topods.Face(face_map.FindKey(face_id))
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        center = properties.CentreOfMass()
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        edge_count = 0
        while explorer.More():
            edge_count += 1
            explorer.Next()
        return (
            properties.Mass(),
            int(BRepAdaptor_Surface(face).GetType()),
            edge_count,
            (center.X(), center.Y(), center.Z()),
        )

    for face_id in (args.primary_face_id, args.secondary_face_id):
        if not 1 <= face_id <= face_map.Size():
            raise ValueError(f"Face ID outside 1..{face_map.Size()}: F{face_id}")
    primary_seed = describe(args.primary_face_id)
    secondary_seed = describe(args.secondary_face_id)
    primary: list[tuple[int, float, tuple[float, float, float]]] = []
    secondary: list[tuple[int, float, tuple[float, float, float]]] = []
    for face_id in range(1, face_map.Size() + 1):
        area, surface_type, edge_count, center = describe(face_id)
        if (
            surface_type == primary_seed[1]
            and edge_count == primary_seed[2]
            and abs(area / primary_seed[0] - 1.0) <= args.area_ratio_tolerance
        ):
            primary.append((face_id, area, center))
        if (
            surface_type == secondary_seed[1]
            and edge_count == secondary_seed[2]
            and abs(area / secondary_seed[0] - 1.0) <= args.area_ratio_tolerance
        ):
            secondary.append((face_id, area, center))

    rows: list[dict[str, object]] = []
    for secondary_id, secondary_area, secondary_center in secondary:
        matches = [
            (face_id, area, center)
            for face_id, area, center in primary
            if math.dist(center, secondary_center) <= args.center_tolerance
        ]
        if not matches:
            continue
        primary_id, primary_area, primary_center = min(matches, key=lambda row: row[0])
        rows.append(
            {
                "model_name": args.step_model.name,
                "candidate_id": len(rows) + 1,
                "host_face_id": "",
                "host_area": "",
                "host_center": "",
                "inner_loop_index": "paired_face",
                "loop_edge_count": primary_seed[2],
                "candidate_face_ids": f"F{primary_id} F{secondary_id}",
                "candidate_face_areas": f"{primary_area:.6f} {secondary_area:.6f}",
                "candidate_face_centers": ";".join(
                    " ".join(f"{value:.6f}" for value in center)
                    for center in (primary_center, secondary_center)
                ),
                "relative_area": "",
                "candidate_surface_types": f"{primary_seed[1]} {secondary_seed[1]}",
                "heuristic_score": "paired_face",
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
    print(f"Exported {len(rows)} paired-face candidates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
