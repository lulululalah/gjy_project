"""Interactively choose one smooth solid face for a star decal."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_shape(step_path: Path):
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def face_area(face) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, properties)
    return properties.Mass()


def face_center(face):
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt

    box = Bnd_Box()
    brepbndlib.Add(face, box)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    return gp_Pnt((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)


def model_center_and_diagonal(shape):
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    center = gp_Pnt((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)
    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5
    return center, max(diagonal, 1.0)


def solid_owned_face_ids(shape) -> set[int]:
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    owned_ids: set[int] = set()
    solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while solid_explorer.More():
        face_explorer = TopExp_Explorer(solid_explorer.Current(), TopAbs_FACE)
        while face_explorer.More():
            face_id = face_map.FindIndex(face_explorer.Current())
            if face_id:
                owned_ids.add(face_id)
            face_explorer.Next()
        solid_explorer.Next()
    return owned_ids


def candidate_faces(shape, max_candidates: int):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Cylinder, GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods

    surface_names = {
        GeomAbs_BSplineSurface: "bspline",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Plane: "plane",
    }
    surface_weights = {
        GeomAbs_BSplineSurface: 5.0,
        GeomAbs_Cylinder: 4.0,
        GeomAbs_Plane: 1.0,
    }
    owned_ids = solid_owned_face_ids(shape)
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    candidates = []
    for face_id in range(1, face_map.Size() + 1):
        if face_id not in owned_ids:
            continue
        face = topods.Face(face_map.FindKey(face_id))
        surface_type = BRepAdaptor_Surface(face).GetType()
        if surface_type not in surface_names:
            continue
        area = face_area(face)
        if area < 2.0:
            continue
        score = area * surface_weights[surface_type]
        candidates.append((score, face_id, face, surface_names[surface_type], area))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[:max_candidates]


def write_candidates(candidates, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["face_id", "surface", "area", "score"])
        for score, face_id, _, surface, area in candidates:
            writer.writerow([face_id, surface, f"{area:.6f}", f"{score:.6f}"])


def display_candidates(shape, candidates) -> None:
    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_ORANGE
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.gp import gp_Dir, gp_Vec
    from OCC.Display.SimpleGui import init_display

    display, start_display, add_menu, add_function_to_menu = init_display(size=(1440, 960))
    model_center, diagonal = model_center_and_diagonal(shape)

    current_index = 0

    def show_current_candidate() -> None:
        score, face_id, face, surface, area = candidates[current_index]
        display.EraseAll()
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.86, update=False)
        display.DisplayShape(face, color=Quantity_NOC_ORANGE, update=False)
        center = face_center(face)
        offset = gp_Vec(model_center, center)
        if offset.SquareMagnitude() < 1.0e-12:
            offset = gp_Vec(0.0, 0.0, 1.0)
        direction = gp_Dir(offset)
        label_point = center.Translated(gp_Vec(direction) * (diagonal * 0.075))
        display.DisplayMessage(
            label_point,
            f"F{face_id}  ({current_index + 1}/{len(candidates)})",
            height=28.0,
            message_color=(0.0, 0.0, 0.0),
            update=False,
        )
        display.Repaint()
        print(f"Showing F{face_id} ({current_index + 1}/{len(candidates)}): {surface}, area={area:.3f}, score={score:.3f}")

    def previous_candidate() -> None:
        nonlocal current_index
        current_index = (current_index - 1) % len(candidates)
        show_current_candidate()

    def next_candidate() -> None:
        nonlocal current_index
        current_index = (current_index + 1) % len(candidates)
        show_current_candidate()

    add_menu("Candidate")
    add_function_to_menu("Candidate", previous_candidate)
    add_function_to_menu("Candidate", next_candidate)
    show_current_candidate()
    display.FitAll()
    print("One candidate is shown at a time. Use Candidate > previous/next candidate to browse.")
    print("Tell Codex the displayed F-number; Codex will generate the corresponding decal model.")
    start_display()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and select a single star-decal host face.")
    parser.add_argument("step_model", type=Path, help="A *_wing_rivets.step model from data/plane_model/new_data.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Maximum highlighted faces (default: 12).")
    parser.add_argument(
        "--face-ids",
        help="Comma-separated eligible face IDs to browse instead of the top-ranked candidates.",
    )
    parser.add_argument("--no-display", action="store_true", help="Only export the candidate CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    shape = load_shape(args.step_model)
    if args.face_ids:
        requested_ids = [int(value.strip()) for value in args.face_ids.split(",") if value.strip()]
        if not requested_ids:
            raise ValueError("--face-ids must include at least one face ID")
        eligible_by_id = {
            face_id: (score, face, surface, area)
            for score, face_id, face, surface, area in candidate_faces(shape, 1_000_000)
        }
        missing_ids = [face_id for face_id in requested_ids if face_id not in eligible_by_id]
        if missing_ids:
            missing_text = ", ".join(f"F{face_id}" for face_id in missing_ids)
            raise ValueError(f"Requested face IDs are not eligible smooth solid faces: {missing_text}")
        candidates = [
            (score, face_id, face, surface, area)
            for face_id in requested_ids
            for score, face, surface, area in [eligible_by_id[face_id]]
        ]
    else:
        candidates = candidate_faces(shape, args.max_candidates)
    if not candidates:
        raise RuntimeError("No eligible smooth faces belonging to solids were found.")
    candidates_path = args.step_model.with_suffix(".decal_face_candidates.csv")
    write_candidates(candidates, candidates_path)
    print(f"Candidate list: {candidates_path}")
    print("Candidates:")
    for _, face_id, _, surface, area in candidates:
        print(f"  F{face_id}: {surface}, area={area:.3f}")
    if not args.no_display:
        display_candidates(shape, candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
