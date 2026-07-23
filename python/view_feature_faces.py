"""Interactively inspect arbitrary face IDs on a STEP model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_face_ids(value: str) -> list[int]:
    face_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not face_ids or any(face_id <= 0 for face_id in face_ids):
        raise argparse.ArgumentTypeError("--face-ids must contain positive face IDs")
    return face_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="View arbitrary STEP face IDs one at a time.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--face-ids", type=parse_face_ids, required=True)
    args = parser.parse_args()

    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_ORANGE
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Pnt
    from OCC.Display.SimpleGui import init_display

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    missing = [face_id for face_id in args.face_ids if face_id > face_map.Size()]
    if missing:
        raise ValueError(f"Face IDs out of range 1..{face_map.Size()}: {missing}")

    display, start_display, add_menu, add_function_to_menu = init_display(size=(1440, 960))
    current = 0

    def show() -> None:
        face_id = args.face_ids[current]
        face = topods.Face(face_map.FindKey(face_id))
        box = Bnd_Box()
        brepbndlib.Add(face, box)
        x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
        point = gp_Pnt((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)
        display.EraseAll()
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.86, update=False)
        display.DisplayShape(face, color=Quantity_NOC_ORANGE, update=False)
        display.DisplayMessage(point, f"F{face_id} ({current + 1}/{len(args.face_ids)})", height=28.0, update=False)
        display.FitAll()
        display.Repaint()
        print(f"Showing F{face_id} ({current + 1}/{len(args.face_ids)})")

    def previous() -> None:
        nonlocal current
        current = (current - 1) % len(args.face_ids)
        show()

    def next_face() -> None:
        nonlocal current
        current = (current + 1) % len(args.face_ids)
        show()

    add_menu("Candidate")
    add_function_to_menu("Candidate", previous)
    add_function_to_menu("Candidate", next_face)
    show()
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
