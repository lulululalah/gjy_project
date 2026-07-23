"""Visualize all faces of one semantic label from a full-face labels.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Highlight labeled STEP faces for manual verification.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--semantic", default="decal", choices=["rivet", "decal", "window"])
    args = parser.parse_args()

    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_ORANGE
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods
    from OCC.Display.SimpleGui import init_display

    document = json.loads(args.labels_json.read_text(encoding="utf-8"))
    face_ids = sorted(
        entry["face_id"] for entry in document.get("faces", []) if entry.get("semantic") == args.semantic
    )
    if not face_ids:
        raise ValueError(f"No {args.semantic} faces found in {args.labels_json}")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    invalid = [face_id for face_id in face_ids if face_id < 1 or face_id > face_map.Size()]
    if invalid:
        raise ValueError(f"Label face IDs outside 1..{face_map.Size()}: {invalid}")

    display, start_display, _, _ = init_display(size=(1440, 960))
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.82, update=False)
    for face_id in face_ids:
        display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False)
    display.FitAll()
    display.Repaint()
    print(f"Highlighted {args.semantic} faces: {', '.join(f'F{face_id}' for face_id in face_ids)}")
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
