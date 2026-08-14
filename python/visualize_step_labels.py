"""Display semantic STEP labels with distinct face colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize STEP faces from a labels JSON file.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--context-transparency", type=float, default=0.72)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.context_transparency <= 1.0:
        raise ValueError("--context-transparency must be in [0, 1]")

    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_TOC_RGB
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods
    from OCC.Display.SimpleGui import init_display

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()

    payload = json.loads(args.labels_json.read_text(encoding="utf-8"))
    labels_by_id = {
        int(face["face_id"]): str(face["semantic"])
        for face in payload.get("faces", [])
        if str(face.get("semantic", "background")) != "background"
    }

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    invalid_ids = sorted(face_id for face_id in labels_by_id if not 1 <= face_id <= face_map.Size())
    if invalid_ids:
        raise ValueError(f"Label face IDs outside STEP range 1..{face_map.Size()}: {invalid_ids[:10]}")

    colors = {
        "rivet": Quantity_Color(0.95, 0.05, 0.05, Quantity_TOC_RGB),
        "window": Quantity_Color(0.10, 0.35, 1.00, Quantity_TOC_RGB),
        "decal": Quantity_Color(1.00, 0.55, 0.00, Quantity_TOC_RGB),
    }
    counts: dict[str, int] = {}
    display, start_display, _, _ = init_display()
    display.DisplayShape(
        shape,
        color=Quantity_NOC_GRAY,
        transparency=args.context_transparency,
        update=False,
    )
    for face_id, semantic in labels_by_id.items():
        color = colors.get(semantic)
        if color is None:
            raise ValueError(f"Unsupported semantic in labels JSON: {semantic}")
        display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=color, update=False)
        counts[semantic] = counts.get(semantic, 0) + 1

    print(f"Displaying {args.step_model}")
    print("red=rivet, blue=window, orange=decal, transparent gray=background")
    print(f"Labeled face counts: {counts}")
    display.FitAll()
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
