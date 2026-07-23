"""Visually review confirmed window seeds and unlabelled mirror candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_face_id(value: str) -> int:
    return int(value.strip().removeprefix("F"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Highlight confirmed windows in green and mirror-only candidates in orange."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    args = parser.parse_args()

    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_GREEN, Quantity_NOC_ORANGE
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods
    from OCC.Display.SimpleGui import init_display

    labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
    seed_ids = {
        int(face["face_id"])
        for face in labels.get("faces", [])
        if face.get("semantic") == "window"
    }
    with args.candidates_csv.open(newline="", encoding="utf-8") as input_file:
        candidate_ids = {parse_face_id(row["mirrored_face_id"]) for row in csv.DictReader(input_file)}
    candidate_ids -= seed_ids
    if not candidate_ids:
        raise ValueError("No new symmetric window candidates found")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    face_count = face_map.Size()
    invalid = [face_id for face_id in seed_ids | candidate_ids if not 1 <= face_id <= face_count]
    if invalid:
        raise ValueError(f"Face IDs outside 1..{face_count}: {sorted(invalid)}")

    display, start_display, _, _ = init_display(size=(1440, 960))
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.84, update=False)
    for face_id in sorted(seed_ids):
        display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_GREEN, update=False)
    for face_id in sorted(candidate_ids):
        display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False)
    display.FitAll()
    display.Repaint()
    print(
        f"Green confirmed windows ({len(seed_ids)}): {', '.join(f'F{face_id}' for face_id in sorted(seed_ids))}\n"
        f"Orange mirror candidates ({len(candidate_ids)}): {', '.join(f'F{face_id}' for face_id in sorted(candidate_ids))}"
    )
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
