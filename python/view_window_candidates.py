"""Interactively review topology-based window candidates exported to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_face_ids(value: str) -> list[int]:
    return [int(item[1:]) for item in value.split() if item.startswith("F")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse window candidates one instance at a time.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("candidates_csv", type=Path)
    parser.add_argument("--candidate-ids", help="Optional comma-separated candidate IDs to review.")
    parser.add_argument(
        "--highlight-host",
        action="store_true",
        help="Highlight the inner-loop host face together with its surrounding candidate faces.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Highlight every selected candidate at once for a whole-aircraft recall check.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.candidates_csv.open(encoding="utf-8", newline="")))
    if args.candidate_ids:
        requested = {int(item.strip()) for item in args.candidate_ids.split(",") if item.strip()}
        rows = [row for row in rows if int(row["candidate_id"]) in requested]
    if not rows:
        raise ValueError("No candidate rows selected")

    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_ORANGE
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
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    current = 0
    display, start_display, add_menu, add_function_to_menu = init_display(size=(1440, 960))

    if args.all:
        face_ids = {
            face_id
            for row in rows
            for face_id in parse_face_ids(row["candidate_face_ids"])
        }
        if args.highlight_host:
            face_ids.update(int(row["host_face_id"]) for row in rows)
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.88, update=False)
        for face_id in sorted(face_ids):
            display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False)
        display.FitAll()
        display.Repaint()
        print(f"Highlighted {len(rows)} candidates and {len(face_ids)} faces.")
        start_display()
        return 0

    def show() -> None:
        row = rows[current]
        face_ids = parse_face_ids(row["candidate_face_ids"])
        host_face_id = int(row["host_face_id"])
        display.EraseAll()
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.88, update=False)
        if args.highlight_host:
            display.DisplayShape(topods.Face(face_map.FindKey(host_face_id)), color=Quantity_NOC_ORANGE, update=False)
        else:
            display.DisplayShape(topods.Face(face_map.FindKey(host_face_id)), color=Quantity_NOC_GRAY, transparency=0.45, update=False)
        for face_id in face_ids:
            display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False)
        display.FitAll()
        display.Repaint()
        print(
            f"Candidate {row['candidate_id']} ({current + 1}/{len(rows)}): "
            f"host=F{host_face_id}, faces={row['candidate_face_ids']}, score={row['heuristic_score']}"
        )

    def previous_candidate() -> None:
        nonlocal current
        current = (current - 1) % len(rows)
        show()

    def next_candidate() -> None:
        nonlocal current
        current = (current + 1) % len(rows)
        show()

    add_menu("Candidate")
    add_function_to_menu("Candidate", previous_candidate)
    add_function_to_menu("Candidate", next_candidate)
    show()
    print("Orange faces form one candidate instance. Use Candidate > previous/next candidate to browse.")
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
