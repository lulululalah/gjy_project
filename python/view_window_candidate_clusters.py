"""Highlight the union of repeated window-candidate clusters for one-pass review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Highlight all faces from selected window candidate clusters.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("candidates_csv", type=Path)
    parser.add_argument("clusters_csv", type=Path)
    parser.add_argument("--cluster-ids", help="Comma-separated cluster IDs. Defaults to groups with at least --min-members.")
    parser.add_argument("--min-members", type=int, default=10)
    args = parser.parse_args()

    clusters = list(csv.DictReader(args.clusters_csv.open(encoding="utf-8", newline="")))
    requested = {int(value) for value in args.cluster_ids.split(",")} if args.cluster_ids else {
        int(row["cluster_id"]) for row in clusters if int(row["member_count"]) >= args.min_members
    }
    candidate_ids = {
        int(value) for row in clusters if int(row["cluster_id"]) in requested
        for value in row["candidate_ids"].split()
    }
    candidates = [row for row in csv.DictReader(args.candidates_csv.open(encoding="utf-8", newline="")) if int(row["candidate_id"]) in candidate_ids]
    face_ids = sorted({int(value[1:]) for row in candidates for value in row["candidate_face_ids"].split() if value.startswith("F")})
    if not face_ids:
        raise ValueError("No candidate faces selected")

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
    display, start_display, _, _ = init_display(size=(1440, 960))
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.86, update=False)
    for face_id in face_ids:
        display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False)
    display.FitAll(); display.Repaint()
    print(f"Highlighted clusters {sorted(requested)}: {len(candidates)} instances, {len(face_ids)} faces")
    start_display()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
