"""Initialise full-face labels and apply confirmed native feature annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def parse_initialise_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize full-face background labels for a STEP file.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--base-model", help="Original source filename recorded in metadata.")
    return parser.parse_args()

def initialise_main() -> int:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    args = parse_initialise_args()
    if args.labels_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing labels: {args.labels_json}")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(reader.OneShape(), TopAbs_FACE, face_map)
    if face_map.Size() <= 0:
        raise RuntimeError("STEP has no faces")
    document = {
        "model_id": args.step_model.stem,
        "base_model": args.base_model or args.step_model.name,
        "output_step": str(args.step_model),
        "faces": [
            {"face_id": face_id, "semantic": "background", "instance_id": -1, "operation": "keep"}
            for face_id in range(1, face_map.Size() + 1)
        ],
        "instances": [],
    }
    args.labels_json.parent.mkdir(parents=True, exist_ok=True)
    args.labels_json.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Initialized {face_map.Size()} background labels: {args.labels_json}")
    return 0


def parse_groups(value: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for group_text in value.split(";"):
        face_ids = sorted({int(item.strip().removeprefix("F")) for item in group_text.split(",") if item.strip()})
        if not face_ids:
            raise ValueError("Each non-empty group must contain at least one face ID")
        groups.append(face_ids)
    return groups

def parse_annotation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply confirmed native feature labels without altering rivets.")
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--window-groups", help="Semicolon-separated window face groups; each group is one instance.")
    parser.add_argument("--decal-groups", help="Semicolon-separated decal face groups; each group is one instance.")
    parser.add_argument("--clear-face-ids", help="Comma-separated native-label face IDs to restore to background.")
    return parser.parse_args()

def apply_groups(document: dict, groups: list[list[int]], semantic: str, operation: str) -> int:
    faces_by_id = {int(face["face_id"]): face for face in document.get("faces", [])}
    used_ids = [int(face.get("instance_id", -1)) for face in faces_by_id.values() if int(face.get("instance_id", -1)) > 0]
    next_instance_id = max(used_ids, default=0) + 1
    written = 0
    for group in groups:
        for face_id in group:
            entry = faces_by_id.get(face_id)
            if entry is None:
                raise ValueError(f"Face F{face_id} is absent from labels JSON")
            if entry.get("semantic") not in {"background", semantic}:
                raise ValueError(f"Face F{face_id} already has semantic {entry.get('semantic')}")
        existing_ids = {int(faces_by_id[face_id].get("instance_id", -1)) for face_id in group if faces_by_id[face_id].get("semantic") == semantic}
        if len(existing_ids) > 1:
            raise ValueError(f"Confirmed group {group} overlaps multiple {semantic} instances")
        has_existing_instance = bool(existing_ids)
        instance_id = existing_ids.pop() if has_existing_instance else next_instance_id
        if not has_existing_instance:
            next_instance_id += 1
        for face_id in group:
            entry = faces_by_id[face_id]
            if entry.get("semantic") != semantic:
                written += 1
            entry["semantic"] = semantic
            entry["instance_id"] = instance_id
            entry["operation"] = operation
    return written

def clear_native_faces(document: dict, value: str) -> int:
    face_ids = sorted({int(item.strip().removeprefix("F")) for item in value.split(",") if item.strip()})
    faces_by_id = {int(face["face_id"]): face for face in document.get("faces", [])}
    cleared = 0
    for face_id in face_ids:
        entry = faces_by_id.get(face_id)
        if entry is None:
            raise ValueError(f"Face F{face_id} is absent from labels JSON")
        if entry.get("semantic") == "rivet":
            raise ValueError(f"Refusing to clear rivet face F{face_id}")
        if entry.get("semantic") != "background":
            cleared += 1
        entry["semantic"] = "background"
        entry["instance_id"] = -1
        entry["operation"] = "keep"
    return cleared

def annotate_main() -> int:
    args = parse_annotation_args()
    if not args.window_groups and not args.decal_groups and not args.clear_face_ids:
        raise ValueError("Provide native groups and/or --clear-face-ids")
    document = json.loads(args.labels_json.read_text(encoding="utf-8"))
    cleared_count = clear_native_faces(document, args.clear_face_ids) if args.clear_face_ids else 0
    window_count = apply_groups(document, parse_groups(args.window_groups), "window", "native_window") if args.window_groups else 0
    decal_count = apply_groups(document, parse_groups(args.decal_groups), "decal", "native_decal") if args.decal_groups else 0
    args.labels_json.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote native labels: cleared={cleared_count}, windows={window_count}, decals={decal_count}, output={args.labels_json}")
    return 0
