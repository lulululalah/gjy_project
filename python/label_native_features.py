"""Add manual window or decal face labels to an existing full-face labels.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_face_ids(value: str) -> list[int]:
    face_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not face_ids or any(face_id <= 0 for face_id in face_ids):
        raise argparse.ArgumentTypeError("--face-ids must contain positive face IDs")
    if len(face_ids) != len(set(face_ids)):
        raise argparse.ArgumentTypeError("--face-ids must not contain duplicates")
    return face_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark one native window or decal instance in a full-face labels.json file."
    )
    parser.add_argument("labels_json", type=Path, help="Existing labels.json covering every STEP face.")
    parser.add_argument("--semantic", choices=["window", "decal"], required=True)
    parser.add_argument("--face-ids", type=parse_face_ids, required=True, help="Comma-separated faces of one instance.")
    parser.add_argument("--instance-id", type=int, help="Defaults to the next unused positive instance ID.")
    parser.add_argument("--output", type=Path, help="Defaults to overwriting labels_json after validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.loads(args.labels_json.read_text(encoding="utf-8"))
    faces = document.get("faces")
    if not isinstance(faces, list):
        raise ValueError("labels_json must contain a faces array")

    faces_by_id = {entry.get("face_id"): entry for entry in faces}
    if len(faces_by_id) != len(faces) or any(face_id not in faces_by_id for face_id in args.face_ids):
        raise ValueError("--face-ids must exist exactly once in the labels file")

    used_instance_ids = [entry.get("instance_id", -1) for entry in faces if entry.get("instance_id", -1) > 0]
    instance_id = args.instance_id or (max(used_instance_ids, default=0) + 1)
    if instance_id <= 0:
        raise ValueError("--instance-id must be positive")

    operation = "fill_window" if args.semantic == "window" else "merge_decal"
    for face_id in args.face_ids:
        entry = faces_by_id[face_id]
        entry["semantic"] = args.semantic
        entry["instance_id"] = instance_id
        entry["operation"] = operation

    output_path = args.output or args.labels_json
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Labeled {args.semantic} instance {instance_id}: {', '.join(f'F{face_id}' for face_id in args.face_ids)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
