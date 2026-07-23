"""Accept symmetric window candidates as individual labeled window instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_face_id(value: str) -> int:
    return int(value.strip().removeprefix("F"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write approved mirror-only window candidates into a full-face labels.json file."
    )
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("candidates_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = json.loads(args.labels_json.read_text(encoding="utf-8"))
    faces_by_id = {int(entry["face_id"]): entry for entry in document.get("faces", [])}
    with args.candidates_csv.open(newline="", encoding="utf-8") as input_file:
        candidate_ids = sorted({parse_face_id(row["mirrored_face_id"]) for row in csv.DictReader(input_file)})
    if not candidate_ids:
        raise ValueError("No symmetric candidates found")
    missing = [face_id for face_id in candidate_ids if face_id not in faces_by_id]
    if missing:
        raise ValueError(f"Candidates absent from labels: {missing}")

    used_instance_ids = [
        int(entry.get("instance_id", -1))
        for entry in faces_by_id.values()
        if int(entry.get("instance_id", -1)) > 0
    ]
    next_instance_id = max(used_instance_ids, default=0) + 1
    added = []
    for face_id in candidate_ids:
        entry = faces_by_id[face_id]
        if entry.get("semantic") == "window":
            continue
        entry["semantic"] = "window"
        entry["instance_id"] = next_instance_id
        entry["operation"] = "fill_window"
        added.append(face_id)
        next_instance_id += 1

    output = args.output or args.labels_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Accepted {len(added)} symmetric candidates as independent window instances")
    print(f"Faces: {', '.join(f'F{face_id}' for face_id in added)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
