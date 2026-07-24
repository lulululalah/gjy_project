"""Accept one clustered window-candidate group as separate window instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def face_ids(value: str) -> list[int]:
    return [int(item[1:]) for item in value.split() if item.startswith("F")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Label every candidate in one cluster as a window instance.")
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("candidates_csv", type=Path)
    parser.add_argument("clusters_csv", type=Path, nargs="?")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cluster-id", type=int)
    selection.add_argument("--all-candidates", action="store_true")
    parser.add_argument(
        "--include-host",
        action="store_true",
        help="Label each candidate's inner-loop host face as part of the window instance.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_candidates = list(csv.DictReader(args.candidates_csv.open(encoding="utf-8", newline="")))
    if args.all_candidates:
        candidates = all_candidates
    else:
        if args.clusters_csv is None:
            raise ValueError("clusters_csv is required when --cluster-id is used")
        clusters = list(csv.DictReader(args.clusters_csv.open(encoding="utf-8", newline="")))
        cluster = next((row for row in clusters if int(row["cluster_id"]) == args.cluster_id), None)
        if cluster is None:
            raise ValueError(f"Unknown cluster ID: {args.cluster_id}")
        candidate_ids = {int(value) for value in cluster["candidate_ids"].split()}
        candidates = [row for row in all_candidates if int(row["candidate_id"]) in candidate_ids]
        if len(candidates) != len(candidate_ids):
            raise ValueError("Cluster references missing candidate rows")

    document = json.loads(args.labels_json.read_text(encoding="utf-8"))
    faces_by_id = {entry.get("face_id"): entry for entry in document.get("faces", [])}
    used_ids = [entry.get("instance_id", -1) for entry in faces_by_id.values() if entry.get("instance_id", -1) > 0]
    next_instance_id = max(used_ids, default=0) + 1
    for candidate in candidates:
        candidate_face_ids = face_ids(candidate["candidate_face_ids"])
        if args.include_host:
            candidate_face_ids.append(int(candidate["host_face_id"]))
        prior_window_instances = {
            int(faces_by_id[face_id]["instance_id"])
            for face_id in candidate_face_ids
            if faces_by_id[face_id].get("semantic") == "window"
        }
        if len(prior_window_instances) > 1:
            raise ValueError(f"Candidate {candidate['candidate_id']} overlaps multiple window instances")
        if prior_window_instances:
            instance_id = prior_window_instances.pop()
        else:
            instance_id = next_instance_id
            next_instance_id += 1
        for face_id in candidate_face_ids:
            if face_id not in faces_by_id:
                raise ValueError(f"Candidate face F{face_id} is absent from labels")
            entry = faces_by_id[face_id]
            if entry.get("semantic") not in {"background", "window"}:
                raise ValueError(f"Candidate face F{face_id} already has semantic {entry.get('semantic')}")
            entry["semantic"] = "window"
            entry["instance_id"] = instance_id
            entry["operation"] = "fill_window"

    output = args.output or args.labels_json
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selection_name = "all candidates" if args.all_candidates else f"cluster {args.cluster_id}"
    print(f"Labeled {selection_name}: {len(candidates)} window instances")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
