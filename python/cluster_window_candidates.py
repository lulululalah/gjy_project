"""Group repeated topology-based window candidates for instance-level review."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster repeated window candidates from a candidate CSV.")
    parser.add_argument("candidates_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.candidates_csv.open(encoding="utf-8", newline="")))
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        face_count = len([item for item in row["candidate_face_ids"].split() if item])
        surface_signature = " ".join(sorted(row["candidate_surface_types"].split()))
        signature = (
            row["loop_edge_count"],
            str(face_count),
            surface_signature,
            f"{float(row['relative_area']):.4f}",
        )
        groups[signature].append(row)

    output_rows = []
    for cluster_id, (signature, members) in enumerate(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])), start=1
    ):
        output_rows.append(
            {
                "cluster_id": cluster_id,
                "member_count": len(members),
                "loop_edge_count": signature[0],
                "candidate_face_count": signature[1],
                "surface_signature": signature[2],
                "relative_area": signature[3],
                "candidate_ids": " ".join(row["candidate_id"] for row in members),
                "host_face_ids": " ".join(f"F{row['host_face_id']}" for row in members),
                "representative_candidate_id": members[0]["candidate_id"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = list(output_rows[0]) if output_rows else ["cluster_id", "member_count"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Exported {len(output_rows)} clusters: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
