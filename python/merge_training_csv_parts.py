"""Merge per-model training CSV exports and assign stable graph IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    part_paths = sorted(args.parts_dir.glob("*.csv"))
    if not part_paths:
        raise ValueError(f"No CSV parts found: {args.parts_dir}")
    frames = [pd.read_csv(path) for path in part_paths]
    columns = list(frames[0].columns)
    if any(list(frame.columns) != columns for frame in frames[1:]):
        raise ValueError("Part CSV columns do not match")
    merged = pd.concat(frames, ignore_index=True)
    model_names = list(dict.fromkeys(merged["model_name"].tolist()))
    if len(model_names) != len(part_paths):
        raise ValueError("Expected exactly one unique model per part CSV")
    graph_ids = {model_name: graph_id for graph_id, model_name in enumerate(model_names)}
    merged["graph_id"] = merged["model_name"].map(graph_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(part_paths)} models, {len(merged)} faces: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
