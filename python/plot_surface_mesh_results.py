"""Create a paper-ready comparison figure from paired surface-mesh CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHORT_LABELS = {
    "109-ww1 standard E-1 aircraft": "109-E1",
    "DC-10": "DC-10",
    "747-400": "747",
    "AULIRA 2": "AULIRA",
    "Air Plane Idea A": "Idea A*",
    "Airbus": "Airbus",
    "Airplane body": "Body†",
    "Cessna Citation 2": "Cessna",
    "Gulfstream G280 v17": "G280",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [SHORT_LABELS.get(row["model"], row["model"]) for row in rows]
    series = [
        ("Mesh nodes", "original_mesh_node_count", "simplified_mesh_node_count"),
        ("Surface triangles", "original_mesh_triangle_count", "simplified_mesh_triangle_count"),
        ("Meshing time (s)", "original_mesh_elapsed_seconds_median", "simplified_mesh_elapsed_seconds_median"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    positions = np.arange(len(names))
    width = 0.35
    for axis, (title, original_key, simplified_key) in zip(axes, series):
        original = [float(row[original_key]) for row in rows]
        simplified = [float(row[simplified_key]) for row in rows]
        axis.bar(positions - width / 2, original, width, label="Original", color="#4E79A7")
        axis.bar(positions + width / 2, simplified, width, label="Simplified", color="#59A14F")
        axis.set_title(title, fontsize=10)
        axis.set_xticks(positions, names, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.text(
        0.5,
        -0.03,
        "* original and simplified B-Reps invalid; † simplified B-Rep invalid",
        ha="center",
        fontsize=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
