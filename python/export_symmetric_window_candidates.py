"""Export mirror-completed window candidates without changing ground-truth labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reflect accepted window seed faces across the best model symmetry plane."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument(
        "--seed-face-ids",
        help="Optional comma-separated face IDs to mirror instead of semantic=window labels.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-normalized-distance", type=float, default=0.025)
    parser.add_argument("--min-area-ratio", type=float, default=0.70)
    parser.add_argument("--max-area-ratio", type=float, default=1.43)
    return parser.parse_args()


def main() -> int:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods

    args = parse_args()
    if args.seed_face_ids:
        seed_ids = sorted({int(value.strip()) for value in args.seed_face_ids.split(",") if value.strip()})
    elif args.labels_json:
        label_data = json.loads(args.labels_json.read_text(encoding="utf-8"))
        seed_ids = sorted(
            int(face["face_id"])
            for face in label_data.get("faces", [])
            if face.get("semantic") == "window"
        )
    else:
        raise RuntimeError("Provide --seed-face-ids or --labels-json")
    if not seed_ids:
        raise RuntimeError("No seed face IDs were selected")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    model_box = Bnd_Box()
    brepbndlib.Add(shape, model_box)
    bounds = model_box.Get()
    mins, maxs = bounds[:3], bounds[3:]
    model_center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    diagonal = math.dist(mins, maxs)
    if diagonal <= 0:
        raise RuntimeError("STEP model has an invalid bounding box")

    faces: dict[int, dict[str, object]] = {}
    for face_id in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(face_id))
        box = Bnd_Box()
        brepbndlib.Add(face, box)
        face_bounds = box.Get()
        center = tuple((face_bounds[i] + face_bounds[i + 3]) * 0.5 for i in range(3))
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        faces[face_id] = {
            "center": center,
            "area": properties.Mass(),
            "surface_type": int(BRepAdaptor_Surface(face).GetType()),
        }

    missing = [face_id for face_id in seed_ids if face_id not in faces]
    if missing:
        raise RuntimeError(f"Seed faces are absent from STEP: {missing}")

    def find_matches(axis: int) -> tuple[int, float, list[dict[str, object]]]:
        matches: list[dict[str, object]] = []
        for seed_id in seed_ids:
            seed = faces[seed_id]
            target = list(seed["center"])
            target[axis] = 2.0 * model_center[axis] - target[axis]
            best: tuple[float, int, float] | None = None
            for face_id, candidate in faces.items():
                if face_id == seed_id or candidate["surface_type"] != seed["surface_type"]:
                    continue
                area_ratio = float(candidate["area"]) / max(float(seed["area"]), 1e-12)
                if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
                    continue
                distance = math.dist(target, candidate["center"])
                normalized = distance / diagonal
                if best is None or normalized < best[0]:
                    best = (normalized, face_id, area_ratio)
            if best is not None and best[0] <= args.max_normalized_distance:
                matches.append(
                    {
                        "seed_face_id": seed_id,
                        "mirrored_face_id": best[1],
                        "normalized_distance": best[0],
                        "area_ratio": best[2],
                    }
                )
        return len(matches), sum(row["normalized_distance"] for row in matches), matches

    trials = [find_matches(axis) for axis in range(3)]
    best_axis = max(range(3), key=lambda axis: (trials[axis][0], -trials[axis][1]))
    _, _, matches = trials[best_axis]
    axis_name = "XYZ"[best_axis]
    known_seed_ids = set(seed_ids)
    rows = []
    seen_mirrors: set[int] = set()
    for row in matches:
        mirror_id = int(row["mirrored_face_id"])
        if mirror_id in known_seed_ids or mirror_id in seen_mirrors:
            continue
        seen_mirrors.add(mirror_id)
        rows.append(
            {
                "model_name": args.step_model.name,
                "symmetry_axis": axis_name,
                "symmetry_plane_coordinate": f"{model_center[best_axis]:.6f}",
                "seed_face_id": f"F{row['seed_face_id']}",
                "mirrored_face_id": f"F{mirror_id}",
                "mirrored_face_center": " ".join(f"{value:.6f}" for value in faces[mirror_id]["center"]),
                "surface_type": faces[mirror_id]["surface_type"],
                "area_ratio": f"{row['area_ratio']:.6f}",
                "normalized_distance": f"{row['normalized_distance']:.6f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "symmetry_axis", "symmetry_plane_coordinate", "seed_face_id",
        "mirrored_face_id", "mirrored_face_center", "surface_type", "area_ratio",
        "normalized_distance",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Selected {axis_name}={model_center[best_axis]:.6f}; "
        f"exported {len(rows)} new mirror candidates from {len(seed_ids)} seed window faces: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
