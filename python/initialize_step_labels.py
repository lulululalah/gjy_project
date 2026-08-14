"""Create an all-background labels JSON for a STEP model without feature injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize full-face background labels for a STEP file.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--base-model", help="Original source filename recorded in metadata.")
    return parser.parse_args()


def main() -> int:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
