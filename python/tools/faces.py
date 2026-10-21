"""Inspect face identifiers and select eligible Boolean host faces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def pick_main() -> int:
    parser = argparse.ArgumentParser(description="Click a STEP face to print its F-number.")
    parser.add_argument("step_model", type=Path)
    parser.add_argument("--labels-json", type=Path, help="Optional labels JSON to color confirmed faces.")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        help="Optional Boolean host-face CSV; eligible faces are highlighted and selection is restricted to them.",
    )
    parser.add_argument(
        "--transparency",
        type=float,
        default=0.25,
        help="Model transparency in [0, 1]; use 0 for an opaque exterior check.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.transparency <= 1.0:
        parser.error("--transparency must be between 0 and 1.")

    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_NOC_ORANGE, Quantity_TOC_RGB
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import TopoDS_Compound, topods
    import tkinter as tk
    from OCC.Display.tkDisplay import tkViewer3d

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    labels_by_id: dict[int, str] = {}
    if args.labels_json:
        payload = json.loads(args.labels_json.read_text(encoding="utf-8"))
        labels_by_id = {
            int(face["face_id"]): str(face["semantic"])
            for face in payload.get("faces", [])
            if str(face.get("semantic", "background")) != "background"
        }
        invalid_ids = sorted(face_id for face_id in labels_by_id if not 1 <= face_id <= face_map.Size())
        if invalid_ids:
            raise ValueError(f"Label face IDs outside STEP range 1..{face_map.Size()}: {invalid_ids[:10]}")
    semantic_colors = {
        "rivet": Quantity_Color(0.95, 0.05, 0.05, Quantity_TOC_RGB),
        "window": Quantity_Color(0.15, 0.90, 0.15, Quantity_TOC_RGB),
        "decal": Quantity_Color(1.00, 0.55, 0.00, Quantity_TOC_RGB),
    }

    candidate_ids: set[int] = set()
    if args.candidate_csv:
        with args.candidate_csv.open(newline="", encoding="utf-8-sig") as input_file:
            candidate_ids = {int(row["face_id"]) for row in csv.DictReader(input_file)}
        invalid_candidates = sorted(face_id for face_id in candidate_ids if not 1 <= face_id <= face_map.Size())
        if invalid_candidates:
            raise ValueError(
                f"Candidate face IDs outside STEP range 1..{face_map.Size()}: {invalid_candidates[:10]}"
            )

    root = tk.Tk()
    root.title(f"STEP face picker: {args.step_model.name}")
    selected_face_text = tk.StringVar(value="Right-click a face to show its F-number here.")
    status = tk.Label(root, textvariable=selected_face_text, anchor="w", padx=8)
    status.pack(side=tk.BOTTOM, fill=tk.X)
    canvas = tkViewer3d(root)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.wait_visibility()
    display = canvas._display

    def draw_model_with_labels() -> None:
        display.DisplayShape(
            shape,
            color=Quantity_NOC_GRAY,
            transparency=args.transparency,
            update=False,
        )
        for face_id in sorted(candidate_ids):
            display.DisplayShape(
                topods.Face(face_map.FindKey(face_id)), color=Quantity_NOC_ORANGE, update=False
            )
        for face_id, semantic in labels_by_id.items():
            color = semantic_colors.get(semantic)
            if color is None:
                raise ValueError(f"Unsupported semantic in labels JSON: {semantic}")
            display.DisplayShape(topods.Face(face_map.FindKey(face_id)), color=color, update=False)

    draw_model_with_labels()
    display.Repaint()
    display.SetSelectionModeFace()

    model_box = Bnd_Box()
    brepbndlib.Add(shape, model_box)
    bounds = model_box.Get()
    axis_spans = [bounds[index + 3] - bounds[index] for index in range(3)]
    long_axis = max(range(3), key=lambda index: axis_spans[index])
    axis_min = bounds[long_axis]
    axis_max = bounds[long_axis + 3]

    def show_end_region(use_minimum_end: bool) -> None:
        cutoff = axis_min + (axis_max - axis_min) * 0.28 if use_minimum_end else axis_max - (axis_max - axis_min) * 0.28
        builder = BRep_Builder()
        region = TopoDS_Compound()
        builder.MakeCompound(region)
        for face_id in range(1, face_map.Size() + 1):
            face = face_map.FindKey(face_id)
            box = Bnd_Box()
            brepbndlib.Add(face, box)
            face_bounds = box.Get()
            center = (face_bounds[long_axis] + face_bounds[long_axis + 3]) * 0.5
            if (use_minimum_end and center <= cutoff) or (not use_minimum_end and center >= cutoff):
                builder.Add(region, face)
        display.EraseAll()
        display.DisplayShape(
            region,
            color=Quantity_NOC_GRAY,
            transparency=args.transparency,
            update=False,
        )
        display.FitAll()
        display.Repaint()
        print(f"Focused {'minimum' if use_minimum_end else 'maximum'} end of longest model axis")

    def show_full_model(event=None) -> None:
        display.EraseAll()
        draw_model_with_labels()
        display.FitAll()
        display.Repaint()
        print("View: full model")

    def on_select(selected_shapes, x: int, y: int) -> None:
        if not selected_shapes:
            print(f"No face selected at ({x}, {y}); click directly on a visible surface.")
            return

        def report_face_id(face_id: int) -> None:
            if candidate_ids and face_id not in candidate_ids:
                message = f"F{face_id} is not a Boolean host candidate"
            elif candidate_ids:
                message = f"Selected candidate face: F{face_id}"
            else:
                message = f"Selected face: F{face_id}"
            selected_face_text.set(message)
            print(message)

        for selected in selected_shapes:
            face_id = face_map.FindIndex(selected)
            if face_id:
                report_face_id(face_id)
                continue
            for candidate_id in range(1, face_map.Size() + 1):
                candidate = face_map.FindKey(candidate_id)
                if selected.IsSame(candidate) or selected.IsEqual(candidate):
                    report_face_id(candidate_id)
                    break
            else:
                print("Selected shape did not match a face; click directly on a visible surface.")

    display.register_select_callback(on_select)

    def select_face(event) -> None:
        display.MoveTo(event.x, event.y)
        display.Select(event.x, event.y)

    canvas.bind("<Button-3>", select_face)

    view_presets = {
        "1": ("View_Front", "front"),
        "2": ("View_Rear", "rear"),
        "3": ("View_Left", "left"),
        "4": ("View_Right", "right"),
        "5": ("View_Top", "top"),
        "0": ("View_Axo", "axonometric"),
    }

    def set_view(event) -> None:
        method_name, label = view_presets[event.char]
        method = getattr(display, method_name, None)
        if method is None:
            print(f"View preset unavailable: {label}")
            return
        method()
        display.FitAll()
        display.Repaint()
        print(f"View: {label}")

    for key in view_presets:
        root.bind(key, set_view)
    root.bind("7", lambda event: show_end_region(True))
    root.bind("8", lambda event: show_end_region(False))
    root.bind("9", show_full_model)
    canvas.focus_set()
    display.FitAll()
    if candidate_ids:
        print(f"Boolean host candidates highlighted: {len(candidate_ids)}")
        print("Left-drag rotates. Right-click an orange candidate face to print its F-number.")
    else:
        print("Left-drag rotates. Right-click a face to print its F-number.")
    print("View keys: 1=front, 2=rear, 3=left, 4=right, 5=top, 0=axonometric.")
    print("Focus keys: 7=one end, 8=the other end, 9=full model.")
    root.mainloop()
    return 0


def load_shape(step_path: Path):
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()

def face_center(face):
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt

    box = Bnd_Box()
    brepbndlib.Add(face, box)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    return gp_Pnt((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)

def model_center_and_diagonal(shape):
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    center = gp_Pnt((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)
    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5
    return center, max(diagonal, 1.0)

def face_map(shape):
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    mapped_faces = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, mapped_faces)
    return mapped_faces

def read_candidates(csv_path: Path, mapped_faces, max_candidates: int):
    candidates = []
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        for row in csv.DictReader(input_file):
            face_id = int(row["face_id"])
            if face_id <= 0 or face_id > mapped_faces.Size():
                continue
            candidates.append((
                face_id,
                mapped_faces.FindKey(face_id),
                int(row["solid_index"]),
                float(row["area"]),
                int(row["surface_type"]),
            ))
    candidates.sort(key=lambda item: item[3], reverse=True)
    return candidates[:max_candidates]

def display_candidates(shape, candidates) -> None:
    from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_ORANGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Dir, gp_Vec
    from OCC.Display.SimpleGui import init_display

    display, start_display, add_menu, add_function_to_menu = init_display(size=(1440, 960))
    model_center, diagonal = model_center_and_diagonal(shape)
    current_index = 0

    def show_current_candidate() -> None:
        face_id, raw_face, solid_index, area, surface_type = candidates[current_index]
        face = topods.Face(raw_face)
        display.EraseAll()
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.0, update=False)
        display.DisplayShape(face, color=Quantity_NOC_ORANGE, update=False)
        center = face_center(face)
        offset = gp_Vec(model_center, center)
        if offset.SquareMagnitude() < 1.0e-12:
            offset = gp_Vec(0.0, 0.0, 1.0)
        label_point = center.Translated(gp_Vec(gp_Dir(offset)) * (diagonal * 0.075))
        display.DisplayMessage(
            label_point,
            f"F{face_id}  ({current_index + 1}/{len(candidates)})",
            height=28.0,
            message_color=(0.0, 0.0, 0.0),
            update=False,
        )
        display.Repaint()
        print(f"Showing F{face_id}: solid={solid_index}, surface={surface_type}, area={area:.3f}")

    def previous_candidate() -> None:
        nonlocal current_index
        current_index = (current_index - 1) % len(candidates)
        show_current_candidate()

    def next_candidate() -> None:
        nonlocal current_index
        current_index = (current_index + 1) % len(candidates)
        show_current_candidate()

    add_menu("Candidate")
    add_function_to_menu("Candidate", previous_candidate)
    add_function_to_menu("Candidate", next_candidate)
    show_current_candidate()
    display.FitAll()
    print("Use Candidate > previous/next candidate to browse.")
    print("Tell Codex one or more displayed F-numbers to use as rivet host faces.")
    start_display()

def parse_args():
    parser = argparse.ArgumentParser(description="Browse Boolean-fusable rivet host faces.")
    parser.add_argument("step_model", type=Path, help="Source STEP model used by Detector.exe.")
    parser.add_argument("--candidate-csv", type=Path, help="CSV generated by --dump-boolean-host-faces.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Display only the largest eligible faces (default: 20).",
    )
    parser.add_argument("--no-display", action="store_true", help="Validate the candidate CSV without opening a GUI.")
    return parser.parse_args()

def host_main() -> int:
    args = parse_args()
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    candidate_csv = args.candidate_csv or args.step_model.with_suffix(".boolean_host_faces.csv")
    if not candidate_csv.is_file():
        raise FileNotFoundError(
            f"Candidate CSV not found: {candidate_csv}. Run Detector.exe --dump-boolean-host-faces first."
        )
    shape = load_shape(args.step_model)
    candidates = read_candidates(candidate_csv, face_map(shape), args.max_candidates)
    if not candidates:
        raise RuntimeError("No Boolean-fusable faces from the candidate CSV could be mapped to this STEP file.")
    print(f"Boolean-fusable faces shown: {len(candidates)} (largest by area)")
    print(f"Candidate CSV: {candidate_csv}")
    if not args.no_display:
        display_candidates(shape, candidates)
    return 0
