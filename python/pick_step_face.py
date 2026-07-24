"""Open a STEP model and print the B-Rep face ID selected with the mouse."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Click a STEP face to print its F-number.")
    parser.add_argument("step_model", type=Path)
    args = parser.parse_args()

    from OCC.Core.Quantity import Quantity_NOC_GRAY
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import TopoDS_Compound
    import tkinter as tk
    from OCC.Display.tkDisplay import tkViewer3d

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step_model)) != 1:
        raise RuntimeError(f"Unable to read STEP: {args.step_model}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    root = tk.Tk()
    root.title("STEP face picker: right-click a face")
    canvas = tkViewer3d(root)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.wait_visibility()
    display = canvas._display
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.25, update=True)
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
        display.DisplayShape(region, color=Quantity_NOC_GRAY, transparency=0.18, update=False)
        display.FitAll()
        display.Repaint()
        print(f"Focused {'minimum' if use_minimum_end else 'maximum'} end of longest model axis")

    def show_full_model(event=None) -> None:
        display.EraseAll()
        display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.25, update=False)
        display.FitAll()
        display.Repaint()
        print("View: full model")

    def on_select(selected_shapes, x: int, y: int) -> None:
        if not selected_shapes:
            print(f"No face selected at ({x}, {y}); click directly on a visible surface.")
            return
        for selected in selected_shapes:
            face_id = face_map.FindIndex(selected)
            if face_id:
                print(f"Selected face: F{face_id}")
                continue
            for candidate_id in range(1, face_map.Size() + 1):
                candidate = face_map.FindKey(candidate_id)
                if selected.IsSame(candidate) or selected.IsEqual(candidate):
                    print(f"Selected face: F{candidate_id}")
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
    print("Left-drag rotates. Right-click a face to print its F-number.")
    print("View keys: 1=front, 2=rear, 3=left, 4=right, 5=top, 0=axonometric.")
    print("Focus keys: 7=one end, 8=the other end, 9=full model.")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
