"""Open a STEP model and print the B-Rep face ID selected with the mouse."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Click a STEP face to print its F-number.")
    parser.add_argument("step_model", type=Path)
    args = parser.parse_args()

    from OCC.Core.Quantity import Quantity_NOC_GRAY
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
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
    display.FitAll()
    print("Left-drag rotates. Right-click the side decal host surface to print its F-number.")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
