"""Render native CAD/mesh paper figures and inspect final STEP boundary edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from OCC.Core.Aspect import Aspect_TOL_SOLID
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Prs3d import Prs3d_LineAspect
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import topexp
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Display.SimpleGui import init_display

ROOT = Path(__file__).resolve().parents[2]

STEP_PATH = ROOT / "data/plane_model/new_data/Gulfstream G280 v17_wing_rivets.step"

LABEL_PATH = ROOT / "data/plane_model/label/Gulfstream G280 v17_wing_rivets.labels.json"

BODY = Quantity_Color(0.30, 0.34, 0.39, Quantity_TOC_RGB)

FIGURE11_STYLE = {"body_rgb": (0.94, 0.96, 0.98),
                  "edge_rgb": (0.015, 0.02, 0.025), "edge_width": 0.85}

COLORS = {
    "rivet": Quantity_Color(0.02, 0.18, 0.95, Quantity_TOC_RGB),
    "window": Quantity_Color(0.95, 0.28, 0.01, Quantity_TOC_RGB),
    "decal": Quantity_Color(0.95, 0.28, 0.01, Quantity_TOC_RGB),
}

def dump_png(display, path):
    """Publish only a complete native render; retry transient Windows dump errors."""
    from PIL import Image
    path = Path(path).resolve()
    temporary = path.with_name(path.stem + '.rendering.png')
    error = None
    for _ in range(3):
        temporary.unlink(missing_ok=True)
        display.Repaint()
        try:
            if not display.View.Dump(str(temporary)):
                continue
            with Image.open(temporary) as rendered:
                rendered.verify()
            temporary.replace(path)
            return
        except OSError as failure:
            error = failure
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f'CAD image export failed: {path}') from error

def load_faces(step_path=STEP_PATH):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read {step_path}")
    reader.TransferRoots()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(reader.OneShape(), TopAbs_FACE, face_map)
    return {
        index: topods.Face(face_map.FindKey(index))
        for index in range(1, face_map.Size() + 1)
    }

def load_labels(label_path=LABEL_PATH):
    data = json.loads(label_path.read_text(encoding="utf-8"))
    labels = {int(row["face_id"]): row["semantic"] for row in data["faces"]}
    if len(labels) != len(data["faces"]):
        raise ValueError(f"Duplicate face labels in {label_path}")
    if set(labels.values()) - {"background", *COLORS}:
        raise ValueError(f"Unsupported semantic labels in {label_path}")
    return labels

def show_faces(display, faces, labels, style=None):
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    display.SetOrthographicProjection()
    display.default_drawer.SetFaceBoundaryDraw(True)
    boundary = Prs3d_LineAspect(
        Quantity_Color(0.003, 0.003, 0.003, Quantity_TOC_RGB), Aspect_TOL_SOLID, 1.3
    )
    display.default_drawer.SetFaceBoundaryAspect(boundary)
    shown = []
    body = Quantity_Color(*style["body_rgb"], Quantity_TOC_RGB) if style else BODY
    for face_id, face in faces.items():
        semantic = labels[face_id]
        ais = display.DisplayShape(
            face, color=COLORS.get(semantic, body), transparency=0.0, update=False
        )[0]
        if style:
            from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NOM_PLASTER

            ais.SetMaterial(Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTER))
            ais.SetColor(COLORS.get(semantic, body))
        ais.Attributes().SetFaceBoundaryAspect(boundary)
        ais.Attributes().SetFaceBoundaryDraw(True)
        display.Context.Redisplay(ais, False)
        shown.append((ais, semantic))
    return shown

def bounds_of_faces(faces):
    box = Bnd_Box()
    for face in faces.values():
        brepbndlib.AddOptimal(face, box, False, False)
    if box.IsVoid():
        raise ValueError("Cannot frame an empty model")
    return box.Get()

def apply_camera(display, view, center, scale):
    directions = {
        "side": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    }
    direction, up = directions[view]
    # Setting the observation point changes the direction; orient only afterwards.
    display.View.SetAt(*center)
    display.View.SetProj(*direction)
    display.View.SetUp(*up)
    display.View.Camera().SetScale(scale)
    display.Repaint()

def select_detail_regions(rivet_bounds, count=3):
    candidates = []
    for face_id, box in sorted(rivet_bounds, key=lambda item: -(item[1][1] + item[1][4])):
        center = tuple((box[i] + box[i + 3]) / 2 for i in range(3))
        # Multiple annotated faces can belong to the same physical rivet.
        if all(sum((center[i] - other["center"][i]) ** 2 for i in (0, 2)) > 0.2 ** 2
               for other in candidates):
            candidates.append({"original_face_id": face_id, "center": center, "box": box})
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct rivet locations, found {len(candidates)}")
    selected = [candidates.pop(0)]
    while len(selected) < count:
        next_region = max(candidates, key=lambda region: min(
            sum((region["center"][i] - chosen["center"][i]) ** 2 for i in (0, 2))
            for chosen in selected
        ))
        selected.append(next_region)
        candidates.remove(next_region)
    scale = max(max(region["box"][i + 3] - region["box"][i]
                    for region in selected for i in range(3)) * 20, 0.4)
    selected.sort(key=lambda region: region["center"][2])
    return [{"number": number, "original_face_id": region["original_face_id"],
             "center": region["center"], "scale": scale}
            for number, region in enumerate(selected, 1)]

def select_surface_detail_regions(feature_bounds, side_separator_z=0.0, window_max_x=float("inf")):
    """Frame one passenger window and the visible-side decal group in side view."""
    windows = []
    decals = []
    for face_id, semantic, box in feature_bounds:
        center = tuple((box[i] + box[i + 3]) / 2 for i in range(3))
        width, height = box[3] - box[0], box[4] - box[1]
        if center[2] <= side_separator_z:
            continue
        if (semantic == "window" and center[0] < window_max_x
                and 0.4 < width < 4 and 0.4 < height < 4):
            windows.append((face_id, box))
        elif semantic == "decal":
            decals.append((face_id, box))
    if not windows or not decals:
        raise ValueError("Need a visible-side passenger window and decal group; "
                         f"candidate bounds: {feature_bounds}")
    middle_x = sum((box[0] + box[3]) / 2 for _, box in windows) / len(windows)
    window = min(windows, key=lambda item: abs((item[1][0] + item[1][3]) / 2 - middle_x))
    regions = []
    for name, selected in (("window", [window]), ("decal", decals)):
        box = tuple(min(item[1][i] for item in selected) if i < 3
                    else max(item[1][i] for item in selected) for i in range(6))
        regions.append({
            "name": name, "view": "side",
            "original_face_ids": [item[0] for item in selected],
            "center": tuple((box[i] + box[i + 3]) / 2 for i in range(3)),
            "scale": max(box[3] - box[0], box[4] - box[1]) * 1.8,
        })
    return regions

def export_figure6(output_dir, horizontal=False):
    """Render the current recorded stages with shared orthographic cameras."""
    from OCC.Display.OCCViewer import Viewer3d

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT/'paper/manuscript/current_render_sources.json').read_text(encoding='utf-8'))
    cases = [(item['name'], ROOT/item['step'], ROOT/item['labels'])
             for item in sources['gulfstream']]
    loaded = []
    for stage, step_path, label_path in cases:
        faces, labels = load_faces(step_path), load_labels(label_path)
        if set(faces) != set(labels):
            raise RuntimeError(f"Face mismatch in {stage}: STEP={len(faces)}, labels={len(labels)}")
        loaded.append((stage, step_path, label_path, faces, labels))

    bounds = bounds_of_faces(loaded[0][3])
    center = tuple((bounds[i] + bounds[i + 3]) / 2 for i in range(3))
    extent = tuple(bounds[i + 3] - bounds[i] for i in range(3))
    # The same horizontal world span in both columns keeps aircraft lengths equal.
    sizes = {"side": (2400, 1100), "top": (2400, 2100)}
    horizontal_span = max(extent[0], extent[1] * 2400 / 1100, extent[2] * 2400 / 2100) * 1.08
    scales = {view: horizontal_span * size[1] / size[0] for view, size in sizes.items()}

    rivet_bounds = [
        (face_id, bounds_of_faces({face_id: face}))
        for face_id, face in loaded[0][3].items()
        if loaded[0][4][face_id] == "rivet"
    ]
    if not rivet_bounds:
        raise RuntimeError("Original stage has no annotated rivet for the inset")
    detail_regions = select_detail_regions(rivet_bounds)
    if horizontal:
        for region in detail_regions:
            region["scale"] *= 0.25
    surface_regions = select_surface_detail_regions([
        (face_id, loaded[0][4][face_id], bounds_of_faces({face_id: face}))
        for face_id, face in loaded[0][3].items()
        if loaded[0][4][face_id] in ("window", "decal")
    ], side_separator_z=center[2], window_max_x=center[0]) if horizontal else []
    if horizontal:
        print(f"Surface detail regions: {json.dumps(surface_regions)}", flush=True)
    prefix = "fig6_g280_horizontal" if horizontal else "fig6_g280"

    display = Viewer3d()
    display.Create(display_glinfo=False)
    display.SetModeShaded()

    def dump_view(path, size):
        requested = size
        for _ in range(3):
            dump_png(display, path)
            header = path.read_bytes()[:24]
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"Invalid PNG export: {path}")
            actual = (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))
            if actual == size:
                return
            # Windows' hidden render window can include non-client borders.
            # Resize and re-render; never crop or stretch the exported pixels.
            requested = tuple(requested[i] + size[i] - actual[i] for i in range(2))
            display.SetSize(*requested)
            display.OnResize()
            display.Repaint()
        raise RuntimeError(f"Unable to export size {size}: actual {actual}")

    def set_size(size):
        display.SetSize(*size)
        display.OnResize()

    def camera_record():
        camera = display.View.Camera()
        return {"center": camera.Center().Coord(), "direction": camera.Direction().Coord(),
                "up": camera.Up().Coord(), "scale": camera.Scale(), "aspect": camera.Aspect()}

    manifest = {
        "projection": "orthographic", "model_geometry_modified": False,
        "reference_bounds": bounds, "center": center,
        "views": {view: {"size": size, "scale": scales[view]} for view, size in sizes.items()
                  if not horizontal or view == "side"},
        "details": detail_regions,
        "surface_details": surface_regions,
        "stages": [],
    }
    for stage, step_path, label_path, faces, labels in loaded:
        display.EraseAll()
        show_faces(display, faces, labels)
        images = []
        cameras = {}
        for view, size in sizes.items():
            if horizontal and view == "top":
                continue
            set_size(size)
            apply_camera(display, view, center, scales[view])
            image_path = output_dir / f"{prefix}_{stage}_{view}_fixed.png"
            dump_view(image_path, size)
            cameras[view] = camera_record()
            images.append(image_path.name)
        if stage != "surface_processed":
            for region in detail_regions:
                set_size((900, 900))
                apply_camera(display, "top", region["center"], region["scale"])
                number = region["number"]
                image_path = output_dir / f"{prefix}_{stage}_detail_{number}_fixed.png"
                dump_view(image_path, (900, 900))
                cameras[f"detail_{number}"] = camera_record()
                images.append(image_path.name)
        for region in surface_regions:
            set_size((900, 900))
            apply_camera(display, region["view"], region["center"], region["scale"])
            name = region["name"]
            image_path = output_dir / f"{prefix}_{stage}_{name}_fixed.png"
            dump_view(image_path, (900, 900))
            cameras[name] = camera_record()
            images.append(image_path.name)
        manifest["stages"].append({
            "stage": stage, "step": str(step_path), "labels": str(label_path),
            "step_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "faces": len(faces), "images": images, "cameras": cameras,
        })
        print(f"Exported {stage}: {len(faces)} faces, {len(images)} views", flush=True)
    manifest_path = output_dir / f"{prefix}_fixed_camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Camera manifest: {manifest_path}", flush=True)

def export_figure7(output_dir, model="airbus"):
    """Export a test-aircraft pair and three independent rivet locations."""
    if model not in ('airbus', 'cessna'):
        raise ValueError(f'Unsupported Figure 7 model: {model}')
    sources = json.loads((ROOT/'paper/manuscript/current_render_sources.json').read_text(encoding='utf-8'))
    cases = [(item['name'], ROOT/item['step'], ROOT/item['labels'])
             for item in sources[model][:2]]
    if model == "cessna":
        # The positive-Y side faces the paired orthographic camera.
        export_paired_rivet_views(output_dir, cases, "fig7_cessna",
                                 lambda box: (box[1] + box[4]) / 2 > 0,
                                 detail_scale_factor=0.25, camera_up=(0, 0, 1))
        return
    # This model's lower-surface rivets are occluded in the canonical top view.
    export_paired_rivet_views(output_dir, cases, "fig7_airbus",
                             lambda box: (box[1] + box[4]) / 2 >= -0.5)


def export_paired_rivet_views(output_dir, cases, prefix, detail_filter,
                             detail_scale_factor=1.0, detail_face_ids=None, camera_up=None):
    """Share exact before/after cameras; keep each model's candidate rule explicit."""
    from OCC.Display.OCCViewer import Viewer3d

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for stage, step_path, label_path in cases:
        faces, labels = load_faces(step_path), load_labels(label_path)
        if set(faces) != set(labels):
            raise RuntimeError(f"Face mismatch in {stage}: STEP={len(faces)}, labels={len(labels)}")
        loaded.append((stage, step_path, label_path, faces, labels))
    bounds = bounds_of_faces(loaded[0][3])
    center = tuple((bounds[i] + bounds[i + 3]) / 2 for i in range(3))
    size = (2400, 2100)
    scale = max((bounds[3] - bounds[0]) * size[1] / size[0], bounds[5] - bounds[2]) * 1.08
    rivet_bounds = [
        (face_id, bounds_of_faces({face_id: face}))
        for face_id, face in loaded[0][3].items()
        if loaded[0][4][face_id] == "rivet"
    ]
    regions = select_detail_regions([
        (face_id, box) for face_id, box in rivet_bounds
        if detail_filter(box) and (detail_face_ids is None or face_id in detail_face_ids)
    ])
    for region in regions:
        region["scale"] *= detail_scale_factor
    display = Viewer3d()
    display.Create(display_glinfo=False)
    display.SetModeShaded()

    def export_view(path, image_size, at, view_scale):
        requested = image_size
        display.SetSize(*requested)
        display.OnResize()
        apply_camera(display, "top", at, view_scale)
        if camera_up is not None:
            display.View.SetUp(*camera_up)
            display.Repaint()
        for _ in range(3):
            dump_png(display, path)
            header = path.read_bytes()[:24]
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"Invalid PNG export: {path}")
            actual = (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))
            if actual == image_size:
                camera = display.View.Camera()
                return {"center": camera.Center().Coord(), "direction": camera.Direction().Coord(),
                        "up": camera.Up().Coord(), "scale": camera.Scale(), "aspect": camera.Aspect()}
            requested = tuple(requested[i] + image_size[i] - actual[i] for i in range(2))
            display.SetSize(*requested)
            display.OnResize()
            display.Repaint()
        raise RuntimeError(f"Unable to export size {image_size}: {actual}")

    manifest = {"projection": "orthographic", "view": "top" if camera_up is None else "custom", "model_geometry_modified": False,
                "reference_bounds": bounds, "center": center, "scale": scale,
                "size": size, "details": regions, "stages": []}
    for stage, step_path, label_path, faces, labels in loaded:
        display.EraseAll()
        show_faces(display, faces, labels)
        overview_path = output_dir / f"{prefix}_{stage}_overview_fixed.png"
        cameras = {"overview": export_view(overview_path, size, center, scale)}
        images = [overview_path.name]
        for region in regions:
            number = region["number"]
            path = output_dir / f"{prefix}_{stage}_detail_{number}_fixed.png"
            cameras[f"detail_{number}"] = export_view(path, (900, 900), region["center"], region["scale"])
            images.append(path.name)
        manifest["stages"].append({
            "stage": stage, "step": str(step_path), "labels": str(label_path),
            "step_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "faces": len(faces), "images": images, "cameras": cameras,
        })
        print(f"Exported {prefix} {stage}: {len(faces)} faces, {len(images)} views", flush=True)
    manifest_path = output_dir / f"{prefix}_fixed_camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Camera manifest: {manifest_path}", flush=True)

def export_fixed_view(display, path, size, camera):
    """Re-render at the requested native size, preserving the paired camera."""
    requested = size
    for _ in range(3):
        display.SetSize(*requested)
        display.OnResize()
        display.View.SetAt(*camera["center"])
        display.View.SetProj(*camera["projection"])
        display.View.SetUp(*camera["up"])
        display.View.Camera().SetScale(camera["scale"])
        display.Repaint()
        dump_png(display, path)
        header = path.read_bytes()[:24]
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Invalid PNG export: {path}")
        actual = tuple(int.from_bytes(header[i:i + 4], "big") for i in (16, 20))
        if actual == size:
            cam = display.View.Camera()
            return {"center": cam.Center().Coord(), "direction": cam.Direction().Coord(),
                    "up": cam.Up().Coord(), "scale": cam.Scale(), "aspect": cam.Aspect()}
        requested = tuple(requested[i] + size[i] - actual[i] for i in range(2))
    raise RuntimeError(f"Unable to export size {size}: {actual}")

def paired_camera(bounds, projection, up, size):
    """Frame the projected original bounds; never fit the processed endpoint."""
    import itertools

    def unit(vector):
        length = math.sqrt(sum(x * x for x in vector))
        if length == 0:
            raise ValueError("Camera vectors must be nonzero")
        return tuple(x / length for x in vector)

    direction, vertical = unit(projection), unit(up)
    horizontal = unit((direction[1] * vertical[2] - direction[2] * vertical[1],
                       direction[2] * vertical[0] - direction[0] * vertical[2],
                       direction[0] * vertical[1] - direction[1] * vertical[0]))
    vertical = unit((horizontal[1] * direction[2] - horizontal[2] * direction[1],
                     horizontal[2] * direction[0] - horizontal[0] * direction[2],
                     horizontal[0] * direction[1] - horizontal[1] * direction[0]))
    corners = list(itertools.product(*[(bounds[i], bounds[i + 3]) for i in range(3)]))
    spans = []
    for axis in (horizontal, vertical):
        projected = [sum(p[i] * axis[i] for i in range(3)) for p in corners]
        spans.append(max(projected) - min(projected))
    return {"center": tuple((bounds[i] + bounds[i + 3]) / 2 for i in range(3)),
            "projection": projection, "up": up,
            "scale": max(spans[1], spans[0] * size[1] / size[0]) * 1.08}

def show_surface_mesh(display, step_path, config, style=None, face_colors=None):
    """Display every actual OCC surface triangle, with visible triangle edges."""
    from OCC.Core.AIS import AIS_Triangulation
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.Poly import Poly_Triangulation, Poly_Triangle
    from OCC.Core.Graphic3d import (Graphic3d_MaterialAspect, Graphic3d_NOM_ALUMINIUM,
                                  Graphic3d_NOM_PLASTER)
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TColStd import TColStd_HArray1OfInteger
    from geometry.measure import load_step, bounding_box_diagonal

    shape = load_step(step_path)
    linear_deflection = bounding_box_diagonal(shape) * config["relative_deflection"]
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                                    config["angular_deflection_rad"], config["parallel"])
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError(f"Meshing did not complete: {step_path}")
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    parts = []
    part_colors = []
    for index in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(index))
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            parts.append((triangulation, location.Transformation(), face.Orientation()))
            part_colors.append((face_colors or {}).get(index))
    nodes = sum(part[0].NbNodes() for part in parts)
    triangles = sum(part[0].NbTriangles() for part in parts)
    if not triangles:
        raise RuntimeError(f"No surface triangles: {step_path}")
    merged = Poly_Triangulation(nodes, triangles, False)
    colors = TColStd_HArray1OfInteger(1, nodes) if face_colors else None
    body_rgb = style["body_rgb"] if style else (0.80, 0.84, 0.88)
    node_offset, triangle_offset = 0, 0
    for (part, transform, orientation), color in zip(parts, part_colors):
        for index in range(1, part.NbNodes() + 1):
            merged.SetNode(node_offset + index, part.Node(index).Transformed(transform))
            if colors is not None:
                # AIS vertex colors use signed ABGR integers; only color changes.
                red, green, blue = (round(value * 255) for value in (color or body_rgb))
                colors.SetValue(node_offset + index,
                                (0xFF000000 | blue << 16 | green << 8 | red) - 0x100000000)
        for index in range(1, part.NbTriangles() + 1):
            a, b, c = part.Triangle(index).Get()
            if orientation == TopAbs_REVERSED:
                b, c = c, b
            merged.SetTriangle(triangle_offset + index,
                               Poly_Triangle(node_offset + a, node_offset + b, node_offset + c))
        node_offset += part.NbNodes()
        triangle_offset += part.NbTriangles()
    merged.ComputeNormals()
    ais = AIS_Triangulation(merged)
    if colors is not None:
        ais.SetColors(colors)
    ais.Attributes().SetupOwnShadingAspect()
    material = Graphic3d_NOM_PLASTER if style else Graphic3d_NOM_ALUMINIUM
    ais.SetMaterial(Graphic3d_MaterialAspect(material))
    ais.SetColor(Quantity_Color(*body_rgb, Quantity_TOC_RGB))
    aspect = ais.Attributes().ShadingAspect().Aspect()
    if style:
        body_color = Quantity_Color(*body_rgb, Quantity_TOC_RGB)
        aspect.SetInteriorColor(body_color)
        aspect.SetBackInteriorColor(body_color)
        aspect.SetDistinguishOff()
    aspect.SetEdgeOn()
    edge_rgb = style["edge_rgb"] if style else (0.08, 0.12, 0.16)
    aspect.SetEdgeColor(Quantity_Color(*edge_rgb, Quantity_TOC_RGB))
    aspect.SetEdgeWidth(style["edge_width"] if style else 0.7)
    # AIS_Triangulation supports mode 0; the CAD viewer defaults to shaded mode 1.
    display.Context.Display(ais, 0, -1, False)
    return {"faces": face_map.Size(), "nodes": nodes, "triangles": triangles,
            "linear_deflection": linear_deflection, "mesh_decimated": False}

def comparison_display_labels(faces, labels_path):
    """An unlabelled processed endpoint is uniformly shaded, not new ground truth."""
    if labels_path is None:
        return {face_id: "background" for face_id in faces}
    return load_labels(labels_path)

def export_paper_comparisons(output_dir, model=None):
    """Export current CAD endpoints and their meshes through one shared camera."""
    from OCC.Display.OCCViewer import Viewer3d

    specification = json.loads((ROOT / "paper/manuscript/figure_comparison_sources.json").read_text(
        encoding="utf-8"))
    config = json.loads((ROOT / "results/current_three_class_complete_20261020/mesh_config.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        import subprocess
        import sys

        # A fresh native graphics context per model avoids retaining OCC display
        # resources across differently sized aircraft and triangulation scenes.
        for case in specification:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--export-paper-comparisons", str(output_dir.resolve()),
                            "--comparison-model", case["key"]], check=True)
        return
    display = Viewer3d()
    display.Create(display_glinfo=False)
    display.SetModeShaded()
    display.SetOrthographicProjection()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    for case in specification:
        if model and case["key"] != model:
            continue
        print(f"Loading {case['name']}", flush=True)
        loaded = []
        for stage in case["stages"]:
            path = ROOT / stage["step"]
            labels_path = ROOT / stage["labels"] if stage.get("labels") else None
            faces = load_faces(path)
            labels = comparison_display_labels(faces, labels_path)
            if set(faces) != set(labels):
                raise RuntimeError(f"Face mismatch in {case['key']} {stage['name']}")
            loaded.append((stage, path, labels_path, faces, labels))
        if "camera_manifest" in case:
            source = json.loads((ROOT / case["camera_manifest"]).read_text(encoding="utf-8"))
            camera = {"center": source["center"], "scale": source["scale"],
                      "projection": (0, 1, 0), "up": (0, 0, -1)}
            size = tuple(source["size"])
        else:
            size = (2400, 800)
            camera = paired_camera(bounds_of_faces(loaded[0][3]), case["projection"], case["up"], size)
        detail = (select_mesh_detail(loaded[0][3], loaded[0][4],
                                    bounds_of_faces(loaded[0][3]), case, size)
                  if "detail_semantic" in case else None)
        manifest = {"model": case["name"], "endpoint_scope": case["scope"],
                    "projection": "orthographic", "size": size, "camera": camera,
                    "geometry_modified": False, "relative_deflection": config["relative_deflection"],
                    "angular_deflection_rad": config["angular_deflection_rad"], "stages": []}
        for stage, path, labels_path, faces, labels in loaded:
            record = {"stage": stage["name"], "step": str(path),
                      "labels": str(labels_path) if labels_path else None,
                      "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest() if labels_path else None}
            if labels_path is None:
                record["label_scope"] = "Uniform background display only; no processed semantic ground truth."
            if case["key"] != "airbus":
                display.EraseAll()
                show_faces(display, faces, labels)
                image_path = output_dir / f"cad_{case['key']}_{stage['name']}.png"
                record["cad_image"] = image_path.name
                record["cad_camera"] = export_fixed_view(display, image_path, size, camera)
                if detail is not None:
                    detail_path = output_dir / f"cad_{case['key']}_{stage['name']}_detail.png"
                    record["cad_detail_image"] = detail_path.name
                    record["cad_detail_camera"] = export_fixed_view(display, detail_path, size, detail)
            if case.get("export_mesh", True):
                display.EraseAll()
                record["mesh"] = show_surface_mesh(display, path, config)
                image_path = output_dir / f"mesh_{case['key']}_{stage['name']}.png"
                record["mesh_image"] = image_path.name
                record["mesh_camera"] = export_fixed_view(display, image_path, size, camera)
            manifest["stages"].append(record)
            summary = f"{record['mesh']['triangles']} triangles" if "mesh" in record else "paired CAD view"
            print(f"Exported {case['key']} {stage['name']}: {summary}", flush=True)
        output = output_dir / f"comparison_{case['key']}_manifest.json"
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {output}", flush=True)

def select_mesh_detail(faces, labels, bounds, case, size):
    """Locate a labelled feature on the observed side of the original model."""
    projection = case["projection"]
    length_axis = case["length_axis"]
    candidates = []
    for face_id, face in faces.items():
        if labels[face_id] != case["detail_semantic"]:
            continue
        box = bounds_of_faces({face_id: face})
        center = tuple((box[i] + box[i + 3]) / 2 for i in range(3))
        candidates.append((face_id, center))
    if not candidates:
        raise ValueError(f"No {case['detail_semantic']} detail on {case['name']}")
    side_positions = [sum(center[i] * projection[i] for i in range(3)) for _, center in candidates]
    halfway = (min(side_positions) + max(side_positions)) / 2
    visible_side = [item for item, position in zip(candidates, side_positions) if position >= halfway]
    visible_side.sort(key=lambda item: item[1][length_axis])
    face_id, center = visible_side[len(visible_side) // 2]
    horizontal_span = (bounds[length_axis + 3] - bounds[length_axis]) * case["detail_fraction"]
    return {"original_face_id": face_id, "semantic": case["detail_semantic"],
            "center": center, "projection": projection, "up": case["up"],
            "scale": horizontal_span * size[1] / size[0]}

def select_extra_mesh_details(faces, labels, case, size):
    """Frame explicit source-labelled regions without reusing processed face IDs."""
    regions = []
    for request in case.get("extra_details", []):
        ids = request["original_face_ids"]
        if not ids or any(face_id not in faces or labels.get(face_id) != request["semantic"]
                          for face_id in ids):
            raise ValueError(f"Invalid source faces for {request['name']}")
        camera = paired_camera(bounds_of_faces({face_id: faces[face_id] for face_id in ids}),
                               case["projection"], case["up"], size)
        regions.append({"name": request["name"], "semantic": request["semantic"],
                        "original_face_ids": ids, **camera})
    return regions

def export_figure11(output_dir, model=None):
    """Render the exact table endpoints, with whole-aircraft and local side views."""
    from OCC.Display.OCCViewer import Viewer3d

    cases = json.loads((ROOT / "paper/manuscript/figure11_mesh_sources.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "results/current_three_class_complete_20261020/mesh_config.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        import subprocess
        import sys

        for case in cases:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--export-figure11", str(output_dir.resolve()),
                            "--figure11-model", case["key"]], check=True)
        return
    case = next(case for case in cases if case["key"] == model)
    evidence_root = ROOT / "results/current_three_class_complete_20261020/mesh"
    metrics = [json.loads((evidence_root / f"{case['metric_prefix']}_{stage}.json").read_text(
        encoding="utf-8")) for stage in ("original", "simplified")]
    paths = [ROOT / item["input_step"] for item in metrics]
    original_faces = load_faces(paths[0])
    labels_path = ROOT / case["labels"]
    labels = load_labels(labels_path)
    if set(labels) != set(original_faces):
        raise RuntimeError(f"Original label mismatch: {case['name']}")
    bounds = bounds_of_faces(original_faces)
    size = (2400, 800)
    camera = paired_camera(bounds, case["projection"], case["up"], size)
    detail = select_mesh_detail(original_faces, labels, bounds, case, size)
    extra_details = select_extra_mesh_details(original_faces, labels, case, size)
    views = [("overview", camera), ("detail", detail)] + [
        (region["name"], region) for region in extra_details]
    manifest = {"model": case["name"], "scope": "Current original and final STEP endpoints of the mesh comparison table",
                "size": size, "camera": camera, "detail": detail,
                "render_style": {**FIGURE11_STYLE, "material": "plaster", "transparency": 0.0},
                "relative_deflection": config["relative_deflection"],
                "angular_deflection_rad": config["angular_deflection_rad"],
                "geometry_modified": False, "stages": []}
    if extra_details:
        manifest["extra_details"] = extra_details
    display = Viewer3d()
    display.Create(display_glinfo=False)
    display.SetModeShaded()
    display.SetOrthographicProjection()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    for index, step_path in enumerate(paths):
        stage = "before" if index == 0 else "after"
        print(f"Meshing {case['name']} {stage}", flush=True)
        display.EraseAll()
        face_colors = None
        if index == 0 and case.get("color_detail_face"):
            face_colors = {detail["original_face_id"]: COLORS[detail["semantic"]].Values(Quantity_TOC_RGB)}
        actual = show_surface_mesh(display, step_path, config, style=FIGURE11_STYLE,
                                   face_colors=face_colors)
        expected = metrics[index]
        for field, key in (("faces", "face_count"), ("nodes", "mesh_node_count"),
                           ("triangles", "mesh_triangle_count")):
            if actual[field] != expected[key]:
                raise RuntimeError(f"Table/mesh mismatch: {case['name']} {stage} {field}: "
                                   f"{actual[field]} vs {expected[key]}")
        record = {"stage": stage, "step": str(step_path),
                  "step_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest(),
                  "metric_source": str(evidence_root / f"{case['metric_prefix']}_{('original', 'simplified')[index]}.json"),
                  "mesh": actual, "images": {}, "cameras": {}}
        if face_colors:
            record["colored_original_faces"] = {str(detail["original_face_id"]): detail["semantic"]}
        for view, settings in views:
            key = f"mesh_{view}"
            image_path = output_dir / f"fig11_{case['key']}_{stage}_{key}.png"
            record["images"][key] = image_path.name
            record["cameras"][key] = export_fixed_view(display, image_path, size, settings)
        if stage == "after":
            display.EraseAll()
            final_faces = load_faces(step_path)
            show_faces(display, final_faces, {face_id: "background" for face_id in final_faces},
                       style=FIGURE11_STYLE)
            for view, settings in views:
                key = f"cad_{view}"
                image_path = output_dir / f"fig11_{case['key']}_{stage}_{key}.png"
                record["images"][key] = image_path.name
                record["cameras"][key] = export_fixed_view(display, image_path, size, settings)
        manifest["stages"].append(record)
        print(f"Exported {case['key']} {stage}: {actual['triangles']} triangles", flush=True)
    manifest["original_labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    output = output_dir / f"fig11_{case['key']}_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {output}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, default=STEP_PATH)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    parser.add_argument("--title", default="G280 | original shading, near-black face boundaries")
    parser.add_argument("--view", choices=("side", "top", "overall"), default="side")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--detail-scale", type=float, default=2.0)
    parser.add_argument("--export-figure6", type=Path, metavar="DIRECTORY")
    parser.add_argument("--export-figure6-horizontal", type=Path, metavar="DIRECTORY")
    parser.add_argument("--export-figure7", type=Path, metavar="DIRECTORY")
    parser.add_argument("--figure7-model", choices=("airbus", "cessna"))
    parser.add_argument("--export-paper-comparisons", type=Path, metavar="DIRECTORY")
    parser.add_argument("--comparison-model", choices=("airbus", "dc10", "cessna", "109", "airplane_body"))
    parser.add_argument("--export-figure11", type=Path, metavar="DIRECTORY")
    parser.add_argument("--figure11-model", choices=("aulira", "dc10", "airbus", "gulfstream", "airplane_body"))
    args = parser.parse_args()
    if args.detail_scale <= 0:
        parser.error("--detail-scale must be positive")
    if sum(value is not None for value in (
            args.export_figure6, args.export_figure6_horizontal,
            args.export_figure7, args.export_paper_comparisons,
            args.export_figure11)) > 1:
        parser.error("Choose one figure export at a time")
    if args.comparison_model and args.export_paper_comparisons is None:
        parser.error("--comparison-model requires --export-paper-comparisons")
    if args.figure11_model and args.export_figure11 is None:
        parser.error("--figure11-model requires --export-figure11")
    if args.figure7_model and args.export_figure7 is None:
        parser.error("--figure7-model requires --export-figure7")
    if args.export_figure11 is not None:
        export_figure11(args.export_figure11, args.figure11_model)
        return
    if args.export_paper_comparisons is not None:
        export_paper_comparisons(args.export_paper_comparisons, args.comparison_model)
        return
    if args.export_figure7 is not None:
        if args.figure7_model is None:
            export_figure7(args.export_figure7)
        else:
            export_figure7(args.export_figure7, args.figure7_model)
        return
    if args.export_figure6 is not None:
        export_figure6(args.export_figure6)
        return
    if args.export_figure6_horizontal is not None:
        export_figure6(args.export_figure6_horizontal, horizontal=True)
        return
    faces = load_faces(args.step)
    labels = load_labels(args.labels)
    if set(faces) != set(labels):
        raise RuntimeError(f"Face mismatch: STEP={len(faces)}, labels={len(labels)}")

    display, start_display, add_menu, add_function_to_menu = init_display(
        backend_str="tk", size=(1400, 900), display_triedron=False
    )
    canvas = display.get_parent()
    window = canvas.winfo_toplevel()
    window.title(args.title)
    window.geometry("1400x900")
    canvas.pack(fill="both", expand=True)

    def resize_view(_event):
        display.OnResize()
        display.Repaint()

    canvas.bind("<Configure>", resize_view)
    window.update_idletasks()
    display.OnResize()
    shown = show_faces(display, faces, labels)

    def Overall_view():
        # The STEP model uses Y for height; Z is the side-view direction.
        display.View.SetProj(-0.45, -0.85, 1.4)
        display.View.SetUp(0.0, 1.0, 0.0)
        display.FitAll()
        display.Repaint()

    def Side_view():
        display.View.SetProj(0.0, 0.0, 1.0)
        display.View.SetUp(0.0, 1.0, 0.0)
        display.FitAll()
        display.Repaint()

    def Top_view():
        # The aircraft's vertical axis is Y; negative Z points up in this view.
        display.View.SetProj(0.0, 1.0, 0.0)
        display.View.SetUp(0.0, 0.0, -1.0)
        display.FitAll()
        display.Repaint()

    def Show_CAD_edges():
        for ais, _ in shown:
            ais.Attributes().SetFaceBoundaryDraw(True)
            display.Context.Redisplay(ais, False)
        display.Repaint()

    def Hide_body_edges():
        for ais, semantic in shown:
            ais.Attributes().SetFaceBoundaryDraw(semantic != "background")
            display.Context.Redisplay(ais, False)
        display.Repaint()

    add_menu("View")
    for action in (Overall_view, Side_view, Top_view, Show_CAD_edges, Hide_body_edges):
        add_function_to_menu("View", action)
    startup_view = {"side": Side_view, "top": Top_view, "overall": Overall_view}[args.view]

    def initialize_view():
        display.OnResize()
        startup_view()
        if args.center is not None:
            display.View.SetAt(*args.center)
            display.View.Camera().SetScale(args.detail_scale)
            display.Repaint()

    initialize_view()
    # Fit again after Tk maps the canvas at its actual window dimensions.
    window.after(250, initialize_view)
    start_display()


def wireframe_main():
    """Open the final STEP with its real boundary edges."""
    import argparse
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_TOC_RGB
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from geometry.measure import load_step

    parser = argparse.ArgumentParser(description="Display a STEP endpoint and its boundary edges.")
    parser.add_argument('step_model', type=Path)
    args = parser.parse_args()
    shape = load_step(args.step_model)
    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, color=Quantity_NOC_GRAY, transparency=0.82, update=False)
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while explorer.More():
        edges.append(explorer.Current())
        explorer.Next()
    display.DisplayShape(edges, color=Quantity_Color(0., 0., 0., Quantity_TOC_RGB), update=False)
    display.FitAll()
    display.Repaint()
    start_display()
