"""Shared STEP loading, paired geometric measurements and surface-mesh statistics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location

def load_step(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:
        raise RuntimeError(f"Cannot read STEP: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP contains no transferable shape: {path}")
    return shape

def bounding_box_diagonal(shape) -> float:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    if box.IsVoid():
        raise RuntimeError("Cannot calculate a bounding box for the shape.")
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))

def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]

def distance(first, second) -> float:
    return math.dist(
        (first.X(), first.Y(), first.Z()),
        (second.X(), second.Y(), second.Z()),
    )

def triangle_quality(first, second, third) -> tuple[float, float] | None:
    sides = (distance(second, third), distance(first, third), distance(first, second))
    semiperimeter = sum(sides) / 2.0
    area_squared = semiperimeter
    for side in sides:
        area_squared *= semiperimeter - side
    if area_squared <= 1.0e-24:
        return None
    area = math.sqrt(area_squared)
    angles = []
    for opposite, adjacent_one, adjacent_two in (
        (sides[0], sides[1], sides[2]),
        (sides[1], sides[0], sides[2]),
        (sides[2], sides[0], sides[1]),
    ):
        cosine = (
            (adjacent_one ** 2 + adjacent_two ** 2 - opposite ** 2)
            / (2.0 * adjacent_one * adjacent_two)
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    aspect_ratio = max(sides) ** 2 / (2.0 * math.sqrt(3.0) * area)
    return min(angles), aspect_ratio

def mesh_statistics(shape) -> tuple[int, int, int, dict[str, float | int]]:
    face_count = 0
    node_count = 0
    triangle_count = 0
    minimum_angles = []
    aspect_ratios = []
    degenerate_triangle_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face_count += 1
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(explorer.Current(), location)
        if triangulation is not None:
            node_count += triangulation.NbNodes()
            triangle_count += triangulation.NbTriangles()
            transformation = location.Transformation()
            for index in range(1, triangulation.NbTriangles() + 1):
                first_index, second_index, third_index = triangulation.Triangle(index).Get()
                quality = triangle_quality(
                    triangulation.Node(first_index).Transformed(transformation),
                    triangulation.Node(second_index).Transformed(transformation),
                    triangulation.Node(third_index).Transformed(transformation),
                )
                if quality is None:
                    degenerate_triangle_count += 1
                else:
                    minimum_angles.append(quality[0])
                    aspect_ratios.append(quality[1])
        explorer.Next()
    quality_summary = {
        "degenerate_triangle_count": degenerate_triangle_count,
        "minimum_angle_deg": min(minimum_angles) if minimum_angles else None,
        "minimum_angle_p05_deg": percentile(minimum_angles, 0.05) if minimum_angles else None,
        "maximum_aspect_ratio": max(aspect_ratios) if aspect_ratios else None,
        "aspect_ratio_p95": percentile(aspect_ratios, 0.95) if aspect_ratios else None,
    }
    return face_count, node_count, triangle_count, quality_summary

def mesh_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("--relative-deflection", type=float, required=True)
    parser.add_argument("--angular-deflection-rad", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 < args.relative_deflection < 1:
        raise ValueError("relative deflection must be in (0, 1).")
    if not 0 < args.angular_deflection_rad < math.pi:
        raise ValueError("angular deflection must be in (0, pi).")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1.")

    mesh_elapsed_samples = []
    shape = None
    diagonal = 0.0
    linear_deflection = 0.0
    for _ in range(args.repeats):
        shape = load_step(args.step)
        diagonal = bounding_box_diagonal(shape)
        linear_deflection = diagonal * args.relative_deflection
        started = time.perf_counter()
        mesher = BRepMesh_IncrementalMesh(
            shape,
            linear_deflection,
            False,
            args.angular_deflection_rad,
            True,
        )
        mesher.Perform()
        mesh_elapsed_samples.append(time.perf_counter() - started)
        if not mesher.IsDone():
            raise RuntimeError(f"Meshing did not complete: {args.step}")

    assert shape is not None
    face_count, node_count, triangle_count, quality_summary = mesh_statistics(shape)
    payload = {
        "input_step": str(args.step),
        "bounding_box_diagonal": diagonal,
        "relative_deflection": args.relative_deflection,
        "linear_deflection": linear_deflection,
        "angular_deflection_rad": args.angular_deflection_rad,
        "face_count": face_count,
        "mesh_node_count": node_count,
        "mesh_triangle_count": triangle_count,
        "mesh_quality": quality_summary,
        "mesh_repeats": args.repeats,
        "mesh_elapsed_seconds_samples": mesh_elapsed_samples,
        "mesh_elapsed_seconds_median": statistics.median(mesh_elapsed_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import topexp_MapShapes, topexp_MapShapesAndAncestors
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

def surface_area(shape) -> float:
    area_props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, area_props)
    return area_props.Mass()

def free_edge_count(shape) -> int:
    edges = TopTools_IndexedMapOfShape()
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapes(shape, TopAbs_EDGE, edges)
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    count = 0
    for index in range(1, edges.Size() + 1):
        edge_index = edge_faces.FindIndex(edges.FindKey(index))
        if edge_index > 0 and edge_faces.FindFromIndex(edge_index).Size() == 1:
            count += 1
    return count

def mesh_sample_points(shape, linear_deflection: float, angular_deflection: float, maximum_samples: int):
    mesher = BRepMesh_IncrementalMesh(
        shape, linear_deflection, False, angular_deflection, True
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("Could not mesh shape for sampling.")

    points = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(explorer.Current(), location)
        if triangulation is not None:
            transformation = location.Transformation()
            for index in range(1, triangulation.NbNodes() + 1):
                points.append(triangulation.Node(index).Transformed(transformation))
        explorer.Next()
    if not points:
        raise RuntimeError("No surface mesh nodes available for sampling.")
    stride = max(1, math.ceil(len(points) / maximum_samples))
    return points[::stride][:maximum_samples]

def distances_to_shape(points, target_shape) -> dict[str, float | int]:
    values = []
    for point in points:
        vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
        distance = BRepExtrema_DistShapeShape(vertex, target_shape)
        distance.Perform()
        if not distance.IsDone():
            raise RuntimeError("Point-to-shape distance query failed.")
        values.append(distance.Value())
    return {
        "sample_count": len(values),
        "mean": sum(values) / len(values),
        "rms": math.sqrt(sum(value * value for value in values) / len(values)),
        "maximum": max(values),
    }

def relative_change(original: float, simplified: float) -> float | None:
    if original == 0:
        return None
    return (simplified - original) * 100.0 / original

def compare_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("simplified", type=Path)
    parser.add_argument("--linear-deflection", type=float, required=True)
    parser.add_argument("--angular-deflection-rad", type=float, required=True)
    parser.add_argument("--maximum-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.maximum_samples < 1:
        raise ValueError("maximum-samples must be at least 1.")

    original_shape = load_step(args.original)
    simplified_shape = load_step(args.simplified)
    original_area = surface_area(original_shape)
    simplified_area = surface_area(simplified_shape)
    original_free_edges = free_edge_count(original_shape)
    simplified_free_edges = free_edge_count(simplified_shape)
    original_points = mesh_sample_points(
        original_shape, args.linear_deflection, args.angular_deflection_rad,
        args.maximum_samples,
    )
    simplified_points = mesh_sample_points(
        simplified_shape, args.linear_deflection, args.angular_deflection_rad,
        args.maximum_samples,
    )

    payload = {
        "original_step": str(args.original),
        "simplified_step": str(args.simplified),
        "sampling_linear_deflection": args.linear_deflection,
        "sampling_angular_deflection_rad": args.angular_deflection_rad,
        "free_edge_count": {
            "original": original_free_edges,
            "simplified": simplified_free_edges,
        },
        "surface_area": {
            "original": original_area,
            "simplified": simplified_area,
            "relative_change_percent": relative_change(original_area, simplified_area),
        },
        "sampled_distance": {
            "original_to_simplified": distances_to_shape(original_points, simplified_shape),
            "simplified_to_original": distances_to_shape(simplified_points, original_shape),
            "scope": "Distances include deliberately removed feature regions; maximum distance therefore captures both intended feature removal and any unintended geometric change.",
        },
    }
    if original_free_edges == 0 and simplified_free_edges == 0:
        original_volume_props = GProp_GProps()
        simplified_volume_props = GProp_GProps()
        brepgprop_VolumeProperties(original_shape, original_volume_props)
        brepgprop_VolumeProperties(simplified_shape, simplified_volume_props)
        original_volume = original_volume_props.Mass()
        simplified_volume = simplified_volume_props.Mass()
        payload["volume"] = {
            "interpretable": True,
            "original": original_volume,
            "simplified": simplified_volume,
            "relative_change_percent": relative_change(original_volume, simplified_volume),
        }
    else:
        payload["volume"] = {
            "interpretable": False,
            "reason": "At least one STEP has free edges, so volume is not treated as a comparable metric.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
