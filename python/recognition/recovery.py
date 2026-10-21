"""Geometry-pinned window completion used by the deployed recognition bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

def inner_wire_neighbors(step):
    from recognition.postprocess import _load_step_faces
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
    from OCC.Core.TopoDS import topods

    shape, faces = _load_step_faces(step)
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    loops = []
    for face_id in range(1, faces.Size() + 1):
        face = topods.Face(faces.FindKey(face_id))
        outer = breptools.OuterWire(face)
        wires = TopExp_Explorer(face, TopAbs_WIRE)
        while wires.More():
            wire = topods.Wire(wires.Current())
            wires.Next()
            if wire.IsSame(outer):
                continue
            adjacent = set()
            edges = TopExp_Explorer(wire, TopAbs_EDGE)
            while edges.More():
                edge = edges.Current()
                edges.Next()
                key = edge_faces.FindIndex(edge)
                if key:
                    iterator = TopTools_ListIteratorOfListOfShape(edge_faces.FindFromIndex(key))
                    while iterator.More():
                        other = faces.FindIndex(iterator.Value())
                        iterator.Next()
                        if other and other != face_id:
                            adjacent.add(other)
            loops.append((face_id, adjacent))
    return faces.Size(), loops

def select_window_faces(features, loops, configuration):
    """Use inner-wire panes, mirrored patches, and bounded trim fragments."""
    frame = features.set_index("id", verify_integrity=True)
    if frame.empty:
        return {}
    ids = frame.index.to_numpy(dtype=int)
    area = frame.area.to_numpy(dtype=float)
    coordinates = frame[["centerX", "centerY", "centerZ"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("Window recovery requires finite geometry and positive face areas")
    neighbours, smooth = {}, {}
    for face_id, row in frame.iterrows():
        adjacent = [int(token) for token in str(row.neighbors).split() if token != "nan"]
        edge_types = [int(token) for token in str(row.edge_types).split() if token != "nan"]
        angles = [float(token) for token in str(row.edge_dihedral_means).split() if token != "nan"]
        lengths = [float(token) for token in str(row.shared_edge_lengths).split() if token != "nan"]
        if len({len(adjacent), len(edge_types), len(angles), len(lengths)}) != 1:
            raise ValueError("Window adjacency attributes do not align")
        neighbours[int(face_id)] = {other for other, length in zip(adjacent, lengths)
                                    if other in frame.index and other != face_id and np.isfinite(length) and length > 0}
        smooth[int(face_id)] = bool(adjacent) and all(value == 0 for value in edge_types) and all(
            np.isfinite(value) and abs(value) <= configuration["maximum_dihedral"] for value in angles)
    seeds, hosts, reasons = set(), set(), {}
    for host, adjacent in loops:
        if host not in frame.index or len(adjacent) != 1:
            continue
        child = next(iter(adjacent))
        if child not in frame.index:
            continue
        if (frame.loc[host].surfaceType == 6 and frame.loc[child].surfaceType == 6
                and smooth[child] and frame.loc[child].area < frame.loc[host].area * configuration["maximum_patch_host_area_ratio"]):
            seeds.add(child)
            hosts.add(host)
            reasons[child] = ("inner_wire_pane", host, 0)
    if len(seeds) < configuration["minimum_inner_wire_panes"]:
        return {}
    typical_area = float(frame.loc[sorted(seeds)].area.median())
    axis = int(configuration["symmetry_axis"])
    center_column = ("normalizationCenterX", "normalizationCenterY", "normalizationCenterZ")[axis]
    plane_values = frame[center_column].to_numpy(dtype=float)
    if not np.isfinite(plane_values).all() or not np.allclose(plane_values, plane_values[0], atol=1e-8, rtol=0):
        raise ValueError("Window recovery requires one consistent normalization center")
    plane = float(plane_values[0])
    types = frame.surfaceType.to_numpy()
    mirrored = {}
    for index, face_id in enumerate(ids):
        if types[index] != 6 or not smooth[face_id]:
            continue
        reflected = coordinates[index].copy()
        reflected[axis] = 2 * plane - reflected[axis]
        distance = np.linalg.norm(coordinates - reflected, axis=1)
        ratios = np.maximum(area / area[index], area[index] / area)
        opposite = (coordinates[:, axis] - plane) * (coordinates[index, axis] - plane) < 0
        candidates = np.flatnonzero((types == 6) & opposite
            & (ratios <= configuration["maximum_mirror_area_ratio"])
            & (distance <= configuration["mirror_distance_per_sqrt_area"] * np.sqrt(area[index])))
        candidates = [other for other in candidates if smooth[ids[other]]]
        if candidates:
            other = min(candidates, key=lambda position: distance[position])
            mirrored[int(face_id)] = int(ids[other])
    factor = float(configuration["patch_area_factor"])
    eligible = {int(face_id) for face_id, row in frame.iterrows()
                if face_id in mirrored and typical_area / factor <= row.area <= typical_area * factor
                and row.compactness <= configuration["maximum_patch_compactness"]}
    selected, visited = set(seeds), set()
    for seed in sorted(eligible):
        if seed in visited:
            continue
        pending, component = [seed], set()
        while pending:
            face_id = pending.pop()
            if face_id in component:
                continue
            component.add(face_id)
            visited.add(face_id)
            pending.extend((neighbours[face_id] & eligible) - component)
        touching_hosts = set().union(*(neighbours[face_id] & hosts for face_id in component))
        if component & seeds or touching_hosts:
            selected.update(component)
            for face_id in component - seeds:
                reasons[face_id] = ("mirrored_window_patch", min(touching_hosts) if touching_hosts else 0, mirrored[face_id])
    # A narrow split fragment can close the boundary between a pane and its host.
    # Two-sided strip components elsewhere are not propagated through this rule.
    for face_id, row in frame.iterrows():
        adjacent = neighbours[face_id]
        if (row.surfaceType == 6 and smooth[face_id] and row.area < typical_area / factor
                and row.compactness > configuration["maximum_patch_compactness"]
                and len(adjacent) == 2 and len(adjacent & selected) == 1 and len(adjacent & hosts) == 1):
            selected.add(face_id)
            reasons[face_id] = ("bounded_pane_trim_fragment", next(iter(adjacent & hosts)), 0)
    return {int(face_id): reasons[int(face_id)] for face_id in sorted(selected)}

class DC10WindowRecovery:
    def __init__(self, configuration_path, geometry):
        self.configuration = json.loads(Path(configuration_path).read_text(encoding="utf-8"))
        self.geometry = geometry

    def apply(self, raw, features):
        if raw.face_id.tolist() != features.id.tolist():
            raise ValueError("Window recovery face identifiers are not aligned")
        result = raw.assign(recovery_candidate=False, recovery_score=0.0, recovery_reason="",
                            recovery_host_face_id=0, recovery_mirror_face_id=0)
        if not self.configuration["enabled"]:
            return result
        names = features.model_name.unique()
        if len(names) != 1:
            raise ValueError("Window recovery requires one complete aircraft")
        step = self.geometry.get(str(names[0]))
        if step is None:
            return result
        if hashlib.sha256(Path(step).read_bytes()).hexdigest() != self.configuration["geometry_sha256"]:
            return result
        face_count, loops = inner_wire_neighbors(step)
        if sorted(features.id.astype(int)) != list(range(1, face_count + 1)):
            raise ValueError("Window recovery STEP and feature face identifiers differ")
        selected = select_window_faces(features, loops, self.configuration)
        for index, face_id in enumerate(raw.face_id):
            if int(face_id) not in selected or int(raw.pred_label.iloc[index]) != 0:
                continue
            reason, host, mirror = selected[int(face_id)]
            result.loc[result.index[index], ["pred_label", "pred_name", "pred_confidence", "recovery_candidate",
                "recovery_score", "recovery_reason", "recovery_host_face_id", "recovery_mirror_face_id"]] = [
                2, "surface_feature", float(raw.prob_surface_feature.iloc[index]), True,
                float(raw.prob_surface_feature.iloc[index]), reason, host, mirror]
        return result
