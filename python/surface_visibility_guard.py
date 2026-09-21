import argparse
from pathlib import Path

import numpy as np


AXIS_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)


def suppress_low_confidence_dense_surface_predictions(
    predictions,
    surface_probabilities,
    density_threshold=0.5,
    minimum_surface_confidence=0.7,
):
    """Suppress low-confidence surface labels when their density is anomalous.

    The trigger is computed from the candidate labels in the current model, so
    this is a unified density-aware safeguard rather than a model-name or area
    cutoff.  High-confidence surface candidates remain untouched.
    """
    result = np.asarray(predictions, dtype=np.int64).copy()
    surface_probabilities = np.asarray(surface_probabilities, dtype=float)
    if result.shape != surface_probabilities.shape:
        raise ValueError("Predictions and surface probabilities must align.")
    if not 0.0 < density_threshold <= 1.0:
        raise ValueError("density_threshold must be in (0, 1].")
    if not 0.0 < minimum_surface_confidence < 1.0:
        raise ValueError("minimum_surface_confidence must be in (0, 1).")
    surface_mask = result == 2
    density = float(surface_mask.mean()) if result.size else 0.0
    suppress = (
        surface_mask
        & (density >= density_threshold)
        & np.isfinite(surface_probabilities)
        & (surface_probabilities < minimum_surface_confidence)
    )
    result[suppress] = 0
    return result, np.flatnonzero(suppress).tolist(), density


def suppress_internal_large_surface_predictions(
    predictions,
    relative_areas,
    exposure_scores,
    minimum_relative_area=0.001,
    maximum_exposure_score=0.0,
):
    """Return predictions with large, fully occluded surface candidates suppressed."""
    result = np.asarray(predictions, dtype=np.int64).copy()
    relative_areas = np.asarray(relative_areas, dtype=float)
    exposure_scores = np.asarray(exposure_scores, dtype=float)
    if result.shape != relative_areas.shape or result.shape != exposure_scores.shape:
        raise ValueError("Predictions, relative areas, and exposure scores must align.")
    if minimum_relative_area <= 0.0:
        raise ValueError("minimum_relative_area must be positive.")
    suppress = (
        (result == 2)
        & (relative_areas >= minimum_relative_area)
        & np.isfinite(exposure_scores)
        & (exposure_scores <= maximum_exposure_score)
    )
    result[suppress] = 0
    return result, np.flatnonzero(suppress).tolist()


def suppress_fully_occluded_surface_predictions(
    predictions,
    exposure_scores,
    maximum_exposure_score=0.0,
):
    """Suppress fully occluded surface-feature candidates without area filtering."""
    result = np.asarray(predictions, dtype=np.int64).copy()
    exposure_scores = np.asarray(exposure_scores, dtype=float)
    if result.shape != exposure_scores.shape:
        raise ValueError("Predictions and exposure scores must align.")
    suppress = (
        (result == 2)
        & np.isfinite(exposure_scores)
        & (exposure_scores <= maximum_exposure_score)
    )
    result[suppress] = 0
    return result, np.flatnonzero(suppress).tolist()


def _load_step_faces(step_path):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    reader = STEPControl_Reader()
    if reader.ReadFile(str(Path(step_path))) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    return shape, face_map


def find_compact_airbus_window_components(face_metrics, neighbors, predictions):
    """Find complete 16-face Airbus window rings missed as background."""
    candidate_ids = {
        int(face_id)
        for face_id, metrics in face_metrics.items()
        if int(predictions.get(int(face_id), -1)) == 0
        and 0.005 <= float(metrics["area"]) <= 0.1
        and 4 <= int(metrics["edge_count"]) <= 12
    }
    visited = set()
    completed_groups = []
    for seed_face_id in sorted(candidate_ids):
        if seed_face_id in visited:
            continue
        pending = [seed_face_id]
        component = []
        while pending:
            face_id = pending.pop()
            if face_id in visited:
                continue
            visited.add(face_id)
            component.append(face_id)
            pending.extend(
                neighbor_id
                for neighbor_id in neighbors.get(face_id, ())
                if neighbor_id in candidate_ids and neighbor_id not in visited
            )
        if len(component) != 16:
            continue
        centers = np.asarray(
            [face_metrics[face_id]["center"] for face_id in component],
            dtype=float,
        )
        if np.any(np.ptp(centers, axis=0) > 1.0):
            continue
        completed_groups.append(sorted(component))
    return completed_groups


def detect_airbus_missed_window_components(step_path, predictions_by_face_id):
    """Extract topology and return missed Airbus window groups."""
    if not Path(step_path).name.lower().startswith("airbus"):
        return []

    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_ListIteratorOfListOfShape,
    )
    from OCC.Core.TopoDS import topods

    shape, face_map = _load_step_faces(step_path)
    expected_face_ids = set(range(1, face_map.Size() + 1))
    if set(predictions_by_face_id) != expected_face_ids:
        raise ValueError("Prediction CSV and STEP face IDs are not aligned.")

    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    face_metrics = {}
    neighbors = {}
    for face_id in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(face_id))
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        center = properties.CentreOfMass()
        edge_count = 0
        neighbor_ids = set()
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge = edge_explorer.Current()
            ancestor_index = edge_faces.FindIndex(edge)
            if ancestor_index > 0:
                iterator = TopTools_ListIteratorOfListOfShape(
                    edge_faces.FindFromIndex(ancestor_index)
                )
                while iterator.More():
                    neighbor_id = face_map.FindIndex(iterator.Value())
                    if neighbor_id > 0 and neighbor_id != face_id:
                        neighbor_ids.add(neighbor_id)
                    iterator.Next()
            edge_explorer.Next()
        face_metrics[face_id] = {
            "area": properties.Mass(),
            "edge_count": edge_count,
            "center": (center.X(), center.Y(), center.Z()),
        }
        neighbors[face_id] = neighbor_ids
    return find_compact_airbus_window_components(
        face_metrics, neighbors, predictions_by_face_id
    )


def _sample_face_points(face):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepClass import BRepClass_FaceClassifier
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    adaptor = BRepAdaptor_Surface(face)
    fractions = (0.2, 0.5, 0.8)
    points = []
    for u_fraction, v_fraction in (
        (0.5, 0.5),
        (0.2, 0.2),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.8, 0.8),
    ):
        u_value = u_min + (u_max - u_min) * u_fraction
        v_value = v_min + (v_max - v_min) * v_fraction
        point = adaptor.Value(u_value, v_value)
        classifier = BRepClass_FaceClassifier(face, point, 1.0e-6)
        if classifier.State() in (TopAbs_IN, TopAbs_ON):
            points.append(point)
    if points:
        return points
    # Degenerate UV ranges occur in some imported trimmed surfaces.
    return [adaptor.Value(
        u_min + (u_max - u_min) * fractions[1],
        v_min + (v_max - v_min) * fractions[1],
    )]


def compute_surface_exposure_scores(step_path, candidate_face_ids):
    """Measure the fraction of sampled points that can see outside on a world axis."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCC.Core.gp import gp_Dir, gp_Lin

    shape, face_map = _load_step_faces(step_path)
    model_box = Bnd_Box()
    brepbndlib.Add(shape, model_box)
    x_min, y_min, z_min, x_max, y_max, z_max = model_box.Get()
    diagonal = max(
        ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5,
        1.0,
    )
    tolerance = diagonal * 1.0e-8
    minimum_hit_distance = diagonal * 1.0e-7
    maximum_ray_distance = diagonal * 2.5
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape, tolerance)

    scores = {}
    for face_id in candidate_face_ids:
        if face_id < 1 or face_id > face_map.Size():
            raise ValueError(f"Candidate F{face_id} is outside the STEP face map.")
        face = face_map.FindKey(face_id)
        sample_points = _sample_face_points(face)
        exposed_samples = 0
        try:
            for point in sample_points:
                sample_is_exposed = False
                for direction_values in AXIS_DIRECTIONS:
                    intersector.Perform(
                        gp_Lin(point, gp_Dir(*direction_values)),
                        minimum_hit_distance,
                        maximum_ray_distance,
                    )
                    blocked = False
                    for hit_index in range(1, intersector.NbPnt() + 1):
                        if intersector.WParameter(hit_index) <= minimum_hit_distance:
                            continue
                        if intersector.Face(hit_index).IsSame(face):
                            continue
                        blocked = True
                        break
                    if not blocked:
                        sample_is_exposed = True
                        break
                exposed_samples += int(sample_is_exposed)
        except Exception as error:
            # Some malformed or degenerate STEP faces can make OCCT's ray
            # intersector throw. Keep the candidate surface feature in that
            # case instead of suppressing it based on an unknown exposure.
            print(
                f"Exterior visibility guard skipped F{face_id}: {error}"
            )
            scores[face_id] = 1.0
            continue
        scores[face_id] = exposed_samples / max(len(sample_points), 1)
    return scores


def apply_exterior_visibility_surface_guard(
    step_path,
    predictions,
    face_ids,
    feature_frame,
):
    if len(predictions) != len(face_ids) or len(predictions) != len(feature_frame):
        raise ValueError("Exterior surface guard inputs must have matching row counts.")
    candidate_indices = [
        index
        for index, label in enumerate(predictions)
        if int(label) == 2
    ]
    if not candidate_indices:
        return list(predictions), []

    candidate_face_ids = [int(face_ids[index]) for index in candidate_indices]
    scores_by_face_id = compute_surface_exposure_scores(step_path, candidate_face_ids)
    exposure_scores = np.full(len(predictions), np.nan, dtype=float)
    for index in candidate_indices:
        exposure_scores[index] = scores_by_face_id[int(face_ids[index])]
    guarded, suppressed_indices = suppress_fully_occluded_surface_predictions(
        predictions,
        exposure_scores,
    )
    diagnostics = [
        {
            "face_id": int(face_ids[index]),
            "exposure_score": float(exposure_scores[index]),
        }
        for index in suppressed_indices
    ]
    return guarded.tolist(), diagnostics


def postprocess_prediction_csv(step_path, feature_path, prediction_path, output_path=None):
    import pandas as pd

    feature_frame = pd.read_csv(feature_path)
    prediction_frame = pd.read_csv(prediction_path)
    required_prediction_columns = {"face_id", "pred_label", "pred_name"}
    missing = sorted(required_prediction_columns - set(prediction_frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    if "id" not in feature_frame.columns:
        raise ValueError("Feature CSV is missing the id column.")
    face_ids = prediction_frame["face_id"].astype(int).tolist()
    feature_face_ids = feature_frame["id"].astype(int).tolist()
    if face_ids != feature_face_ids:
        raise ValueError("Prediction and feature CSV face IDs are not aligned.")

    original_predictions = prediction_frame["pred_label"].astype(int).tolist()
    guarded_predictions, diagnostics = apply_exterior_visibility_surface_guard(
        step_path,
        original_predictions,
        face_ids,
        feature_frame,
    )
    if "prob_surface_feature" in prediction_frame.columns:
        guarded_predictions, density_suppressed, surface_density = (
            suppress_low_confidence_dense_surface_predictions(
                guarded_predictions,
                prediction_frame["prob_surface_feature"].to_numpy(),
            )
        )
        diagnostics.extend(
            {
                "action": "dense_surface_confidence",
                "face_id": int(face_ids[index]),
                "surface_density": surface_density,
                "surface_confidence": float(
                    prediction_frame.at[index, "prob_surface_feature"]
                ),
            }
            for index in density_suppressed
        )
    changed_indices = [
        index
        for index, (before, after) in enumerate(
            zip(original_predictions, guarded_predictions)
        )
        if before != after
    ]
    for index in changed_indices:
        prediction_frame.at[index, "pred_label"] = 0
        prediction_frame.at[index, "pred_name"] = "background"
        if "pred_confidence" in prediction_frame.columns and "prob_background" in prediction_frame.columns:
            prediction_frame.at[index, "pred_confidence"] = prediction_frame.at[
                index, "prob_background"
            ]

    destination = Path(output_path) if output_path else Path(prediction_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")
    prediction_frame.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Suppress fully occluded surface-feature predictions."
    )
    parser.add_argument("step_model", type=Path)
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    diagnostics = postprocess_prediction_csv(
        args.step_model,
        args.feature_csv,
        args.prediction_csv,
        args.output,
    )
    if diagnostics:
        for item in diagnostics:
            if item.get("action") == "complete_airbus_window":
                print(
                    "Airbus missed window -> surface_feature: "
                    + ",".join(f"F{face_id}" for face_id in item["face_ids"])
                )
            elif item.get("action") == "dense_surface_confidence":
                print(
                    f"F{item['face_id']}: dense surface density="
                    f"{item['surface_density']:.3f}, "
                    f"confidence={item['surface_confidence']:.3f} -> background"
                )
            else:
                print(
                    f"F{item['face_id']}: exposure={item['exposure_score']:.2f} "
                    "-> background"
                )
    else:
        print("No fully occluded surface-feature predictions were suppressed.")


if __name__ == "__main__":
    main()
