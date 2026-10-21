"""The deployed geometry screens: size, topology, exterior exposure and density."""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np

import hashlib
import json
from pathlib import Path

REQUIRED_FEATURE_COLUMNS = {
    "surfaceType",
    "neighborPlaneCount",
    "smoothEdgeCount",
}

REQUIRED_PREDICTION_COLUMNS = {"face_id", "pred_label"}

def apply_wing_shell_surface_guard(predictions, feature_frame):
    """Reject characteristic wing and symmetric shell patches predicted as surface features.

    The guard deliberately combines trimmed-face geometry and local topology.
    It is not a global area cutoff, so large decals with curved neighbors or a
    smooth attachment edge remain eligible as surface features.
    """
    required_columns = {
        "surfaceType",
        "innerWireCount",
        "numEdges",
        "relativeArea",
        "compactness",
        "neighborPlaneCount",
        "neighborCurvedCount",
        "convexEdgeCount",
        "smoothEdgeCount",
    }
    missing_columns = sorted(required_columns - set(feature_frame.columns))
    if missing_columns:
        raise ValueError(f"Wing-shell guard is missing feature columns: {missing_columns}")
    if len(feature_frame) != predictions.numel():
        raise ValueError(
            "Wing-shell guard row count does not match predictions: "
            f"features={len(feature_frame)}, predictions={predictions.numel()}"
        )

    four_edge_wing_patch = (
        (feature_frame["surfaceType"].astype(int).to_numpy() == 6)
        & (feature_frame["innerWireCount"].astype(int).to_numpy() == 0)
        & (feature_frame["numEdges"].astype(int).to_numpy() == 4)
        & (feature_frame["relativeArea"].astype(float).to_numpy() >= 0.0013)
        & (feature_frame["compactness"].astype(float).to_numpy() >= 1.8)
        & (feature_frame["neighborPlaneCount"].astype(int).to_numpy() == 4)
        & (feature_frame["neighborCurvedCount"].astype(int).to_numpy() == 0)
        & (feature_frame["convexEdgeCount"].astype(int).to_numpy() == 4)
        & (feature_frame["smoothEdgeCount"].astype(int).to_numpy() == 0)
    )
    five_edge_shell_patch = (
        (feature_frame["surfaceType"].astype(int).to_numpy() == 6)
        & (feature_frame["innerWireCount"].astype(int).to_numpy() == 0)
        & (feature_frame["numEdges"].astype(int).to_numpy() == 5)
        & (feature_frame["relativeArea"].astype(float).to_numpy() >= 0.0004)
        & (feature_frame["relativeArea"].astype(float).to_numpy() <= 0.001)
        & (feature_frame["compactness"].astype(float).to_numpy() >= 1.4)
        & (feature_frame["compactness"].astype(float).to_numpy() <= 1.6)
        & (feature_frame["neighborPlaneCount"].astype(int).to_numpy() >= 4)
        & (feature_frame["neighborCurvedCount"].astype(int).to_numpy() <= 1)
        & (feature_frame["convexEdgeCount"].astype(int).to_numpy() >= 4)
        & (feature_frame["smoothEdgeCount"].astype(int).to_numpy() <= 1)
    )
    candidate = four_edge_wing_patch | five_edge_shell_patch
    candidate = torch.as_tensor(candidate, dtype=torch.bool, device=predictions.device)
    result = predictions.clone()
    result[(result == 2) & candidate] = 0
    return result

def apply_rivet_size_guard(predictions, feature_frame, maximum_relative_area=5e-6):
    """Reject rivet predictions whose normalized face area is too large."""
    if maximum_relative_area <= 0.0:
        raise ValueError("maximum_relative_area must be positive.")
    if "relativeArea" not in feature_frame.columns:
        raise ValueError("Rivet size guard is missing the relativeArea column.")
    if len(feature_frame) != predictions.numel():
        raise ValueError(
            "Rivet size guard row count does not match predictions: "
            f"features={len(feature_frame)}, predictions={predictions.numel()}"
        )
    oversized = torch.as_tensor(
        feature_frame["relativeArea"].astype(float).to_numpy() > maximum_relative_area,
        dtype=torch.bool,
        device=predictions.device,
    )
    result = predictions.clone()
    result[(result == 1) & oversized] = 0
    return result

def suppress_planar_patch_rivet_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """Return guarded predictions and the face IDs changed to background."""
    missing_prediction = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if "face_id" in features.columns:
        feature_face_id = "face_id"
    elif "id" in features.columns:
        feature_face_id = "id"
    else:
        raise ValueError("Feature CSV requires either face_id or id for face-id alignment.")
    missing_feature = REQUIRED_FEATURE_COLUMNS - set(features.columns)
    if missing_prediction:
        raise ValueError(f"Prediction CSV is missing: {sorted(missing_prediction)}")
    if missing_feature:
        raise ValueError(f"Feature CSV is missing: {sorted(missing_feature)}")
    if predictions["face_id"].duplicated().any() or features[feature_face_id].duplicated().any():
        raise ValueError("face-id values must be unique in both CSV files.")

    feature_view = features.loc[:, [
        feature_face_id, "surfaceType", "neighborPlaneCount", "smoothEdgeCount"
    ]].rename(columns={feature_face_id: "face_id"})
    merged = predictions.merge(
        feature_view,
        on="face_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged[["surfaceType", "neighborPlaneCount", "smoothEdgeCount"]].isna().any().any():
        raise ValueError("Prediction and feature CSV face-id values do not align.")

    suppress = (
        (merged["pred_label"].astype(int) == 1)
        & (merged["surfaceType"].astype(int) == 0)
        & (merged["neighborPlaneCount"].astype(int) >= 4)
        & (merged["smoothEdgeCount"].astype(int) == 0)
    )
    result = predictions.copy()
    result.loc[suppress.to_numpy(), "pred_label"] = 0
    if "pred_name" in result.columns:
        result.loc[suppress.to_numpy(), "pred_name"] = "background"
    if "pred_confidence" in result.columns and "prob_background" in result.columns:
        result.loc[suppress.to_numpy(), "pred_confidence"] = result.loc[
            suppress.to_numpy(), "prob_background"
        ]
    return result, result.loc[suppress.to_numpy(), "face_id"].astype(int).tolist()

def apply_base_guards(predictions, features):
    """Apply the recognition pipeline's existing wing/shell and rivet-area guards."""
    import torch

    if predictions.face_id.tolist() != features.id.tolist():
        raise ValueError("Base-guard prediction and feature face IDs are not aligned")
    labels = torch.tensor(predictions.pred_label.to_numpy(), dtype=torch.long)
    labels = apply_wing_shell_surface_guard(labels, features)
    labels = apply_rivet_size_guard(labels, features)
    output = predictions.copy()
    changed = output.pred_label.to_numpy() != labels.numpy()
    output.loc[changed, "pred_label"] = 0
    output.loc[changed, "pred_name"] = "background"
    if "pred_confidence" in output and "prob_background" in output:
        output.loc[changed, "pred_confidence"] = output.loc[changed, "prob_background"]
    return output, int(changed.sum())

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

def _compute_surface_exposure_scores(step_path, candidate_face_ids):
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

def compute_surface_exposure_scores(step_path, candidate_face_ids, cache_dir=None):
    """Optionally reuse geometry-only scores across models on identical STEP bytes."""
    if cache_dir is None:
        return _compute_surface_exposure_scores(step_path, candidate_face_ids)
    import OCC

    step_hash = hashlib.sha256(Path(step_path).read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    version = str(getattr(OCC, "VERSION", "unknown"))
    key = hashlib.sha256(f"{step_hash}:{code_hash}:{version}".encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scores": {}}
    scores = {int(face_id): float(score) for face_id, score in cached["scores"].items()}
    if any(not 0.0 <= score <= 1.0 for score in scores.values()):
        raise ValueError("Invalid cached surface exposure score")
    missing = [int(face_id) for face_id in candidate_face_ids if int(face_id) not in scores]
    if missing:
        scores.update(_compute_surface_exposure_scores(step_path, missing))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"step_sha256": step_hash, "code_sha256": code_hash,
                                         "occ_version": version, "scores": scores}), encoding="utf-8")
        temporary.replace(path)
    return {int(face_id): scores[int(face_id)] for face_id in candidate_face_ids}

def apply_exterior_visibility_surface_guard(
    step_path,
    predictions,
    face_ids,
    feature_frame,
    exposure_cache_dir=None,
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
    scores_by_face_id = compute_surface_exposure_scores(step_path, candidate_face_ids, exposure_cache_dir)
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
