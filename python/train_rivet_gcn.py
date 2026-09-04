import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_max_pool, global_mean_pool


LEGACY_BASE_FEATURE_COLS = [
    "relativeArea",
    "compactness",
    "meanCurvature",
    "uvInsideFraction",
    "normalVariation",
    "curvatureVariation",
    "radius",
    "numWires",
    "innerWireCount",
    "minInnerWireLength",
    "maxInnerWireLength",
    "numEdges",
    "neighborAreaMean",
    "neighborAreaMax",
    "areaToNeighborMean",
    "areaToNeighborMax",
    "neighborPlaneCount",
    "neighborCylinderCount",
    "neighborCurvedCount",
    "convexEdgeCount",
    "concaveEdgeCount",
    "smoothEdgeCount",
    "convexEdgeRatio",
    "concaveEdgeRatio",
]

NORMALIZED_TRIMMED_AREA_COL = "normalizedTrimmedArea"
NORMALIZATION_SCALE_COL = "normalizationScale"
NORMALIZATION_CENTER_COLS = [
    "normalizationCenterX",
    "normalizationCenterY",
    "normalizationCenterZ",
]
NORMALIZED_GEOMETRY_FEATURE_COLS = [
    NORMALIZED_TRIMMED_AREA_COL,
    "normalizedPerimeter",
    "normalizedMeanCurvature",
    "normalizedRadius",
    "normalizedMinInnerWireLength",
    "normalizedMaxInnerWireLength",
    "normalizedNeighborAreaMean",
    "normalizedNeighborAreaMax",
]

INNER_LOOP_FEATURE_COLS = [
    "minInnerLoopBoundaryDihedralMax",
    "minInnerLoopBoundaryRightAngleDeviation",
    "hasValidInnerLoopBoundaryDihedral",
    "innerLoopAllDihedralBelowThreshold",
    "hasInnerLoopInteriorBfsDepthAtMost2",
    "hasSmallFlatInnerLoop",
    "hasSmallRightAngleInnerLoop",
]

ATTACHED_SURFACE_FEATURE_COLS = [
    "logNormalizedTrimmedArea",
    "logNormalizedPerimeter",
    "logAreaToNeighborMean",
    "logAreaToNeighborMax",
    "sameSurfaceNeighborRatio",
    "smoothSameSurfaceNeighborRatio",
    "smoothSameSurfaceBoundaryRatio",
    "smoothSameSurfaceAreaContrast",
    "hasLargerSmoothSameSurfaceNeighbor",
    "uniqueNeighborToEdgeRatio",
    "sharedBoundaryRatio",
    "logSmoothComponentNormalizedArea",
    "smoothComponentFaceCount",
    "smoothComponentFaceAreaRatio",
]

BASE_FEATURE_COLS = [
    "relativeArea",
    NORMALIZED_TRIMMED_AREA_COL,
    "normalizedPerimeter",
    "compactness",
    "normalizedMeanCurvature",
    "normalizedRadius",
    "has_radius",
    "numWires",
    "innerWireCount",
    "normalizedMinInnerWireLength",
    "normalizedMaxInnerWireLength",
    "minInnerLoopBoundaryDihedralMax",
    "minInnerLoopBoundaryRightAngleDeviation",
    "hasValidInnerLoopBoundaryDihedral",
    "innerLoopAllDihedralBelowThreshold",
    "hasInnerLoopInteriorBfsDepthAtMost2",
    "hasSmallFlatInnerLoop",
    "hasSmallRightAngleInnerLoop",
    "numEdges",
    "normalizedNeighborAreaMean",
    "normalizedNeighborAreaMax",
    "areaToNeighborMean",
    "areaToNeighborMax",
    "neighborPlaneCount",
    "neighborCylinderCount",
    "neighborCurvedCount",
    "convexEdgeCount",
    "concaveEdgeCount",
    "smoothEdgeCount",
    "convexEdgeRatio",
    "concaveEdgeRatio",
]

RELATIVE_NORMAL_FEATURE_COLS = [
    "normalNeighborDotMean",
    "normalNeighborDotMin",
    "normalNeighborDotMax",
]

MODEL_RELATIVE_POSITION_FEATURE_COLS = [
    "axisPositionAbs",
    "radialDistance",
]

ABSOLUTE_POSE_FEATURE_COLS = {
    "centerZ",
    "nx",
    "ny",
    "nz",
}

EDGE_ATTR_COLS = [
    "edge_types",
    "edge_area_ratios",
    "edge_neighbor_surface_types",
    "shared_edge_lengths",
    "edge_dihedral_means",
    "edge_dihedral_stds",
]

DEFAULT_CSV = Path("work/uv_train17_xian20_simpletest_train.csv")
DEFAULT_MODEL_PATH = Path("work/rivet_gnn_xian20_train_simpletest_50ep.pth")
DEFAULT_STATS_PATH = Path("work/rivet_gnn_xian20_train_simpletest_50ep_stats.npz")
DEFAULT_EVAL_PATH = Path("work/rivet_gnn_xian20_train_simpletest_50ep_eval.csv")

LABEL_NAMES = ["background", "rivet", "surface_feature"]


RADIUS_COL = "radius"
SURFACE_TYPE_COL = "surfaceType"
HAS_RADIUS_COL = "has_radius"


def split_tokens(value):
    if pd.isna(value):
        return []
    return str(value).split()


def label_names_from_stats(stats, num_classes):
    if "label_names" not in stats.files:
        raise ValueError("Stats file is missing the required three-class label_names metadata.")
    label_names = [str(value) for value in stats["label_names"].tolist()]
    if label_names != LABEL_NAMES or num_classes != len(LABEL_NAMES):
        raise ValueError(
            "Only the three-class background/rivet/surface_feature schema is supported: "
            f"classes={num_classes}, label_names={label_names}"
        )
    return label_names


def token_at(tokens, index, default="0"):
    return tokens[index] if index < len(tokens) else default


def infer_feature_columns(df):
    required_normalization_columns = {
        "area",
        "perimeter",
        "meanCurvature",
        "radius",
        "minInnerWireLength",
        "maxInnerWireLength",
        "neighborAreaMean",
        "neighborAreaMax",
        NORMALIZATION_SCALE_COL,
        *NORMALIZATION_CENTER_COLS,
    }
    missing_normalization_columns = required_normalization_columns.difference(df.columns)
    if missing_normalization_columns:
        raise ValueError(
            "CSV is missing model-normalization columns. Re-export it with the updated Detector: "
            f"{sorted(missing_normalization_columns)}"
        )
    surface_types = sorted({int(value) for value in df[SURFACE_TYPE_COL].fillna(0).astype(int).tolist()})
    feature_cols = list(BASE_FEATURE_COLS)
    feature_cols = [column for column in feature_cols if column not in INNER_LOOP_FEATURE_COLS]
    feature_cols.extend(ATTACHED_SURFACE_FEATURE_COLS)
    feature_cols.extend(RELATIVE_NORMAL_FEATURE_COLS)
    if {"centerX", "centerY", "centerZ"}.issubset(df.columns):
        feature_cols.extend(MODEL_RELATIVE_POSITION_FEATURE_COLS)
    feature_cols.extend(f"{SURFACE_TYPE_COL}_{surface_type}" for surface_type in surface_types)
    return feature_cols


def compute_attached_surface_feature_frame(df):
    """Derive local attachment and smooth-component geometry features.

    A decal is normally a small, separate face attached to a host.  In
    contrast, a fuselage shell can be represented by one or a few huge faces
    joined by smooth, tangent edges.  Build those tangent components here so
    the classifier and loss can distinguish the two cases without an absolute
    face-area cutoff.
    """
    local_df = df.reset_index(drop=True)
    face_ids = (
        local_df["id"].astype(int).tolist()
        if "id" in local_df.columns
        else list(range(1, len(local_df) + 1))
    )
    id_to_index = {face_id: index for index, face_id in enumerate(face_ids)}
    parent = list(range(len(local_df)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for node_idx, row in local_df.iterrows():
        neighbor_ids = split_tokens(row.get("neighbors"))
        edge_types = [int(value) for value in split_tokens(row.get("edge_types"))]
        neighbor_types = [
            int(value) for value in split_tokens(row.get("edge_neighbor_surface_types"))
        ]
        dihedral_means = [float(value) for value in split_tokens(row.get("edge_dihedral_means"))]
        surface_type = int(float(row.get(SURFACE_TYPE_COL, 0)))
        edge_count = min(len(neighbor_ids), len(edge_types), len(neighbor_types))
        for edge_index in range(edge_count):
            neighbor_idx = id_to_index.get(int(neighbor_ids[edge_index]))
            if neighbor_idx is None:
                continue
            if (
                edge_types[edge_index] == 0
                and neighbor_types[edge_index] == surface_type
                and abs(float(token_at(dihedral_means, edge_index))) <= 0.05
            ):
                union(node_idx, neighbor_idx)

    normalized_areas = (
        local_df["area"].astype(float).abs().to_numpy()
        * local_df[NORMALIZATION_SCALE_COL].astype(float).to_numpy() ** 2
    )
    component_areas = {}
    component_counts = {}
    for node_idx, area in enumerate(normalized_areas):
        root = find(node_idx)
        component_areas[root] = component_areas.get(root, 0.0) + float(area)
        component_counts[root] = component_counts.get(root, 0) + 1

    rows = []
    for node_idx, row in local_df.iterrows():
        edge_types = [int(value) for value in split_tokens(row.get("edge_types"))]
        neighbor_types = [
            int(value) for value in split_tokens(row.get("edge_neighbor_surface_types"))
        ]
        shared_lengths = [float(value) for value in split_tokens(row.get("shared_edge_lengths"))]
        area_ratios = [float(value) for value in split_tokens(row.get("edge_area_ratios"))]
        dihedral_means = [float(value) for value in split_tokens(row.get("edge_dihedral_means"))]
        edge_count = min(
            len(edge_types),
            len(neighbor_types),
            len(shared_lengths),
            len(area_ratios),
        )
        current_surface_type = int(float(row.get(SURFACE_TYPE_COL, 0)))
        same_surface = [
            edge_index
            for edge_index in range(edge_count)
            if neighbor_types[edge_index] == current_surface_type
        ]
        smooth_same_surface = [
            edge_index
            for edge_index in same_surface
            if edge_types[edge_index] == 0
            and abs(float(token_at(dihedral_means, edge_index))) <= 0.05
        ]
        perimeter = max(abs(float(row.get("perimeter", 0.0))), 1e-12)
        smooth_boundary_ratio = min(
            sum(abs(shared_lengths[index]) for index in smooth_same_surface) / perimeter,
            1.0,
        )
        smooth_area_ratios = [
            max(abs(area_ratios[index]), 1e-12)
            for index in smooth_same_surface
        ]
        min_smooth_area_ratio = min(smooth_area_ratios, default=1.0)
        normalization_scale = max(float(row.get(NORMALIZATION_SCALE_COL, 0.0)), 1e-12)
        normalized_area = abs(float(row.get("area", 0.0))) * normalization_scale ** 2
        normalized_perimeter = abs(float(row.get("perimeter", 0.0))) * normalization_scale
        unique_neighbor_count = len(set(split_tokens(row.get("neighbors"))))
        num_edges = max(int(float(row.get("numEdges", 0))), 1)
        shared_boundary_ratio = min(
            sum(abs(length) for length in shared_lengths) / perimeter,
            1.0,
        )
        component_root = find(node_idx)
        component_area = component_areas[component_root]
        component_count = component_counts[component_root]
        rows.append({
            "logNormalizedTrimmedArea": np.log(max(normalized_area, 1e-12)),
            "logNormalizedPerimeter": np.log(max(normalized_perimeter, 1e-12)),
            "logAreaToNeighborMean": np.log1p(max(float(row.get("areaToNeighborMean", 0.0)), 0.0)),
            "logAreaToNeighborMax": np.log1p(max(float(row.get("areaToNeighborMax", 0.0)), 0.0)),
            "sameSurfaceNeighborRatio": len(same_surface) / max(edge_count, 1),
            "smoothSameSurfaceNeighborRatio": len(smooth_same_surface) / max(edge_count, 1),
            "smoothSameSurfaceBoundaryRatio": smooth_boundary_ratio,
            "smoothSameSurfaceAreaContrast": float(
                np.clip(-np.log(min_smooth_area_ratio), -12.0, 12.0)
            ),
            "hasLargerSmoothSameSurfaceNeighbor": float(min_smooth_area_ratio < 1.0),
            "uniqueNeighborToEdgeRatio": unique_neighbor_count / num_edges,
            "sharedBoundaryRatio": shared_boundary_ratio,
            "logSmoothComponentNormalizedArea": np.log(max(component_area, 1e-12)),
            "smoothComponentFaceCount": float(component_count),
            "smoothComponentFaceAreaRatio": float(normalized_area / max(component_area, 1e-12)),
        })
    return pd.DataFrame(rows, index=df.index, columns=ATTACHED_SURFACE_FEATURE_COLS)


def build_feature_frame(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = infer_feature_columns(df)
    feature_cols = [str(col) for col in feature_cols]
    if (
        SURFACE_TYPE_COL in feature_cols or
        HAS_RADIUS_COL not in feature_cols or
        any(col in ABSOLUTE_POSE_FEATURE_COLS for col in feature_cols)
    ):
        raise ValueError(
            "Legacy feature schema detected in stats/model files. "
            "Please retrain to regenerate feature_cols without absolute pose features."
        )

    feature_df = pd.DataFrame(index=df.index)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    requested_attached_features = set(ATTACHED_SURFACE_FEATURE_COLS).intersection(feature_cols)
    if requested_attached_features:
        attached_features = compute_attached_surface_feature_frame(df)
        for col in requested_attached_features:
            feature_df[col] = attached_features[col]

    for col in LEGACY_BASE_FEATURE_COLS:
        if col not in feature_cols and not (col == RADIUS_COL and HAS_RADIUS_COL in feature_cols):
            continue
        if col == RADIUS_COL:
            radius_values = numeric_df[RADIUS_COL].fillna(0.0).astype(float)
            feature_df[RADIUS_COL] = radius_values
            feature_df[HAS_RADIUS_COL] = (radius_values.abs() > 1e-9).astype(float)
        else:
            feature_df[col] = numeric_df[col].fillna(0.0).astype(float)

    uses_normalized_geometry = NORMALIZED_TRIMMED_AREA_COL in feature_cols
    if uses_normalized_geometry:
        required_columns = {
            "area",
            "perimeter",
            "meanCurvature",
            "radius",
            "minInnerWireLength",
            "maxInnerWireLength",
            "neighborAreaMean",
            "neighborAreaMax",
            NORMALIZATION_SCALE_COL,
            *NORMALIZATION_CENTER_COLS,
        }
        missing_columns = required_columns.difference(numeric_df.columns)
        if missing_columns:
            raise ValueError(
                "Normalized feature schema requires re-exported geometry columns: "
                f"{sorted(missing_columns)}"
            )
        scale = numeric_df[NORMALIZATION_SCALE_COL].fillna(0.0).astype(float)
        if (scale <= 0.0).any():
            raise ValueError("normalizationScale must be positive for every face.")
        scale_squared = scale * scale
        radius_values = numeric_df[RADIUS_COL].fillna(0.0).astype(float)
        normalized_values = {
            NORMALIZED_TRIMMED_AREA_COL: numeric_df["area"].fillna(0.0).astype(float).abs() * scale_squared,
            "normalizedPerimeter": numeric_df["perimeter"].fillna(0.0).astype(float) * scale,
            "normalizedMeanCurvature": numeric_df["meanCurvature"].fillna(0.0).astype(float) / scale,
            "normalizedRadius": radius_values * scale,
            "normalizedMinInnerWireLength": numeric_df["minInnerWireLength"].fillna(0.0).astype(float) * scale,
            "normalizedMaxInnerWireLength": numeric_df["maxInnerWireLength"].fillna(0.0).astype(float) * scale,
            "normalizedNeighborAreaMean": numeric_df["neighborAreaMean"].fillna(0.0).astype(float).abs() * scale_squared,
            "normalizedNeighborAreaMax": numeric_df["neighborAreaMax"].fillna(0.0).astype(float).abs() * scale_squared,
        }
        for col, values in normalized_values.items():
            if col in feature_cols:
                feature_df[col] = values
        feature_df[HAS_RADIUS_COL] = (radius_values.abs() > 1e-9).astype(float)

    missing_inner_loop_columns = set(INNER_LOOP_FEATURE_COLS).intersection(feature_cols).difference(numeric_df.columns)
    if missing_inner_loop_columns:
        raise ValueError(
            "CSV is missing inner-loop features. Re-export it with the updated Detector: "
            f"{sorted(missing_inner_loop_columns)}"
        )
    for col in INNER_LOOP_FEATURE_COLS:
        if col in feature_cols:
            feature_df[col] = numeric_df[col].fillna(0.0).astype(float)

    for col in RELATIVE_NORMAL_FEATURE_COLS:
        feature_df[col] = numeric_df[col].fillna(0.0).astype(float)

    if {"centerX", "centerY", "centerZ"}.issubset(df.columns):
        positions = numeric_df[["centerX", "centerY", "centerZ"]].fillna(0.0).to_numpy(dtype=float)
        relative_position = np.zeros((len(df), 2), dtype=float)
        for _, indices in df.groupby("graph_id", sort=False).groups.items():
            index_array = np.fromiter(indices, dtype=int)
            coords = positions[index_array]
            if uses_normalized_geometry:
                centers = numeric_df.loc[index_array, NORMALIZATION_CENTER_COLS].to_numpy(dtype=float)
                scales = numeric_df.loc[index_array, NORMALIZATION_SCALE_COL].to_numpy(dtype=float)
                normalized_coords = (coords - centers) * scales[:, None]
                axis_index = int(np.argmax(normalized_coords.max(axis=0) - normalized_coords.min(axis=0)))
                axial = normalized_coords[:, axis_index]
                radial = np.sqrt(np.sum(np.delete(normalized_coords, axis_index, axis=1) ** 2, axis=1))
                relative_position[index_array, 0] = np.abs(axial)
                relative_position[index_array, 1] = radial
            else:
                centered = coords - coords.mean(axis=0, keepdims=True)
                axis_index = int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))
                axial = centered[:, axis_index]
                radial = np.sqrt(np.sum(np.delete(centered, axis_index, axis=1) ** 2, axis=1))
                position_scale = max(np.sqrt(np.sum(centered ** 2, axis=1)).max(), 1e-6)
                relative_position[index_array, 0] = np.abs(axial) / position_scale
                relative_position[index_array, 1] = radial / position_scale
        feature_df["axisPositionAbs"] = relative_position[:, 0]
        feature_df["radialDistance"] = relative_position[:, 1]

    surface_type_values = numeric_df[SURFACE_TYPE_COL].fillna(0).astype(int)
    for col in feature_cols:
        if not col.startswith(f"{SURFACE_TYPE_COL}_"):
            continue
        surface_type = int(col.split("_")[-1])
        feature_df[col] = (surface_type_values == surface_type).astype(float)

    missing_cols = [col for col in feature_cols if col not in feature_df.columns]
    for col in missing_cols:
        feature_df[col] = 0.0

    return feature_df[feature_cols].astype(float), feature_cols


def build_graph(group_df, feature_mean, feature_std, edge_mean=None, edge_std=None, feature_cols=None):
    local_df = group_df.reset_index(drop=True).copy()
    feature_frame, feature_cols = build_feature_frame(local_df, feature_cols)
    node_features = feature_frame.to_numpy()
    node_features = (node_features - feature_mean) / feature_std

    id_to_index = {int(row_id): idx for idx, row_id in enumerate(local_df["id"].tolist())}
    edge_index = []
    edge_attr = []
    uses_normalized_geometry = NORMALIZED_TRIMMED_AREA_COL in feature_cols

    for node_idx, row in local_df.iterrows():
        neighbor_ids = split_tokens(row["neighbors"])
        edge_attr_tokens = {
            col: split_tokens(row[col]) if col in local_df.columns else []
            for col in EDGE_ATTR_COLS
        }

        for edge_pos, neighbor_id_str in enumerate(neighbor_ids):
            neighbor_id = int(neighbor_id_str)
            if neighbor_id not in id_to_index:
                continue

            edge_index.append([node_idx, id_to_index[neighbor_id]])
            values = []
            for col in EDGE_ATTR_COLS:
                value = float(token_at(edge_attr_tokens[col], edge_pos))
                if uses_normalized_geometry and col == "shared_edge_lengths":
                    value *= float(row[NORMALIZATION_SCALE_COL])
                values.append(value)
            edge_attr.append(values)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        if edge_mean is not None and edge_std is not None:
            edge_mean_tensor = torch.tensor(edge_mean, dtype=torch.float)
            edge_std_tensor = torch.tensor(edge_std, dtype=torch.float)
            edge_attr = (edge_attr - edge_mean_tensor) / edge_std_tensor
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(EDGE_ATTR_COLS)), dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(local_df["label"].astype(int).to_numpy(), dtype=torch.long)
    attached_features = compute_attached_surface_feature_frame(local_df)
    normalized_area = torch.tensor(
        local_df["area"].astype(float).abs().to_numpy()
        * local_df[NORMALIZATION_SCALE_COL].astype(float).to_numpy() ** 2,
        dtype=torch.float,
    )
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        normalized_area=normalized_area,
        shared_boundary_ratio=torch.tensor(
            attached_features["sharedBoundaryRatio"].to_numpy(),
            dtype=torch.float,
        ),
        smooth_component_normalized_area=torch.tensor(
            np.exp(attached_features["logSmoothComponentNormalizedArea"].to_numpy()),
            dtype=torch.float,
        ),
        smooth_component_face_count=torch.tensor(
            attached_features["smoothComponentFaceCount"].to_numpy(),
            dtype=torch.float,
        ),
        smooth_component_face_area_ratio=torch.tensor(
            attached_features["smoothComponentFaceAreaRatio"].to_numpy(),
            dtype=torch.float,
        ),
        face_id=torch.tensor(local_df["id"].astype(int).to_numpy(), dtype=torch.long),
        target_mask=torch.ones(y.size(0), dtype=torch.bool),
        graph_id=int(local_df["graph_id"].iloc[0]),
        model_name=str(local_df["model_name"].iloc[0]),
    )


def build_window_cache(full_graph):
    adjacency = [[] for _ in range(full_graph.num_nodes)]
    outgoing_edges = [[] for _ in range(full_graph.num_nodes)]
    for edge_pos, (src, dst) in enumerate(full_graph.edge_index.t().tolist()):
        edge_attributes = full_graph.edge_attr[edge_pos].tolist()
        adjacency[src].append(dst)
        adjacency[dst].append(src)
        outgoing_edges[src].append((dst, edge_attributes))
    return adjacency, outgoing_edges


def khop_nodes(edge_index, center_idx, hop_count, num_nodes, adjacency=None):
    if adjacency is None:
        adjacency = [[] for _ in range(num_nodes)]
        for src, dst in edge_index.t().tolist():
            adjacency[src].append(dst)
            adjacency[dst].append(src)

    visited = {center_idx}
    frontier = {center_idx}
    for _ in range(hop_count):
        next_frontier = set()
        for node_idx in frontier:
            for neighbor_idx in adjacency[node_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    next_frontier.add(neighbor_idx)
        frontier = next_frontier
        if not frontier:
            break

    return sorted(visited)


def build_window_graph(full_graph, center_idx, hop_count, window_cache=None):
    if window_cache is None:
        window_cache = build_window_cache(full_graph)
    adjacency, outgoing_edges = window_cache
    node_ids = khop_nodes(
        full_graph.edge_index,
        center_idx,
        hop_count,
        full_graph.num_nodes,
        adjacency,
    )
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_ids)}

    local_edges = []
    local_edge_attrs = []
    for src in node_ids:
        for dst, edge_attributes in outgoing_edges[src]:
            if dst in old_to_new:
                local_edges.append([old_to_new[src], old_to_new[dst]])
                local_edge_attrs.append(edge_attributes)

    if local_edges:
        edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(local_edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, full_graph.edge_attr.size(1)), dtype=torch.float)

    target_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    target_mask[old_to_new[center_idx]] = True

    return Data(
        x=full_graph.x[node_ids],
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=full_graph.y[node_ids],
        normalized_area=full_graph.normalized_area[node_ids],
        shared_boundary_ratio=full_graph.shared_boundary_ratio[node_ids],
        smooth_component_normalized_area=full_graph.smooth_component_normalized_area[node_ids],
        smooth_component_face_count=full_graph.smooth_component_face_count[node_ids],
        smooth_component_face_area_ratio=full_graph.smooth_component_face_area_ratio[node_ids],
        face_id=full_graph.face_id[node_ids],
        target_mask=target_mask,
        graph_id=int(full_graph.graph_id),
        model_name=str(full_graph.model_name),
        center_face_id=int(full_graph.face_id[center_idx]),
    )


def sample_window_centers(
    full_graph,
    background_ratio,
    rng,
    large_background_fraction=0.0,
    min_background_centers=0,
    background_coverage_fraction=0.0,
):
    labels = full_graph.y.cpu().numpy()
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels == 0)

    if background_ratio is None or background_ratio <= 0 or len(positive) == 0:
        return np.arange(full_graph.num_nodes)

    if min_background_centers < 0:
        raise ValueError("min_background_centers must be non-negative.")
    if not 0.0 <= background_coverage_fraction <= 1.0:
        raise ValueError("background_coverage_fraction must be between 0 and 1.")
    coverage_quota = int(np.ceil(len(negative) * background_coverage_fraction))
    max_negative = min(
        len(negative),
        max(len(positive) * background_ratio, min_background_centers, coverage_quota),
    )
    if max_negative <= 0:
        centers = positive.astype(int)
        rng.shuffle(centers)
        return centers

    if not 0.0 <= large_background_fraction <= 1.0:
        raise ValueError("large_background_fraction must be between 0 and 1.")
    remaining_negative = negative
    random_count = max_negative
    large_negative = np.empty(0, dtype=int)
    if random_count > 0 and large_background_fraction > 0.0 and len(remaining_negative) > 0:
        large_count = min(
            random_count,
            max(1, int(np.floor(max_negative * large_background_fraction))),
        )
        normalized_areas = full_graph.normalized_area.cpu().numpy()
        top_quartile_count = max(1, int(np.ceil(len(remaining_negative) * 0.25)))
        top_quartile = remaining_negative[
            np.argsort(normalized_areas[remaining_negative])[-top_quartile_count:]
        ]
        large_negative = rng.choice(
            top_quartile,
            size=min(large_count, len(top_quartile)),
            replace=False,
        )

    remaining_after_large = np.setdiff1d(remaining_negative, large_negative, assume_unique=False)
    random_count -= len(large_negative)
    random_negative = (
        rng.choice(remaining_after_large, size=random_count, replace=False)
        if random_count > 0
        else np.empty(0, dtype=int)
    )
    centers = np.concatenate([positive, large_negative, random_negative]).astype(int)
    rng.shuffle(centers)
    return centers


def build_window_dataset(
    full_graphs,
    hop_count,
    background_ratio,
    seed,
    large_background_fraction=0.0,
    min_background_centers=0,
    background_coverage_fraction=0.0,
):
    rng = np.random.default_rng(seed)
    windows = []
    for full_graph in full_graphs:
        window_cache = build_window_cache(full_graph)
        for center_idx in sample_window_centers(
            full_graph,
            background_ratio,
            rng,
            large_background_fraction,
            min_background_centers,
            background_coverage_fraction,
        ):
            windows.append(build_window_graph(full_graph, int(center_idx), hop_count, window_cache))
    return windows


def build_full_balanced_dataset(full_graphs, background_ratio, seed):
    rng = np.random.default_rng(seed)
    balanced_graphs = []
    for full_graph in full_graphs:
        graph = full_graph.clone()
        selected_nodes = sample_window_centers(graph, background_ratio, rng)
        graph.target_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        graph.target_mask[torch.tensor(selected_nodes, dtype=torch.long)] = True
        balanced_graphs.append(graph)
    return balanced_graphs


def collect_edge_attr_matrix(df, feature_cols=None):
    uses_normalized_geometry = bool(feature_cols) and NORMALIZED_TRIMMED_AREA_COL in feature_cols
    rows = []
    for _, row in df.iterrows():
        neighbor_ids = split_tokens(row["neighbors"])
        edge_attr_tokens = {
            col: split_tokens(row[col]) if col in df.columns else []
            for col in EDGE_ATTR_COLS
        }
        for edge_pos in range(len(neighbor_ids)):
            values = []
            for col in EDGE_ATTR_COLS:
                value = float(token_at(edge_attr_tokens[col], edge_pos))
                if uses_normalized_geometry and col == "shared_edge_lengths":
                    value *= float(row[NORMALIZATION_SCALE_COL])
                values.append(value)
            rows.append(values)

    if not rows:
        return np.empty((0, len(EDGE_ATTR_COLS)), dtype=float)
    return np.asarray(rows, dtype=float)


def load_feature_stats(stats_path):
    stats_file = Path(stats_path)
    if not stats_file.exists():
        return None, None, None, None, None

    stats = np.load(stats_file, allow_pickle=True)
    feature_cols = stats["feature_cols"].tolist() if "feature_cols" in stats.files else None
    edge_mean = stats["edge_mean"] if "edge_mean" in stats.files else None
    edge_std = stats["edge_std"] if "edge_std" in stats.files else None
    return stats["mean"], stats["std"], edge_mean, edge_std, feature_cols


def load_cad_data(csv_path, stats_path):
    df = pd.read_csv(csv_path)
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export inference data.")

    feature_mean, feature_std, edge_mean, edge_std, feature_cols = load_feature_stats(stats_path)
    if feature_mean is None or feature_std is None:
        raise FileNotFoundError(
            f"Checkpoint normalization stats are required for inference: {stats_path}"
        )
    elif feature_cols is None:
        feature_cols = infer_feature_columns(df)
    if edge_mean is None or edge_std is None:
        edge_matrix = collect_edge_attr_matrix(df, feature_cols)
        edge_mean = edge_matrix.mean(axis=0) if len(edge_matrix) else np.zeros(len(EDGE_ATTR_COLS))
        edge_std = edge_matrix.std(axis=0) + 1e-6 if len(edge_matrix) else np.ones(len(EDGE_ATTR_COLS))

    first_graph_id = df["graph_id"].iloc[0]
    group = df[df["graph_id"] == first_graph_id]
    return build_graph(group, feature_mean, feature_std, edge_mean, edge_std, feature_cols)


def build_graphs_from_dataframe(df, feature_mean, feature_std, edge_mean=None, edge_std=None, feature_cols=None):
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export training data.")

    graphs = []
    for _, group in df.groupby("graph_id", sort=True):
        graphs.append(build_graph(group, feature_mean, feature_std, edge_mean, edge_std, feature_cols))
    return graphs


def prepare_graph_samples(
    full_graphs,
    training_mode="full",
    window_hop=2,
    background_ratio=5,
    seed=42,
    large_background_fraction=0.0,
    min_background_centers=0,
    background_coverage_fraction=0.0,
):
    if training_mode == "window":
        return build_window_dataset(
            full_graphs,
            window_hop,
            background_ratio,
            seed,
            large_background_fraction,
            min_background_centers,
            background_coverage_fraction,
        )
    if training_mode == "full-balanced":
        return build_full_balanced_dataset(full_graphs, background_ratio, seed)
    return full_graphs


def load_graph_dataset(csv_path, training_mode="full", window_hop=2, background_ratio=5, seed=42):
    df = pd.read_csv(csv_path)
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export training data.")

    feature_frame, feature_cols = build_feature_frame(df)
    feature_matrix = feature_frame.to_numpy()
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-6
    edge_matrix = collect_edge_attr_matrix(df, feature_cols)
    edge_mean = edge_matrix.mean(axis=0) if len(edge_matrix) else np.zeros(len(EDGE_ATTR_COLS))
    edge_std = edge_matrix.std(axis=0) + 1e-6 if len(edge_matrix) else np.ones(len(EDGE_ATTR_COLS))

    full_graphs = build_graphs_from_dataframe(df, feature_mean, feature_std, edge_mean, edge_std, feature_cols)
    graphs = prepare_graph_samples(full_graphs, training_mode, window_hop, background_ratio, seed)

    return graphs, feature_mean, feature_std, edge_mean, edge_std, feature_cols


def load_explicit_train_val_datasets(
    train_csv_path,
    val_csv_path,
    training_mode="full",
    window_hop=2,
    background_ratio=5,
    seed=42,
    large_background_fraction=0.0,
    min_background_centers=0,
    background_coverage_fraction=0.0,
    label_names=LABEL_NAMES,
):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    for name, df in [("train", train_df), ("val", val_df)]:
        if "graph_id" not in df.columns or "model_name" not in df.columns:
            raise ValueError(f"{name} CSV is missing graph_id/model_name columns. Please re-export training data.")
        if "label" not in df.columns:
            raise ValueError(f"{name} CSV is missing the label column.")

    for name, df in [("train", train_df), ("val", val_df)]:
        labels = set(df["label"].astype(int).unique())
        if not labels.issubset(set(range(len(label_names)))):
            raise ValueError(
                f"{name} CSV labels do not match {label_names}; "
                f"found labels={sorted(labels)}"
            )

    train_feature_frame, feature_cols = build_feature_frame(train_df)
    feature_matrix = train_feature_frame.to_numpy()
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-6
    edge_matrix = collect_edge_attr_matrix(train_df, feature_cols)
    edge_mean = edge_matrix.mean(axis=0) if len(edge_matrix) else np.zeros(len(EDGE_ATTR_COLS))
    edge_std = edge_matrix.std(axis=0) + 1e-6 if len(edge_matrix) else np.ones(len(EDGE_ATTR_COLS))

    train_full_graphs = build_graphs_from_dataframe(train_df, feature_mean, feature_std, edge_mean, edge_std, feature_cols)
    val_full_graphs = build_graphs_from_dataframe(val_df, feature_mean, feature_std, edge_mean, edge_std, feature_cols)
    train_graphs = prepare_graph_samples(
        train_full_graphs,
        training_mode,
        window_hop,
        background_ratio,
        seed,
        large_background_fraction,
        min_background_centers,
        background_coverage_fraction,
    )
    val_training_mode = "full" if training_mode == "full-balanced" else training_mode
    val_background_ratio = 0 if training_mode == "window" else background_ratio
    val_graphs = prepare_graph_samples(
        val_full_graphs,
        val_training_mode,
        window_hop,
        val_background_ratio,
        seed + 1,
    )
    return (
        train_graphs,
        val_graphs,
        feature_mean,
        feature_std,
        edge_mean,
        edge_std,
        feature_cols,
        train_full_graphs,
    )


class RivetGNN(torch.nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_classes=2,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be at least 3")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        # AAGNet-style separate encoders for face and adjacency attributes.
        # Keeping the output widths unchanged preserves the NNConv interface.
        self.node_input_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_features, node_features),
            torch.nn.LayerNorm(node_features),
            torch.nn.ReLU(),
        )
        self.edge_input_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_features, edge_features),
            torch.nn.LayerNorm(edge_features),
            torch.nn.ReLU(),
        )

        def create_nn(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Linear(edge_features, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, in_f * out_f),
            )

        if num_layers == 3:
            self.conv1 = NNConv(node_features, hidden_dim, create_nn(node_features, hidden_dim))
            self.bn1 = torch.nn.LayerNorm(hidden_dim)
            self.conv2 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
            self.bn2 = torch.nn.LayerNorm(hidden_dim)
            self.conv3 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
            self.bn3 = torch.nn.LayerNorm(hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)
            return
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer_index in range(num_layers):
            in_features = node_features if layer_index == 0 else hidden_dim
            self.convs.append(NNConv(in_features, hidden_dim, create_nn(in_features, hidden_dim)))
            self.batch_norms.append(torch.nn.LayerNorm(hidden_dim))
        self.classifier = torch.nn.Linear(hidden_dim * 3, num_classes)

    def encode(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_input_encoder(x)
        edge_attr = self.edge_input_encoder(edge_attr)

        if self.num_layers == 3:
            x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
            x = self.dropout(x)
            identity = x
            x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
            x = self.dropout(x)
            x = x + identity
            x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
            x = self.dropout(x)
            return x
        for layer_index, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            updated = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            updated = self.dropout(updated)
            x = updated if layer_index == 0 else updated + x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_features = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return torch.cat([x, global_features[batch]], dim=1)

    def forward(self, data):
        return F.log_softmax(self.classifier(self.encode(data)), dim=1)


class DualHeadRivetGNN(torch.nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        # These are deliberately separate encoders.  The rivet and surface
        # objectives must not alter the same message-passing representation.
        self.rivet_encoder = RivetGNN(
            node_features, edge_features, hidden_dim, 3, num_layers, dropout
        )
        self.surface_encoder = RivetGNN(
            node_features,
            edge_features,
            hidden_dim,
            3,
            num_layers,
            dropout,
        )
        classifier_input = hidden_dim if num_layers == 3 else hidden_dim * 3
        adapter_dim = max(hidden_dim, classifier_input // 2)
        self.rivet_adapter = torch.nn.Sequential(
            torch.nn.Linear(classifier_input, adapter_dim),
            torch.nn.LayerNorm(adapter_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.surface_adapter = torch.nn.Sequential(
            torch.nn.Linear(classifier_input, adapter_dim),
            torch.nn.LayerNorm(adapter_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.rivet_classifier = torch.nn.Linear(adapter_dim, 1)
        self.surface_classifier = torch.nn.Linear(adapter_dim, 1)

    def forward(self, data):
        rivet_encoded = self.rivet_encoder.encode(data)
        surface_encoded = self.surface_encoder.encode(data)
        return torch.cat(
            [
                self.rivet_classifier(self.rivet_adapter(rivet_encoded)),
                self.surface_classifier(self.surface_adapter(surface_encoded)),
            ],
            dim=1,
        )


def dual_head_probabilities(logits):
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Dual-head logits must have shape [N, 2], got {tuple(logits.shape)}")
    return torch.sigmoid(logits)


def dual_head_decision(logits, rivet_threshold=0.5, surface_threshold=0.5):
    if not 0.0 < rivet_threshold < 1.0:
        raise ValueError("rivet_threshold must be between 0 and 1.")
    if not 0.0 < surface_threshold < 1.0:
        raise ValueError("surface_threshold must be between 0 and 1.")
    probabilities = dual_head_probabilities(logits)
    rivet_score = probabilities[:, 0]
    surface_score = probabilities[:, 1]
    predictions = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    # Rivet precision is the hard safety constraint.  Once a face clears the
    # rivet threshold, surface evidence must not overwrite that decision.
    rivet_selected = rivet_score >= rivet_threshold
    surface_selected = (~rivet_selected) & (surface_score >= surface_threshold)
    predictions[rivet_selected] = 1
    predictions[surface_selected] = 2
    return predictions, probabilities


def compute_class_weights(graphs, num_classes=2, max_weight=5.0):
    labels = []
    for graph in graphs:
        target_mask = getattr(graph, "target_mask", torch.ones(graph.y.size(0), dtype=torch.bool))
        labels.append(graph.y[target_mask].cpu())

    if not labels:
        return torch.ones(num_classes, dtype=torch.float)

    all_labels = torch.cat(labels)
    counts = torch.bincount(all_labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    inverse_sqrt_weights = torch.sqrt(counts.sum() / (len(counts) * counts))
    capped_weights = torch.clamp(inverse_sqrt_weights, max=max_weight)
    return capped_weights / capped_weights.mean()


def compute_metrics_from_confusion(confusion):
    metrics = []
    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id].item())
        fp = int(confusion[:, class_id].sum().item() - tp)
        fn = int(confusion[class_id, :].sum().item() - tp)
        support = int(confusion[class_id, :].sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        metrics.append({
            "class_id": class_id,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        })
    return metrics


def summarize_metrics(metrics):
    if not metrics:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "miou": 0.0,
            "foreground_macro_f1": 0.0,
            "class_2_f1": 0.0,
            "rivet_fp": 0,
            "surface_fp": 0,
            "background_fp": 0,
        }
    foreground_metrics = metrics[1:]
    return {
        "macro_precision": float(np.mean([item["precision"] for item in metrics])),
        "macro_recall": float(np.mean([item["recall"] for item in metrics])),
        "macro_f1": float(np.mean([item["f1"] for item in metrics])),
        "miou": float(np.mean([item["iou"] for item in metrics])),
        "foreground_macro_f1": float(np.mean([item["f1"] for item in foreground_metrics]))
        if foreground_metrics
        else 0.0,
        "class_2_f1": float(metrics[2]["f1"]) if len(metrics) > 2 else 0.0,
        "rivet_fp": int(metrics[1]["fp"]) if len(metrics) > 1 else 0,
        "surface_fp": int(metrics[2]["fp"]) if len(metrics) > 2 else 0,
        # Here "background FP" means background faces predicted as any foreground class.
        "background_fp": int(metrics[0]["fn"]),
    }


def focal_nll_loss(
    log_probabilities,
    targets,
    class_weights=None,
    gamma=2.0,
    sample_weights=None,
):
    target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    weights = class_weights[targets] if class_weights is not None else 1.0
    if sample_weights is not None:
        weights = weights * sample_weights
    return ((1.0 - target_log_probabilities.exp()).pow(gamma) * -target_log_probabilities * weights).mean()


def compute_dual_head_positive_weights(
    graphs,
    max_weight=5.0,
    rivet_scale=1.0,
    surface_scale=1.0,
):
    if max_weight <= 0.0 or rivet_scale <= 0.0 or surface_scale <= 0.0:
        raise ValueError("Dual-head positive-weight parameters must be positive.")
    labels = []
    for graph in graphs:
        target_mask = getattr(graph, "target_mask", torch.ones(graph.y.size(0), dtype=torch.bool))
        labels.append(graph.y[target_mask].cpu())
    all_labels = torch.cat(labels)
    weights = []
    for class_id, scale in ((1, rivet_scale), (2, surface_scale)):
        positives = max(int((all_labels == class_id).sum()), 1)
        negatives = max(int((all_labels != class_id).sum()), 1)
        weights.append(min(np.sqrt(negatives / positives), max_weight) * scale)
    return torch.tensor(weights, dtype=torch.float)


def dual_head_focal_loss(
    logits,
    labels,
    positive_weights,
    gamma=2.0,
    sample_weights=None,
    head_sample_weights=None,
    negative_weights=None,
):
    targets = torch.stack([(labels == 1), (labels == 2)], dim=1).to(logits.dtype)
    base_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weights,
    )
    if negative_weights is not None:
        if negative_weights.shape != (2,):
            raise ValueError("negative_weights must have shape [2].")
        base_loss = base_loss * torch.where(
            targets > 0.5,
            torch.ones_like(base_loss),
            negative_weights.view(1, 2),
        )
    probabilities = torch.sigmoid(logits)
    target_probabilities = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
    loss = base_loss * (1.0 - target_probabilities).pow(gamma)
    if sample_weights is not None:
        loss = loss * sample_weights.unsqueeze(1)
    if head_sample_weights is not None:
        if head_sample_weights.shape != loss.shape:
            raise ValueError(
                f"head_sample_weights must match loss shape {tuple(loss.shape)}, "
                f"got {tuple(head_sample_weights.shape)}"
            )
        loss = loss * head_sample_weights
    return loss.mean()


def apply_large_background_surface_head_weight(
    head_weights,
    targets,
    normalized_areas,
    area_threshold,
    loss_weight,
):
    if loss_weight <= 0.0:
        raise ValueError("large_background_loss_weight must be positive.")
    if head_weights.shape != (targets.numel(), 2):
        raise ValueError("Dual-head sample weights must have shape [N, 2].")
    result = head_weights.clone()
    large_background = (targets == 0) & (normalized_areas >= area_threshold)
    result[large_background, 1] *= loss_weight
    return result


def apply_host_background_surface_head_weight(
    head_weights,
    targets,
    normalized_areas,
    shared_boundary_ratios,
    minimum_area,
    maximum_shared_boundary_ratio,
    loss_weight,
):
    if loss_weight <= 0.0:
        raise ValueError("host_background_loss_weight must be positive.")
    if minimum_area < 0.0:
        raise ValueError("host_background_min_area must be non-negative.")
    if not 0.0 <= maximum_shared_boundary_ratio <= 1.0:
        raise ValueError("host_background_max_shared_boundary_ratio must be between 0 and 1.")
    if head_weights.shape != (targets.numel(), 2):
        raise ValueError("Dual-head sample weights must have shape [N, 2].")
    result = head_weights.clone()
    host_background = (
        (targets == 0)
        & (normalized_areas >= minimum_area)
        & (shared_boundary_ratios <= maximum_shared_boundary_ratio)
    )
    result[host_background, 1] *= loss_weight
    return result


def apply_smooth_component_background_surface_head_weight(
    head_weights,
    targets,
    component_normalized_areas,
    component_face_counts,
    minimum_component_area,
    loss_weight,
):
    """Prioritize background shells represented by large tangent components."""
    if loss_weight <= 0.0:
        raise ValueError("smooth_component_background_loss_weight must be positive.")
    if minimum_component_area < 0.0:
        raise ValueError("smooth_component_background_min_area must be non-negative.")
    if head_weights.shape != (targets.numel(), 2):
        raise ValueError("Dual-head sample weights must have shape [N, 2].")
    result = head_weights.clone()
    shell_background = (
        (targets == 0)
        & (component_normalized_areas >= minimum_component_area)
        & (component_face_counts >= 2)
    )
    result[shell_background, 1] *= loss_weight
    return result


def apply_smooth_shell_surface_guard(
    predictions,
    component_normalized_areas,
    component_face_counts,
    component_face_area_ratios,
    minimum_component_area=0.25,
    minimum_face_area_ratio=0.4,
    dense_component_min_face_count=100,
    dense_component_min_face_area_ratio=0.03,
):
    """Reject a surface prediction only for a dominant face of a large shell.

    This is a topology-aware safeguard, not a global area threshold: a large
    decal remains eligible unless it is itself a dominant member of a smooth,
    same-surface tangent shell component, or an unusually large member of a
    dense shell component.
    """
    result = predictions.clone()
    large_component = component_normalized_areas >= minimum_component_area
    dominant_two_face_shell = (
        (component_face_counts >= 2)
        & (component_face_area_ratios >= minimum_face_area_ratio)
    )
    dominant_dense_shell = (
        (component_face_counts >= dense_component_min_face_count)
        & (component_face_area_ratios >= dense_component_min_face_area_ratio)
    )
    shell_face = (result == 2) & large_component & (
        dominant_two_face_shell | dominant_dense_shell
    )
    result[shell_face] = 0
    return result


def background_area_threshold(full_graphs, quantile):
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("large_background_loss_quantile must be between 0 and 1.")
    background_areas = [
        graph.normalized_area[graph.y == 0].detach().cpu()
        for graph in full_graphs
        if bool((graph.y == 0).any())
    ]
    if not background_areas:
        raise ValueError("Training data contains no background faces.")
    return float(torch.quantile(torch.cat(background_areas), quantile).item())


def large_background_sample_weights(
    targets,
    normalized_areas,
    area_threshold,
    loss_weight,
):
    if loss_weight <= 0.0:
        raise ValueError("large_background_loss_weight must be positive.")
    weights = torch.ones_like(normalized_areas, dtype=torch.float)
    large_background = (targets == 0) & (normalized_areas >= area_threshold)
    weights[large_background] = loss_weight
    return weights


def evaluate(
    model,
    loader,
    device,
    num_classes,
    class_weights=None,
    dual_head_positive_weights=None,
    dual_head_negative_weights=None,
    focal_gamma=2.0,
    model_architecture="three-class",
    rivet_threshold=0.5,
    surface_threshold=0.5,
    smooth_shell_guard=False,
    smooth_shell_guard_min_component_area=0.25,
    smooth_shell_guard_min_face_area_ratio=0.4,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            target_mask = getattr(batch, "target_mask", torch.ones(batch.y.size(0), dtype=torch.bool, device=batch.y.device))
            if model_architecture == "dual-head":
                loss = dual_head_focal_loss(
                    output[target_mask],
                    batch.y[target_mask],
                    dual_head_positive_weights,
                    focal_gamma,
                    negative_weights=dual_head_negative_weights,
                )
                pred, _ = dual_head_decision(
                    output,
                    rivet_threshold=rivet_threshold,
                    surface_threshold=surface_threshold,
                )
                if smooth_shell_guard:
                    pred = apply_smooth_shell_surface_guard(
                        pred,
                        batch.smooth_component_normalized_area,
                        batch.smooth_component_face_count,
                        batch.smooth_component_face_area_ratio,
                        smooth_shell_guard_min_component_area,
                        smooth_shell_guard_min_face_area_ratio,
                    )
            else:
                loss = focal_nll_loss(
                    output[target_mask],
                    batch.y[target_mask],
                    class_weights,
                    focal_gamma,
                )
                pred = output.argmax(dim=1)
            target_count = int(target_mask.sum().item())
            total_loss += loss.item() * target_count
            total_correct += int((pred[target_mask] == batch.y[target_mask]).sum().item())
            total_nodes += target_count
            for true_label, pred_label in zip(batch.y[target_mask].cpu(), pred[target_mask].cpu()):
                confusion[int(true_label), int(pred_label)] += 1

    if total_nodes == 0:
        metrics = compute_metrics_from_confusion(confusion)
        return 0.0, 0.0, metrics, summarize_metrics(metrics)

    metrics = compute_metrics_from_confusion(confusion)
    return total_loss / total_nodes, total_correct / total_nodes, metrics, summarize_metrics(metrics)


def write_eval_csv(output_path, metrics, summary, val_loss, val_acc, label_names=None):
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "class_id",
            "class_name",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "iou",
            "val_loss",
            "val_acc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "miou",
            "foreground_macro_f1",
            "class_2_f1",
            "rivet_fp",
            "surface_fp",
            "background_fp",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                **item,
                "class_name": label_names[item["class_id"]]
                if label_names and item["class_id"] < len(label_names)
                else f"class_{item['class_id']}",
                "val_loss": val_loss,
                "val_acc": val_acc,
                **summary,
            })


def print_metric_table(metrics, summary, split_name="Validation"):
    print(f"{split_name} class metrics:")
    print("class | support | precision | recall | f1 | iou")
    for item in metrics:
        print(
            f"{item['class_id']:>5} | "
            f"{item['support']:>7} | "
            f"{item['precision']:.4f} | "
            f"{item['recall']:.4f} | "
            f"{item['f1']:.4f} | "
            f"{item['iou']:.4f}"
        )
    print(
        "macro | "
        f"precision={summary['macro_precision']:.4f} | "
        f"recall={summary['macro_recall']:.4f} | "
        f"f1={summary['macro_f1']:.4f} | "
        f"mIoU={summary['miou']:.4f} | "
        f"foreground_f1={summary['foreground_macro_f1']:.4f} | "
        f"surface_feature_f1={summary['class_2_f1']:.4f} | "
        f"background_fp={summary['background_fp']}"
    )


def derive_surface_output_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}_best_surface_feature_f1{path.suffix}")


def save_checkpoint_bundle(
    model,
    model_out,
    stats_out,
    eval_out,
    feature_cols,
    feature_mean,
    feature_std,
    edge_mean,
    edge_std,
    num_layers,
    metrics,
    summary,
    val_loss,
    val_acc,
    label_names=LABEL_NAMES,
    classification_schema="three_class_surface_feature_v1",
    hidden_dim=None,
    training_mode=None,
    window_hop=None,
    training_seed=None,
    training_epochs=None,
    resample_window_centers=None,
    min_background_centers=None,
    background_coverage_fraction=None,
    large_background_loss_weight=None,
    large_background_loss_quantile=None,
    large_background_area_threshold=None,
    host_background_loss_weight=None,
    host_background_min_area=None,
    host_background_max_shared_boundary_ratio=None,
    model_architecture=None,
    rivet_threshold=None,
    surface_threshold=None,
):
    for output_path in (model_out, stats_out, eval_out):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    checkpoint_metadata = {
        "feature_cols": np.array(feature_cols, dtype=object),
        "edge_attr_cols": np.array(EDGE_ATTR_COLS, dtype=object),
        "mean": feature_mean,
        "std": feature_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "num_layers": np.array(num_layers),
        "feature_schema_version": np.array(
            3 if set(ATTACHED_SURFACE_FEATURE_COLS).intersection(feature_cols)
            else 2 if NORMALIZED_TRIMMED_AREA_COL in feature_cols
            else 1
        ),
        "label_names": np.array(label_names, dtype=object),
        "classification_schema": np.array(classification_schema),
    }
    optional_metadata = {
        "hidden_dim": hidden_dim,
        "training_mode": training_mode,
        "inference_mode": (
            None if training_mode is None
            else "window" if training_mode == "window"
            else "full"
        ),
        "window_hop": window_hop,
        "training_seed": training_seed,
        "training_epochs": training_epochs,
        "resample_window_centers": resample_window_centers,
        "min_background_centers": min_background_centers,
        "background_coverage_fraction": background_coverage_fraction,
        "large_background_loss_weight": large_background_loss_weight,
        "large_background_loss_quantile": large_background_loss_quantile,
        "large_background_area_threshold": large_background_area_threshold,
        "host_background_loss_weight": host_background_loss_weight,
        "host_background_min_area": host_background_min_area,
        "host_background_max_shared_boundary_ratio": host_background_max_shared_boundary_ratio,
        "model_architecture": model_architecture,
        "independent_specialist_encoders": (
            model_architecture == "dual-head"
            if model_architecture is not None
            else None
        ),
        "aagnet_style_input_encoders": True,
        "rivet_threshold": rivet_threshold,
        "surface_threshold": surface_threshold,
    }
    checkpoint_metadata.update({
        key: np.array(value)
        for key, value in optional_metadata.items()
        if value is not None
    })
    np.savez(stats_out, **checkpoint_metadata)
    write_eval_csv(
        eval_out,
        metrics,
        summary,
        val_loss,
        val_acc,
        label_names=label_names,
    )


def train(args):
    if args.val_csv is None and args.test_csv is None:
        raise ValueError("Provide --val-csv for validation training, or --test-csv for one final test-only evaluation.")
    if args.val_csv is not None and args.test_csv is not None:
        raise ValueError("Use either --val-csv or --test-csv, never both in one run.")
    if args.validation_interval <= 0:
        raise ValueError("validation_interval must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if args.rivet_negative_weight <= 0.0 or args.surface_negative_weight <= 0.0:
        raise ValueError("Dual-head negative weights must be positive.")
    if not 0.0 <= args.surface_recall_floor <= 1.0:
        raise ValueError("surface_recall_floor must be between 0 and 1.")
    if args.large_background_loss_weight <= 0.0:
        raise ValueError("large_background_loss_weight must be positive.")
    if args.host_background_loss_weight <= 0.0:
        raise ValueError("host_background_loss_weight must be positive.")
    if args.host_background_min_area < 0.0:
        raise ValueError("host_background_min_area must be non-negative.")
    if not 0.0 <= args.host_background_max_shared_boundary_ratio <= 1.0:
        raise ValueError("host_background_max_shared_boundary_ratio must be between 0 and 1.")
    if args.smooth_component_background_loss_weight <= 0.0:
        raise ValueError("smooth_component_background_loss_weight must be positive.")
    if args.smooth_component_background_min_area < 0.0:
        raise ValueError("smooth_component_background_min_area must be non-negative.")
    if args.rivet_positive_weight_scale <= 0.0:
        raise ValueError("rivet_positive_weight_scale must be positive.")
    if not 0.0 < args.rivet_threshold < 1.0:
        raise ValueError("rivet_threshold must be between 0 and 1.")
    if not 0.0 < args.surface_threshold < 1.0:
        raise ValueError("surface_threshold must be between 0 and 1.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    label_names = LABEL_NAMES
    test_only = args.test_csv is not None
    (
        train_graphs,
        val_graphs,
        feature_mean,
        feature_std,
        edge_mean,
        edge_std,
        feature_cols,
        train_full_graphs,
    ) = load_explicit_train_val_datasets(
        args.csv,
        args.test_csv if test_only else args.val_csv,
        training_mode=args.training_mode,
        window_hop=args.window_hop,
        background_ratio=args.background_ratio,
        seed=args.seed,
        large_background_fraction=args.large_background_fraction,
        min_background_centers=args.min_background_centers,
        background_coverage_fraction=args.background_coverage_fraction,
        label_names=label_names,
    )
    graphs = train_graphs + val_graphs
    split_description = (
        f"explicit train={args.csv}, final_test={args.test_csv}"
        if test_only
        else f"explicit train={args.csv}, val={args.val_csv}"
    )

    if not train_graphs or not val_graphs:
        raise ValueError("Training and validation datasets must both contain at least one graph/sample.")

    num_classes = max(int(graph.y.max().item()) for graph in graphs) + 1
    if num_classes != len(label_names):
        raise ValueError(
            "Training labels do not match the selected schema: "
            f"classes={num_classes}, label_names={label_names}"
        )

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {
        "node_features": train_graphs[0].num_node_features,
        "edge_features": train_graphs[0].edge_attr.size(1),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = (
        DualHeadRivetGNN(**model_kwargs)
        if args.model_architecture == "dual-head"
        else RivetGNN(**model_kwargs, num_classes=num_classes)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    class_weights = compute_class_weights(
        train_graphs,
        num_classes=num_classes,
        max_weight=args.max_class_weight,
    ).to(device)
    if args.surface_feature_weight_scale <= 0.0:
        raise ValueError("surface_feature_weight_scale must be positive.")
    class_weights[2] *= args.surface_feature_weight_scale
    dual_head_positive_weights = compute_dual_head_positive_weights(
        train_graphs,
        max_weight=args.max_class_weight,
        rivet_scale=args.rivet_positive_weight_scale,
        surface_scale=args.surface_feature_weight_scale,
    ).to(device)
    dual_head_negative_weights = torch.tensor(
        [args.rivet_negative_weight, args.surface_negative_weight],
        dtype=torch.float,
        device=device,
    )
    large_background_threshold = background_area_threshold(
        train_full_graphs,
        args.large_background_loss_quantile,
    )

    print(f"Loaded {len(graphs)} graph samples ({split_description})")
    print(f"Classes: {num_classes}")
    print(f"Label names: {label_names}")
    print(f"Training mode: {args.training_mode}")
    print(f"Model architecture: {args.model_architecture}")
    print("Attached-surface features: enabled")
    print(f"Optimizer: AdamW | dropout={args.dropout:.3f} | weight_decay={args.weight_decay:.6f}")
    if test_only:
        print(f"Final-test-only mode: fixed {args.epochs} epochs; test metrics are computed once after training.")
    else:
        print(
            f"Checkpoint selection: precision-first (rivet FP, surface FP, "
            f"{label_names[2]} F1, mIoU) and best eligible {label_names[2]} F1 "
            f"(validation every {args.validation_interval} epochs)"
        )
    if args.training_mode == "window":
        print(f"Window hop: {args.window_hop}, background ratio: {args.background_ratio}")
        print(f"Resample window centers each epoch: {not args.fixed_window_centers}")
        print(
            "Large-background sampling fraction: "
            f"{args.large_background_fraction:g} from each model's largest-area quartile"
        )
        print(
            "Per-model background quota: "
            f"max(positive * ratio, {args.min_background_centers}, "
            f"ceil(background * {args.background_coverage_fraction:g}))"
        )
    elif args.training_mode == "full-balanced":
        print(f"Full graph with balanced loss mask, background ratio: {args.background_ratio}")
    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
    print(f"Training on: {device}")
    if args.model_architecture == "dual-head":
        print(f"Dual-head positive weights [rivet, surface]: {dual_head_positive_weights.cpu().tolist()}")
        print(f"Dual-head negative weights [rivet, surface]: {dual_head_negative_weights.cpu().tolist()}")
        print(
            f"Decision thresholds: rivet={args.rivet_threshold:g}, "
            f"surface={args.surface_threshold:g}"
        )
    else:
        print(f"Class weights: {class_weights.cpu().tolist()}")
    print(
        "Large-background loss: "
        f"weight={args.large_background_loss_weight:g}, "
        f"background_quantile={args.large_background_loss_quantile:g}, "
        f"area_threshold={large_background_threshold:.8g}"
    )
    if args.model_architecture == "dual-head":
        print(
            "Host-background surface-head loss: "
            f"weight={args.host_background_loss_weight:g}, "
            f"min_area={args.host_background_min_area:g}, "
            "max_shared_boundary_ratio="
            f"{args.host_background_max_shared_boundary_ratio:g}"
        )
        print(
            "Smooth-shell background surface-head loss: "
            f"weight={args.smooth_component_background_loss_weight:g}, "
            f"min_component_area={args.smooth_component_background_min_area:g}"
        )

    surface_model_out = args.surface_model_out or derive_surface_output_path(args.model_out)
    surface_stats_out = args.surface_stats_out or derive_surface_output_path(args.stats_out)
    surface_eval_out = args.surface_eval_out or derive_surface_output_path(args.eval_out)
    best_miou = float("-inf")
    best_surface_f1 = float("-inf")
    best_safety_score = None
    best_miou_epoch = None
    best_surface_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            target_mask = getattr(batch, "target_mask", torch.ones(batch.y.size(0), dtype=torch.bool, device=batch.y.device))
            sample_weights = large_background_sample_weights(
                batch.y[target_mask],
                batch.normalized_area[target_mask],
                large_background_threshold,
                args.large_background_loss_weight,
            )
            if args.model_architecture == "dual-head":
                head_sample_weights = torch.ones(
                    (int(target_mask.sum().item()), 2),
                    dtype=torch.float,
                    device=batch.y.device,
                )
                head_sample_weights = apply_large_background_surface_head_weight(
                    head_sample_weights,
                    batch.y[target_mask],
                    batch.normalized_area[target_mask],
                    large_background_threshold,
                    args.large_background_loss_weight,
                )
                head_sample_weights = apply_host_background_surface_head_weight(
                    head_sample_weights,
                    batch.y[target_mask],
                    batch.normalized_area[target_mask],
                    batch.shared_boundary_ratio[target_mask],
                    args.host_background_min_area,
                    args.host_background_max_shared_boundary_ratio,
                    args.host_background_loss_weight,
                )
                head_sample_weights = apply_smooth_component_background_surface_head_weight(
                    head_sample_weights,
                    batch.y[target_mask],
                    batch.smooth_component_normalized_area[target_mask],
                    batch.smooth_component_face_count[target_mask],
                    args.smooth_component_background_min_area,
                    args.smooth_component_background_loss_weight,
                )
                loss = dual_head_focal_loss(
                    output[target_mask],
                    batch.y[target_mask],
                    dual_head_positive_weights,
                    args.focal_gamma,
                    head_sample_weights=head_sample_weights,
                    negative_weights=dual_head_negative_weights,
                )
            else:
                loss = focal_nll_loss(
                    output[target_mask],
                    batch.y[target_mask],
                    class_weights,
                    args.focal_gamma,
                    sample_weights=sample_weights,
                )
            loss.backward()
            optimizer.step()

            target_count = int(target_mask.sum().item())
            total_loss += loss.item() * target_count
            total_nodes += target_count

        train_loss = total_loss / max(total_nodes, 1)
        if test_only:
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | final_test=pending")
            continue

        validation_due = epoch % args.validation_interval == 0 or epoch == args.epochs
        if not validation_due:
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | validation=skipped")
            continue

        val_loss, val_acc, val_metrics, val_summary = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            class_weights=class_weights,
            dual_head_positive_weights=dual_head_positive_weights,
            dual_head_negative_weights=dual_head_negative_weights,
            focal_gamma=args.focal_gamma,
            model_architecture=args.model_architecture,
            rivet_threshold=args.rivet_threshold,
            surface_threshold=args.surface_threshold,
            smooth_shell_guard=args.enable_smooth_shell_surface_guard,
        )

        # Precision-first checkpoint policy: false positives are the primary
        # business risk for both heads; mIoU is only a tie-breaker.
        rivet_eligible = val_metrics[1]["tp"] > 0
        surface_recall_eligible = (
            val_metrics[2]["recall"] >= args.surface_recall_floor
        )
        safety_score = (
            -val_summary["rivet_fp"],
            -val_summary["surface_fp"],
            val_summary["class_2_f1"],
            val_summary["miou"],
        )
        if (
            rivet_eligible
            and surface_recall_eligible
            and (best_miou_epoch is None or safety_score > best_safety_score)
        ):
            best_miou = val_summary["miou"]
            best_miou_epoch = epoch
            best_safety_score = safety_score
            save_checkpoint_bundle(
                model,
                args.model_out,
                args.stats_out,
                args.eval_out,
                feature_cols,
                feature_mean,
                feature_std,
                edge_mean,
                edge_std,
                args.num_layers,
                val_metrics,
                val_summary,
                val_loss,
                val_acc,
                label_names=label_names,
                classification_schema="three_class_surface_feature_v1",
                hidden_dim=args.hidden_dim,
                training_mode=args.training_mode,
                window_hop=args.window_hop,
                training_seed=args.seed,
                training_epochs=args.epochs,
                resample_window_centers=(
                    args.training_mode == "window" and not args.fixed_window_centers
                ),
                min_background_centers=args.min_background_centers,
                background_coverage_fraction=args.background_coverage_fraction,
                large_background_loss_weight=args.large_background_loss_weight,
                large_background_loss_quantile=args.large_background_loss_quantile,
                large_background_area_threshold=large_background_threshold,
                host_background_loss_weight=args.host_background_loss_weight,
                host_background_min_area=args.host_background_min_area,
                host_background_max_shared_boundary_ratio=args.host_background_max_shared_boundary_ratio,
                model_architecture=args.model_architecture,
                rivet_threshold=args.rivet_threshold,
                surface_threshold=args.surface_threshold,
            )

        surface_eligible = val_metrics[1]["tp"] > 0 and val_metrics[2]["tp"] > 0
        if surface_eligible and val_summary["class_2_f1"] > best_surface_f1:
            best_surface_f1 = val_summary["class_2_f1"]
            best_surface_epoch = epoch
            save_checkpoint_bundle(
                model,
                surface_model_out,
                surface_stats_out,
                surface_eval_out,
                feature_cols,
                feature_mean,
                feature_std,
                edge_mean,
                edge_std,
                args.num_layers,
                val_metrics,
                val_summary,
                val_loss,
                val_acc,
                label_names=label_names,
                classification_schema="three_class_surface_feature_v1",
                hidden_dim=args.hidden_dim,
                training_mode=args.training_mode,
                window_hop=args.window_hop,
                training_seed=args.seed,
                training_epochs=args.epochs,
                resample_window_centers=(
                    args.training_mode == "window" and not args.fixed_window_centers
                ),
                min_background_centers=args.min_background_centers,
                background_coverage_fraction=args.background_coverage_fraction,
                large_background_loss_weight=args.large_background_loss_weight,
                large_background_loss_quantile=args.large_background_loss_quantile,
                large_background_area_threshold=large_background_threshold,
                host_background_loss_weight=args.host_background_loss_weight,
                host_background_min_area=args.host_background_min_area,
                host_background_max_shared_boundary_ratio=args.host_background_max_shared_boundary_ratio,
                model_architecture=args.model_architecture,
                rivet_threshold=args.rivet_threshold,
                surface_threshold=args.surface_threshold,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"miou={val_summary['miou']:.4f} | "
            f"foreground_f1={val_summary['foreground_macro_f1']:.4f} | "
            f"{label_names[2]}_f1={val_summary['class_2_f1']:.4f} | "
            f"background_fp={val_summary['background_fp']} | "
            f"surface_eligible={surface_eligible}"
        )

    if test_only:
        test_loss, test_acc, test_metrics, test_summary = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            class_weights=class_weights,
            dual_head_positive_weights=dual_head_positive_weights,
            dual_head_negative_weights=dual_head_negative_weights,
            focal_gamma=args.focal_gamma,
            model_architecture=args.model_architecture,
            rivet_threshold=args.rivet_threshold,
            surface_threshold=args.surface_threshold,
            smooth_shell_guard=args.enable_smooth_shell_surface_guard,
        )
        save_checkpoint_bundle(
            model,
            args.model_out,
            args.stats_out,
            args.eval_out,
            feature_cols,
            feature_mean,
            feature_std,
            edge_mean,
            edge_std,
            args.num_layers,
            test_metrics,
            test_summary,
            test_loss,
            test_acc,
            label_names=label_names,
            classification_schema="three_class_surface_feature_v1",
            hidden_dim=args.hidden_dim,
            training_mode=args.training_mode,
            window_hop=args.window_hop,
            training_seed=args.seed,
            training_epochs=args.epochs,
            resample_window_centers=(
                args.training_mode == "window" and not args.fixed_window_centers
            ),
            min_background_centers=args.min_background_centers,
            background_coverage_fraction=args.background_coverage_fraction,
            large_background_loss_weight=args.large_background_loss_weight,
            large_background_loss_quantile=args.large_background_loss_quantile,
            large_background_area_threshold=large_background_threshold,
            host_background_loss_weight=args.host_background_loss_weight,
            host_background_min_area=args.host_background_min_area,
            host_background_max_shared_boundary_ratio=args.host_background_max_shared_boundary_ratio,
            model_architecture=args.model_architecture,
            rivet_threshold=args.rivet_threshold,
            surface_threshold=args.surface_threshold,
        )
        print_metric_table(test_metrics, test_summary, split_name="Final test")
        print(f"Final test checkpoint after epoch={args.epochs}")
        print(f"  model: {args.model_out}")
        print(f"  stats: {args.stats_out}")
        print(f"  eval:  {args.eval_out}")
        return

    if best_miou_epoch is None:
        if best_surface_epoch is None:
            raise RuntimeError("No validation pass produced a checkpoint.")
        # The safety selector may reject every epoch when the configured
        # surface recall floor is too strict. Preserve the best surface
        # checkpoint as a recoverable fallback instead of losing the run.
        shutil.copy2(surface_model_out, args.model_out)
        shutil.copy2(surface_stats_out, args.stats_out)
        shutil.copy2(surface_eval_out, args.eval_out)
        best_miou_epoch = best_surface_epoch
        print(
            "Warning: no checkpoint met surface_recall_floor; "
            f"fell back to best surface checkpoint at epoch={best_surface_epoch}."
        )

    if best_safety_score is None:
        print(
            f"Best precision-first checkpoint: fallback to surface epoch={best_miou_epoch}; "
            "no epoch met the safety constraints"
        )
    else:
        print(
            f"Best precision-first checkpoint: epoch={best_miou_epoch}, "
            f"miou_tiebreak={best_miou:.4f}"
        )
    print(f"  model: {args.model_out}")
    print(f"  stats: {args.stats_out}")
    print(f"  eval:  {args.eval_out}")
    if best_surface_epoch is None:
        print(f"No {label_names[2]} checkpoint was saved because rivet or {label_names[2]} had zero true positives at every validation pass.")
    else:
        print(
            f"Best {label_names[2]} checkpoint: "
            f"epoch={best_surface_epoch}, score={best_surface_f1:.4f}"
        )
        print(f"  model: {surface_model_out}")
        print(f"  stats: {surface_stats_out}")
        print(f"  eval:  {surface_eval_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RivetGNN using one STEP model per graph.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Training CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--val-csv", type=Path, help="Explicit validation CSV path. Random graph splits are disabled.")
    parser.add_argument("--test-csv", type=Path, help="Final test CSV for one post-training evaluation. Cannot be combined with --val-csv.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help=f"Model output path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats output. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_PATH, help=f"Evaluation CSV output. Default: {DEFAULT_EVAL_PATH}")
    parser.add_argument("--surface-model-out", type=Path, help="Best eligible surface-feature-F1 model output. Derived from --model-out by default.")
    parser.add_argument("--surface-stats-out", type=Path, help="Stats paired with --surface-model-out. Derived from --stats-out by default.")
    parser.add_argument("--surface-eval-out", type=Path, help="Evaluation CSV for the best surface-feature checkpoint. Derived from --eval-out by default.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs. Default: 150")
    parser.add_argument("--validation-interval", type=int, default=5, help="Run full validation every N epochs, plus the final epoch. Default: 5")
    parser.add_argument("--batch-size", type=int, default=8, help="Graphs per batch. Default: 8")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension. Default: 64")
    parser.add_argument("--num-layers", type=int, default=4, help="Residual NNConv layers. Default: 4")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate. Default: 0.005")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout after each GNN layer. Default: 0.2")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay. Default: 0.0001")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal-loss gamma. Default: 2.0")
    parser.add_argument("--max-class-weight", type=float, default=5.0, help="Maximum inverse-sqrt class weight before normalization. Default: 5.0")
    parser.add_argument(
        "--surface-feature-weight-scale",
        type=float,
        default=2.0,
        help="Multiplier for the surface_feature (class 2) loss weight. Default: 2.0",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--model-architecture",
        choices=["three-class", "dual-head"],
        default="dual-head",
        help="Three-class softmax or independent rivet/surface binary heads. Default: dual-head.",
    )
    parser.add_argument(
        "--rivet-positive-weight-scale",
        type=float,
        default=1.0,
        help="Additional positive-loss scale for the dual-head rivet output. Default: 1.",
    )
    parser.add_argument(
        "--rivet-negative-weight",
        type=float,
        default=4.0,
        help="Negative-loss multiplier for rivet non-positive faces in dual-head mode. Default: 4.",
    )
    parser.add_argument(
        "--surface-negative-weight",
        type=float,
        default=1.0,
        help="Negative-loss multiplier for surface non-positive faces in dual-head mode. Default: 1.",
    )
    parser.add_argument(
        "--rivet-threshold",
        type=float,
        default=0.9,
        help="Dual-head rivet decision threshold saved with the checkpoint. Default: 0.9.",
    )
    parser.add_argument(
        "--surface-threshold",
        type=float,
        default=0.5,
        help="Dual-head surface decision threshold saved with the checkpoint. Default: 0.5.",
    )
    parser.add_argument(
        "--surface-recall-floor",
        type=float,
        default=0.50,
        help="Minimum validation recall required for a precision-first checkpoint. Default: 0.50.",
    )
    parser.add_argument("--training-mode", choices=["full", "full-balanced", "window"], default="full", help="Use full graphs, balanced-loss full graphs, or k-hop windows. Default: full")
    parser.add_argument("--window-hop", type=int, default=2, help="k-hop radius for window training. Default: 2")
    parser.add_argument("--background-ratio", type=int, default=5, help="Background center samples per positive center in window mode. Default: 5")
    parser.add_argument(
        "--min-background-centers",
        type=int,
        default=0,
        help="Minimum background centers sampled per training model and epoch. Default: 0.",
    )
    parser.add_argument(
        "--background-coverage-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of each model's background faces sampled per epoch. Default: 0.25.",
    )
    parser.add_argument(
        "--fixed-window-centers",
        dest="fixed_window_centers",
        action="store_true",
        help="Reuse one sampled set of window centers for every epoch (default).",
    )
    parser.add_argument(
        "--resample-window-centers",
        dest="fixed_window_centers",
        action="store_false",
        help="Experimentally resample window centers deterministically each epoch.",
    )
    parser.set_defaults(fixed_window_centers=False)
    parser.add_argument(
        "--large-background-fraction",
        type=float,
        default=0.0,
        help="Fraction of each training model's background window quota sampled from its largest-area quartile. Default: 0.",
    )
    parser.add_argument(
        "--large-background-loss-weight",
        type=float,
        default=2.0,
        help="Loss multiplier for true background faces above the configured area quantile. Default: 2.",
    )
    parser.add_argument(
        "--large-background-loss-quantile",
        type=float,
        default=0.9,
        help="Training-background normalized-area quantile used by the large-background loss. Default: 0.9.",
    )
    parser.add_argument(
        "--host-background-loss-weight",
        type=float,
        default=3.0,
        help="Surface-head loss multiplier for large, weakly attached true-background host faces. Default: 3.",
    )
    parser.add_argument(
        "--host-background-min-area",
        type=float,
        default=0.01,
        help="Minimum normalized area for host-background loss. Default: 0.01.",
    )
    parser.add_argument(
        "--host-background-max-shared-boundary-ratio",
        type=float,
        default=0.2,
        help="Maximum shared-boundary/perimeter ratio for host-background loss. Default: 0.2.",
    )
    parser.add_argument(
        "--smooth-component-background-loss-weight",
        type=float,
        default=4.0,
        help="Surface-head loss multiplier for true-background tangent shell components. Default: 4.",
    )
    parser.add_argument(
        "--smooth-component-background-min-area",
        type=float,
        default=0.25,
        help="Minimum normalized tangent-component area for shell-background loss. Default: 0.25.",
    )
    parser.add_argument(
        "--enable-smooth-shell-surface-guard",
        action="store_true",
        help="At inference, reject surface predictions on dominant faces of large tangent shell components.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
