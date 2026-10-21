"""Face attributes, graph construction and the checkpoint-compatible graph encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
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
]

SMOOTH_COMPONENT_RUNTIME_COLS = [
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

RADIUS_COL = "radius"

SURFACE_TYPE_COL = "surfaceType"

HAS_RADIUS_COL = "has_radius"

def split_tokens(value):
    if pd.isna(value):
        return []
    return str(value).split()

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
    return pd.DataFrame(
        rows,
        index=df.index,
        columns=ATTACHED_SURFACE_FEATURE_COLS + SMOOTH_COMPONENT_RUNTIME_COLS,
    )

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

class FaceGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_classes=3,
        num_layers=4,
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

def backward_graph_batch(model, batch, loss_function, device, micro_batch_size=None):
    """Accumulate a face-weighted batch gradient while limiting resident graphs."""
    target_count = int(batch.target_mask.sum()) if hasattr(batch, "target_mask") else batch.y.numel()
    if target_count <= 0:
        raise ValueError("Training batch has no supervised faces.")
    if micro_batch_size is None:
        batches = [batch]
    else:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        graphs = batch.to_data_list()
        batches = [Batch.from_data_list(graphs[i:i + micro_batch_size])
                   for i in range(0, len(graphs), micro_batch_size)]
    total_loss = 0.0
    for micro in batches:
        micro = micro.to(device)
        count = int(micro.target_mask.sum()) if hasattr(micro, "target_mask") else micro.y.numel()
        if count == 0:
            continue
        loss = loss_function(model(micro), micro)
        (loss * (count / target_count)).backward()
        total_loss += loss.detach().item() * count
    return total_loss, target_count

def compute_class_weights(graphs, num_classes=3, max_weight=5.0):
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
