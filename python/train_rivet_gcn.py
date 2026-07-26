import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_max_pool, global_mean_pool


BASE_FEATURE_COLS = [
    "relativeArea",
    "compactness",
    "meanCurvature",
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

DEFAULT_CSV = Path("data/wing_rivet_training_set.csv")
DEFAULT_MODEL_PATH = Path("rivet_gnn_no_centerz_split19_5.pth")
DEFAULT_STATS_PATH = Path("rivet_gnn_no_centerz_split19_5_stats.npz")
DEFAULT_EVAL_PATH = Path("rivet_gnn_no_centerz_split19_5_eval.csv")

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
    surface_types = sorted({int(value) for value in df[SURFACE_TYPE_COL].fillna(0).astype(int).tolist()})
    feature_cols = []
    for col in BASE_FEATURE_COLS:
        if col == RADIUS_COL:
            feature_cols.append(RADIUS_COL)
            feature_cols.append(HAS_RADIUS_COL)
        else:
            feature_cols.append(col)
    feature_cols.extend(RELATIVE_NORMAL_FEATURE_COLS)
    if {"centerX", "centerY", "centerZ"}.issubset(df.columns):
        feature_cols.extend(MODEL_RELATIVE_POSITION_FEATURE_COLS)
    feature_cols.extend(f"{SURFACE_TYPE_COL}_{surface_type}" for surface_type in surface_types)
    return feature_cols


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

    for col in BASE_FEATURE_COLS:
        if col == RADIUS_COL:
            radius_values = numeric_df[RADIUS_COL].fillna(0.0).astype(float)
            feature_df[RADIUS_COL] = radius_values
            feature_df[HAS_RADIUS_COL] = (radius_values.abs() > 1e-9).astype(float)
        else:
            feature_df[col] = numeric_df[col].fillna(0.0).astype(float)

    for col in RELATIVE_NORMAL_FEATURE_COLS:
        feature_df[col] = numeric_df[col].fillna(0.0).astype(float)

    if {"centerX", "centerY", "centerZ"}.issubset(df.columns):
        positions = numeric_df[["centerX", "centerY", "centerZ"]].fillna(0.0).to_numpy(dtype=float)
        relative_position = np.zeros((len(df), 2), dtype=float)
        for _, indices in df.groupby("graph_id", sort=False).groups.items():
            index_array = np.fromiter(indices, dtype=int)
            coords = positions[index_array]
            centered = coords - coords.mean(axis=0, keepdims=True)
            axis_index = int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))
            axial = centered[:, axis_index]
            radial = np.sqrt(np.sum(np.delete(centered, axis_index, axis=1) ** 2, axis=1))
            scale = max(np.sqrt(np.sum(centered ** 2, axis=1)).max(), 1e-6)
            relative_position[index_array, 0] = np.abs(axial) / scale
            relative_position[index_array, 1] = radial / scale
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
    feature_frame, _ = build_feature_frame(local_df, feature_cols)
    node_features = feature_frame.to_numpy()
    node_features = (node_features - feature_mean) / feature_std

    id_to_index = {int(row_id): idx for idx, row_id in enumerate(local_df["id"].tolist())}
    edge_index = []
    edge_attr = []

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
            edge_attr.append([
                float(token_at(edge_attr_tokens[col], edge_pos))
                for col in EDGE_ATTR_COLS
            ])

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

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        target_mask=torch.ones(y.size(0), dtype=torch.bool),
        graph_id=int(local_df["graph_id"].iloc[0]),
        model_name=str(local_df["model_name"].iloc[0]),
    )


def khop_nodes(edge_index, center_idx, hop_count, num_nodes):
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


def build_window_graph(full_graph, center_idx, hop_count):
    node_ids = khop_nodes(full_graph.edge_index, center_idx, hop_count, full_graph.num_nodes)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_ids)}

    local_edges = []
    local_edge_attrs = []
    for edge_pos, (src, dst) in enumerate(full_graph.edge_index.t().tolist()):
        if src in old_to_new and dst in old_to_new:
            local_edges.append([old_to_new[src], old_to_new[dst]])
            local_edge_attrs.append(full_graph.edge_attr[edge_pos].tolist())

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
        target_mask=target_mask,
        graph_id=int(full_graph.graph_id),
        model_name=str(full_graph.model_name),
        center_face_id=center_idx + 1,
    )


def sample_window_centers(full_graph, background_ratio, rng):
    labels = full_graph.y.cpu().numpy()
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels == 0)

    if background_ratio is None or background_ratio <= 0 or len(positive) == 0:
        return np.arange(full_graph.num_nodes)

    max_negative = min(len(negative), len(positive) * background_ratio)
    if max_negative <= 0:
        centers = positive.astype(int)
        rng.shuffle(centers)
        return centers

    random_negative = rng.choice(negative, size=max_negative, replace=False)
    centers = np.concatenate([positive, random_negative]).astype(int)
    rng.shuffle(centers)
    return centers


def build_window_dataset(full_graphs, hop_count, background_ratio, seed):
    rng = np.random.default_rng(seed)
    windows = []
    for full_graph in full_graphs:
        for center_idx in sample_window_centers(full_graph, background_ratio, rng):
            windows.append(build_window_graph(full_graph, int(center_idx), hop_count))
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


def collect_edge_attr_matrix(df):
    rows = []
    for _, row in df.iterrows():
        neighbor_ids = split_tokens(row["neighbors"])
        edge_attr_tokens = {
            col: split_tokens(row[col]) if col in df.columns else []
            for col in EDGE_ATTR_COLS
        }
        for edge_pos in range(len(neighbor_ids)):
            rows.append([
                float(token_at(edge_attr_tokens[col], edge_pos))
                for col in EDGE_ATTR_COLS
            ])

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


def load_cad_data(csv_path, stats_path=DEFAULT_STATS_PATH):
    df = pd.read_csv(csv_path)
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export inference data.")

    feature_mean, feature_std, edge_mean, edge_std, feature_cols = load_feature_stats(stats_path)
    if feature_mean is None or feature_std is None:
        feature_frame, feature_cols = build_feature_frame(df)
        feature_matrix = feature_frame.to_numpy()
        feature_mean = feature_matrix.mean(axis=0)
        feature_std = feature_matrix.std(axis=0) + 1e-6
    elif feature_cols is None:
        feature_cols = infer_feature_columns(df)
    if edge_mean is None or edge_std is None:
        edge_matrix = collect_edge_attr_matrix(df)
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


def prepare_graph_samples(full_graphs, training_mode="full", window_hop=2, background_ratio=5, seed=42):
    if training_mode == "window":
        return build_window_dataset(full_graphs, window_hop, background_ratio, seed)
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
    edge_matrix = collect_edge_attr_matrix(df)
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
):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    for name, df in [("train", train_df), ("val", val_df)]:
        if "graph_id" not in df.columns or "model_name" not in df.columns:
            raise ValueError(f"{name} CSV is missing graph_id/model_name columns. Please re-export training data.")

    train_feature_frame, feature_cols = build_feature_frame(train_df)
    feature_matrix = train_feature_frame.to_numpy()
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-6
    edge_matrix = collect_edge_attr_matrix(train_df)
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
    )
    val_training_mode = "full" if training_mode == "full-balanced" else training_mode
    val_graphs = prepare_graph_samples(
        val_full_graphs,
        val_training_mode,
        window_hop,
        background_ratio,
        seed + 1,
    )
    return train_graphs, val_graphs, feature_mean, feature_std, edge_mean, edge_std, feature_cols


class RivetGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes=2, num_layers=3):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be at least 3")
        self.num_layers = num_layers

        def create_nn(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Linear(edge_features, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, in_f * out_f),
            )

        if num_layers == 3:
            self.conv1 = NNConv(node_features, hidden_dim, create_nn(node_features, hidden_dim))
            self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
            self.conv2 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
            self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
            self.conv3 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
            self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)
            return
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer_index in range(num_layers):
            in_features = node_features if layer_index == 0 else hidden_dim
            self.convs.append(NNConv(in_features, hidden_dim, create_nn(in_features, hidden_dim)))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.classifier = torch.nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.num_layers == 3:
            x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
            identity = x
            x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
            x = x + identity
            x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
            return F.log_softmax(self.classifier(x), dim=1)
        for layer_index, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            updated = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = updated if layer_index == 0 else updated + x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_features = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = self.classifier(torch.cat([x, global_features[batch]], dim=1))
        return F.log_softmax(x, dim=1)


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
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "miou": 0.0}
    return {
        "macro_precision": float(np.mean([item["precision"] for item in metrics])),
        "macro_recall": float(np.mean([item["recall"] for item in metrics])),
        "macro_f1": float(np.mean([item["f1"] for item in metrics])),
        "miou": float(np.mean([item["iou"] for item in metrics])),
    }


def focal_nll_loss(log_probabilities, targets, class_weights=None, gamma=2.0):
    target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    weights = class_weights[targets] if class_weights is not None else 1.0
    return ((1.0 - target_log_probabilities.exp()).pow(gamma) * -target_log_probabilities * weights).mean()


def evaluate(model, loader, device, num_classes, class_weights=None, focal_gamma=2.0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            target_mask = getattr(batch, "target_mask", torch.ones(batch.y.size(0), dtype=torch.bool, device=batch.y.device))
            loss = focal_nll_loss(out[target_mask], batch.y[target_mask], class_weights, focal_gamma)

            pred = out.argmax(dim=1)
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


def write_eval_csv(output_path, metrics, summary, val_loss, val_acc):
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "class_id",
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
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                **item,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **summary,
            })


def print_metric_table(metrics, summary):
    print("Validation class metrics:")
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
        f"mIoU={summary['miou']:.4f}"
    )


def train(args):
    if args.val_csv is None:
        raise ValueError("Explicit --val-csv is required. Random graph splits are not part of the main protocol.")

    train_graphs, val_graphs, feature_mean, feature_std, edge_mean, edge_std, feature_cols = load_explicit_train_val_datasets(
        args.csv,
        args.val_csv,
        training_mode=args.training_mode,
        window_hop=args.window_hop,
        background_ratio=args.background_ratio,
        seed=args.seed,
    )
    graphs = train_graphs + val_graphs
    split_description = f"explicit train={args.csv}, val={args.val_csv}"

    if not train_graphs or not val_graphs:
        raise ValueError("Training and validation datasets must both contain at least one graph/sample.")

    num_classes = max(int(graph.y.max().item()) for graph in graphs) + 1

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RivetGNN(
        node_features=train_graphs[0].num_node_features,
        edge_features=train_graphs[0].edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(
        train_graphs,
        num_classes=num_classes,
        max_weight=args.max_class_weight,
    ).to(device)

    print(f"Loaded {len(graphs)} graph samples ({split_description})")
    print(f"Classes: {num_classes}")
    print(f"Training mode: {args.training_mode}")
    if args.training_mode == "window":
        print(f"Window hop: {args.window_hop}, background ratio: {args.background_ratio}")
    elif args.training_mode == "full-balanced":
        print(f"Full graph with balanced loss mask, background ratio: {args.background_ratio}")
    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
    print(f"Training on: {device}")
    print(f"Class weights: {class_weights.cpu().tolist()}")

    best_miou = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target_mask = getattr(batch, "target_mask", torch.ones(batch.y.size(0), dtype=torch.bool, device=batch.y.device))
            loss = focal_nll_loss(out[target_mask], batch.y[target_mask], class_weights, args.focal_gamma)
            loss.backward()
            optimizer.step()

            target_count = int(target_mask.sum().item())
            total_loss += loss.item() * target_count
            total_nodes += target_count

        train_loss = total_loss / max(total_nodes, 1)
        val_loss, val_acc, val_metrics, val_summary = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
        )

        if val_summary["miou"] > best_miou:
            best_miou = val_summary["miou"]
            torch.save(model.state_dict(), args.model_out)
            np.savez(
                args.stats_out,
                feature_cols=np.array(feature_cols, dtype=object),
                edge_attr_cols=np.array(EDGE_ATTR_COLS, dtype=object),
                mean=feature_mean,
                std=feature_std,
                edge_mean=edge_mean,
                edge_std=edge_std,
                num_layers=np.array(args.num_layers),
            )
            write_eval_csv(args.eval_out, val_metrics, val_summary, val_loss, val_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"miou={val_summary['miou']:.4f}"
            )

    best_state = torch.load(args.model_out, map_location=device)
    model.load_state_dict(best_state)
    final_val_loss, final_val_acc, final_metrics, final_summary = evaluate(
        model,
        val_loader,
        device,
        num_classes,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    )
    write_eval_csv(args.eval_out, final_metrics, final_summary, final_val_loss, final_val_acc)
    print_metric_table(final_metrics, final_summary)
    print(f"Best model saved to: {args.model_out}")
    print(f"Feature stats saved to: {args.stats_out}")
    print(f"Evaluation metrics saved to: {args.eval_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RivetGNN using one STEP model per graph.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Training CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--val-csv", type=Path, required=True, help="Explicit validation CSV path. Random graph splits are disabled.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help=f"Model output path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats output. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_PATH, help=f"Evaluation CSV output. Default: {DEFAULT_EVAL_PATH}")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs. Default: 150")
    parser.add_argument("--batch-size", type=int, default=8, help="Graphs per batch. Default: 8")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension. Default: 64")
    parser.add_argument("--num-layers", type=int, default=6, help="Residual NNConv layers. Default: 6")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate. Default: 0.005")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal-loss gamma. Default: 2.0")
    parser.add_argument("--max-class-weight", type=float, default=5.0, help="Maximum inverse-sqrt class weight before normalization. Default: 5.0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--training-mode", choices=["full", "full-balanced", "window"], default="full", help="Use full graphs, balanced-loss full graphs, or k-hop windows. Default: full")
    parser.add_argument("--window-hop", type=int, default=2, help="k-hop radius for window training. Default: 2")
    parser.add_argument("--background-ratio", type=int, default=5, help="Background center samples per positive center in window mode. Default: 5")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
