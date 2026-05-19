import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv


FEATURE_COLS = [
    "relativeArea",
    "compactness",
    "surfaceType",
    "nx",
    "ny",
    "nz",
    "centerZ",
    "meanCurvature",
    "radius",
    "numWires",
    "innerWireCount",
    "minInnerWireLength",
    "maxInnerWireLength",
    "numEdges",
]

DEFAULT_CSV = Path("data/full_training_set.csv")
DEFAULT_MODEL_PATH = Path("rivet_gnn.pth")
DEFAULT_STATS_PATH = Path("rivet_gnn_stats.npz")


def split_tokens(value):
    if pd.isna(value):
        return []
    return str(value).split()


def build_graph(group_df, feature_mean, feature_std):
    local_df = group_df.reset_index(drop=True).copy()
    node_features = local_df[FEATURE_COLS].astype(float).to_numpy()
    node_features = (node_features - feature_mean) / feature_std

    id_to_index = {int(row_id): idx for idx, row_id in enumerate(local_df["id"].tolist())}
    edge_index = []
    edge_attr = []

    for node_idx, row in local_df.iterrows():
        neighbor_ids = split_tokens(row["neighbors"])
        edge_types = split_tokens(row["edge_types"])

        for neighbor_id_str, edge_type_str in zip(neighbor_ids, edge_types):
            neighbor_id = int(neighbor_id_str)
            if neighbor_id not in id_to_index:
                continue

            edge_index.append([node_idx, id_to_index[neighbor_id]])
            edge_attr.append([float(edge_type_str)])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(local_df["label"].astype(int).to_numpy(), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        graph_id=int(local_df["graph_id"].iloc[0]),
        model_name=str(local_df["model_name"].iloc[0]),
    )


def load_feature_stats(stats_path):
    stats_file = Path(stats_path)
    if not stats_file.exists():
        return None, None

    stats = np.load(stats_file, allow_pickle=True)
    return stats["mean"], stats["std"]


def load_cad_data(csv_path, stats_path=DEFAULT_STATS_PATH):
    df = pd.read_csv(csv_path)
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export inference data.")

    feature_mean, feature_std = load_feature_stats(stats_path)
    if feature_mean is None or feature_std is None:
        feature_matrix = df[FEATURE_COLS].astype(float).to_numpy()
        feature_mean = feature_matrix.mean(axis=0)
        feature_std = feature_matrix.std(axis=0) + 1e-6

    first_graph_id = df["graph_id"].iloc[0]
    group = df[df["graph_id"] == first_graph_id]
    return build_graph(group, feature_mean, feature_std)


def load_graph_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "graph_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV is missing graph_id/model_name columns. Please re-export training data.")

    feature_matrix = df[FEATURE_COLS].astype(float).to_numpy()
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-6

    graphs = []
    for _, group in df.groupby("graph_id", sort=True):
        graphs.append(build_graph(group, feature_mean, feature_std))

    return graphs, feature_mean, feature_std


def split_dataset(graphs, train_ratio=0.8, seed=42):
    if len(graphs) < 2:
        return graphs, graphs

    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)

    split_index = max(1, int(len(indices) * train_ratio))
    split_index = min(split_index, len(indices) - 1)

    train_graphs = [graphs[idx] for idx in indices[:split_index]]
    val_graphs = [graphs[idx] for idx in indices[split_index:]]
    return train_graphs, val_graphs


class RivetGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes=3):
        super().__init__()

        def create_nn(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Linear(edge_features, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, in_f * out_f),
            )

        self.conv1 = NNConv(node_features, hidden_dim, create_nn(node_features, hidden_dim))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv2 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv3 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))

        identity = x
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = x + identity

        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def compute_class_weights(graphs, num_classes=3):
    labels = []
    for graph in graphs:
        labels.append(graph.y.cpu())

    if not labels:
        return torch.ones(num_classes, dtype=torch.float)

    all_labels = torch.cat(labels)
    counts = torch.bincount(all_labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights


def evaluate(model, loader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights)

            total_loss += loss.item() * batch.num_nodes
            total_correct += int((out.argmax(dim=1) == batch.y).sum().item())
            total_nodes += batch.num_nodes

    if total_nodes == 0:
        return 0.0, 0.0

    return total_loss / total_nodes, total_correct / total_nodes


def train(args):
    graphs, feature_mean, feature_std = load_graph_dataset(args.csv)
    train_graphs, val_graphs = split_dataset(graphs, train_ratio=args.train_ratio, seed=args.seed)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RivetGNN(
        node_features=train_graphs[0].num_node_features,
        edge_features=1,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(train_graphs).to(device)

    print(f"Loaded {len(graphs)} graphs from {args.csv}")
    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
    print(f"Training on: {device}")
    print(f"Class weights: {class_weights.cpu().tolist()}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        train_loss = total_loss / max(total_nodes, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, class_weights=class_weights)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_out)
            np.savez(
                args.stats_out,
                feature_cols=np.array(FEATURE_COLS, dtype=object),
                mean=feature_mean,
                std=feature_std,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

    print(f"Best model saved to: {args.model_out}")
    print(f"Feature stats saved to: {args.stats_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RivetGNN using one STEP model per graph.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Training CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help=f"Model output path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats output. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs. Default: 150")
    parser.add_argument("--batch-size", type=int, default=8, help="Graphs per batch. Default: 8")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension. Default: 64")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate. Default: 0.005")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio. Default: 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
