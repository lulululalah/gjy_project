import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLS = [
    "hostFaceArea",
    "hostFaceRelativeArea",
    "hostFacePerimeter",
    "hostInnerWireCount",
    "hostNeighborFaceCount",
    "wireLength",
    "wireLengthRatio",
    "estimatedHoleRadius",
    "estimatedHoleArea",
    "adjacentFaceCount",
    "adjacentCylinderCount",
    "adjacentSmallCylinderCount",
    "minAdjacentCylinderRadius",
    "maxAdjacentCylinderRadius",
    "concaveEdgeRatio",
]

DEFAULT_CSV = Path("data/hole_candidate_training_set.csv")
DEFAULT_MODEL_PATH = Path("hole_candidate_mlp.pth")
DEFAULT_STATS_PATH = Path("hole_candidate_stats.npz")


class HoleCandidateMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def split_by_graph(df, train_ratio=0.8, seed=42):
    graph_ids = df["graph_id"].drop_duplicates().to_numpy().copy()
    if len(graph_ids) < 2:
        return df, df

    rng = np.random.default_rng(seed)
    rng.shuffle(graph_ids)
    split_index = max(1, int(len(graph_ids) * train_ratio))
    split_index = min(split_index, len(graph_ids) - 1)

    train_ids = set(graph_ids[:split_index].tolist())
    val_ids = set(graph_ids[split_index:].tolist())
    return df[df["graph_id"].isin(train_ids)].copy(), df[df["graph_id"].isin(val_ids)].copy()


def normalize(train_df, val_df):
    train_x = train_df[FEATURE_COLS].astype(float).to_numpy()
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6

    train_x = (train_x - mean) / std
    val_x = (val_df[FEATURE_COLS].astype(float).to_numpy() - mean) / std
    return train_x, val_x, mean, std


def to_loader(features, labels, batch_size, shuffle):
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels.astype(int).to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(labels):
    y = torch.tensor(labels.astype(int).to_numpy(), dtype=torch.long)
    counts = torch.bincount(y, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)
    return counts.sum() / (len(counts) * counts)


def evaluate(model, loader, device, class_weights):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
            total_loss += loss.item() * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_samples += y.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def train(args):
    df = pd.read_csv(args.csv)
    train_df, val_df = split_by_graph(df, train_ratio=args.train_ratio, seed=args.seed)
    train_x, val_x, mean, std = normalize(train_df, val_df)

    train_loader = to_loader(train_x, train_df["label"], args.batch_size, True)
    val_loader = to_loader(val_x, val_df["label"], args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoleCandidateMLP(len(FEATURE_COLS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(train_df["label"]).to(device)

    print(f"Loaded {len(df)} candidates from {args.csv}")
    print(f"Train graphs: {train_df['graph_id'].nunique()}, Val graphs: {val_df['graph_id'].nunique()}")
    print(f"Train candidates: {len(train_df)}, Val candidates: {len(val_df)}")
    print(f"Training on: {device}")
    print(f"Class weights: {class_weights.cpu().tolist()}")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        train_loss = total_loss / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, class_weights)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_out)
            np.savez(
                args.stats_out,
                feature_cols=np.array(FEATURE_COLS, dtype=object),
                mean=mean,
                std=std,
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
    parser = argparse.ArgumentParser(description="Train a first-pass hole candidate classifier.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
