import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from train_rivet_gcn import (
    RivetGNN,
    build_graphs_from_dataframe,
    build_window_cache,
    build_window_graph,
    label_names_from_stats,
)


def error_type(truth_label, pred_label, class_names):
    truth_name = class_names.get(truth_label, f"class_{truth_label}")
    pred_name = class_names.get(pred_label, f"class_{pred_label}")
    return f"{truth_name}_to_{pred_name}"


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Audit complete model-level RivetGNN predictions without changing labels or training data."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Complete model-level labeled CSV.")
    parser.add_argument("--model", type=Path, required=True, help="RivetGNN checkpoint.")
    parser.add_argument("--stats", type=Path, required=True, help="Normalization stats paired with the checkpoint.")
    parser.add_argument("--out", type=Path, required=True, help="Misclassified-face review CSV.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Grouped confusion summary CSV.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension. Default: 64")
    parser.add_argument("--window-hop", type=int, default=2, help="Window hop count. Default: 2")
    args = parser.parse_args()

    source_df = pd.read_csv(args.csv)
    required_columns = {"graph_id", "model_name", "id", "label"}
    missing_columns = required_columns.difference(source_df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    input_keys = source_df[["graph_id", "model_name", "id"]].copy()
    if input_keys.duplicated().any():
        raise ValueError("Input CSV contains duplicate graph_id/model_name/id rows.")

    stats = np.load(args.stats, allow_pickle=True)
    feature_cols = [str(value) for value in stats["feature_cols"].tolist()]
    feature_mean = stats["mean"]
    feature_std = stats["std"]
    edge_mean = stats["edge_mean"]
    edge_std = stats["edge_std"]
    num_layers = int(stats["num_layers"]) if "num_layers" in stats.files else 3
    labels = set(source_df["label"].astype(int).unique())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(
            "Audit CSV must use background=0, rivet=1, surface_feature=2; "
            f"found labels={sorted(labels)}"
        )

    full_graphs = build_graphs_from_dataframe(
        source_df,
        feature_mean,
        feature_std,
        edge_mean,
        edge_std,
        feature_cols,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model, map_location=device)
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise ValueError("Checkpoint is missing classifier.weight.")
    num_classes = int(classifier_weight.shape[0])
    label_names = label_names_from_stats(stats, num_classes)
    class_names = dict(enumerate(label_names))
    model = RivetGNN(
        node_features=full_graphs[0].num_node_features,
        edge_features=full_graphs[0].edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    error_rows = []
    prediction_count = 0
    with torch.no_grad():
        for graph in full_graphs:
            graph = graph.cpu()
            window_cache = build_window_cache(graph)
            windows = [
                build_window_graph(graph, center_idx, args.window_hop, window_cache)
                for center_idx in range(graph.num_nodes)
            ]
            loader = DataLoader(windows, batch_size=1, shuffle=False)
            model_errors = 0
            for batch in loader:
                batch = batch.to(device)
                target_log_probabilities = model(batch)[batch.target_mask]
                if target_log_probabilities.size(0) != 1:
                    raise RuntimeError("Window inference must produce exactly one center prediction.")
                probabilities = target_log_probabilities.exp().squeeze(0).cpu().numpy()
                pred_label = int(probabilities.argmax())
                truth_label = int(batch.y[batch.target_mask].item())
                face_id = int(batch.face_id[batch.target_mask].item())
                prediction_count += 1
                if pred_label == truth_label:
                    continue

                row = {
                    "graph_id": int(graph.graph_id),
                    "model_name": str(graph.model_name),
                    "face_id": face_id,
                    "truth_label": truth_label,
                    "truth_name": class_names.get(truth_label, f"class_{truth_label}"),
                    "pred_label": pred_label,
                    "pred_name": class_names.get(pred_label, f"class_{pred_label}"),
                    "pred_confidence": float(probabilities[pred_label]),
                    "error_type": error_type(truth_label, pred_label, class_names),
                }
                for class_id in range(num_classes):
                    row[f"prob_{class_names.get(class_id, f'class_{class_id}')}"] = float(probabilities[class_id])
                error_rows.append(row)
                model_errors += 1

            print(
                f"Audited graph_id={graph.graph_id}, model={graph.model_name}, "
                f"faces={graph.num_nodes}, errors={model_errors}"
            )

    if prediction_count != len(source_df):
        raise RuntimeError(
            f"Prediction count mismatch: predictions={prediction_count}, input_rows={len(source_df)}"
        )

    error_rows.sort(key=lambda row: (-row["pred_confidence"], row["model_name"], row["face_id"]))
    probability_fields = [f"prob_{class_names.get(class_id, f'class_{class_id}')}" for class_id in range(num_classes)]
    review_fields = [
        "graph_id",
        "model_name",
        "face_id",
        "truth_label",
        "truth_name",
        "pred_label",
        "pred_name",
        "pred_confidence",
        "error_type",
        *probability_fields,
    ]
    write_csv(args.out, error_rows, review_fields)

    summary_rows = []
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        grouped = error_df.groupby(
            ["graph_id", "model_name", "truth_label", "truth_name", "pred_label", "pred_name", "error_type"],
            sort=True,
        )
        for keys, group in grouped:
            summary_rows.append({
                "scope": "model",
                "graph_id": keys[0],
                "model_name": keys[1],
                "truth_label": keys[2],
                "truth_name": keys[3],
                "pred_label": keys[4],
                "pred_name": keys[5],
                "error_type": keys[6],
                "count": len(group),
                "mean_confidence": float(group["pred_confidence"].mean()),
                "max_confidence": float(group["pred_confidence"].max()),
            })
        for keys, group in error_df.groupby(
            ["truth_label", "truth_name", "pred_label", "pred_name", "error_type"],
            sort=True,
        ):
            summary_rows.append({
                "scope": "aggregate",
                "graph_id": "",
                "model_name": "",
                "truth_label": keys[0],
                "truth_name": keys[1],
                "pred_label": keys[2],
                "pred_name": keys[3],
                "error_type": keys[4],
                "count": len(group),
                "mean_confidence": float(group["pred_confidence"].mean()),
                "max_confidence": float(group["pred_confidence"].max()),
            })

    summary_fields = [
        "scope",
        "graph_id",
        "model_name",
        "truth_label",
        "truth_name",
        "pred_label",
        "pred_name",
        "error_type",
        "count",
        "mean_confidence",
        "max_confidence",
    ]
    write_csv(args.summary_out, summary_rows, summary_fields)
    print(f"Predictions verified: {prediction_count}/{len(source_df)}")
    print(f"Misclassified faces: {len(error_rows)}")
    print(f"Review CSV: {args.out}")
    print(f"Summary CSV: {args.summary_out}")


if __name__ == "__main__":
    main()
