import argparse
import csv
import subprocess
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from train_rivet_gcn import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    RivetGNN,
    build_window_graph,
    load_cad_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CSV = PROJECT_ROOT / "data" / "current_inference.csv"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Release" / "Detector.exe"


def load_truth_labels(labels_path):
    import json

    truth = json.load(Path(labels_path).open("r", encoding="utf-8"))
    return {
        int(row["face_id"]): (1 if row["semantic"] == "rivet" else 0)
        for row in truth["faces"]
    }


def export_inference_csv(step_path, detector_path, csv_path):
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector executable not found: {detector_path}")

    result = subprocess.run(
        [str(detector_path), "--predict", str(step_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Detector failed to export inference CSV.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"Inference CSV was not generated: {csv_path}")


def run_inference(csv_path, model_path, stats_path, hidden_dim, inference_mode="full", window_hop=2, inference_batch_size=128):
    data = load_cad_data(csv_path, stats_path=stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    classifier_weight = state_dict.get("classifier.weight")
    num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 3
    model = RivetGNN(
        node_features=data.num_node_features,
        edge_features=1,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    if inference_mode == "window":
        windows = [build_window_graph(data.cpu(), center_idx, window_hop) for center_idx in range(data.num_nodes)]
        loader = DataLoader(windows, batch_size=inference_batch_size, shuffle=False)
        pred = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                pred.extend(out.argmax(dim=1)[batch.target_mask].cpu().tolist())
    else:
        with torch.no_grad():
            out = model(data.to(device))
            pred = out.argmax(dim=1).cpu().tolist()

    return pred, num_classes


def write_predictions_csv(prediction_path, pred_labels):
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["face_id", "pred_label"])
        for face_id, label in enumerate(pred_labels, start=1):
            writer.writerow([face_id, label])
    print(f"Prediction CSV ready: {prediction_path}")


def suppress_large_positive_faces(csv_path, pred_labels, max_relative_area):
    if max_relative_area is None or max_relative_area <= 0:
        return pred_labels

    filtered_labels = list(pred_labels)
    suppressed_count = 0

    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            face_idx = int(row["id"]) - 1
            if face_idx < 0 or face_idx >= len(filtered_labels):
                continue

            if filtered_labels[face_idx] == 1 and float(row["relativeArea"]) > max_relative_area:
                filtered_labels[face_idx] = 0
                suppressed_count += 1

    if suppressed_count:
        print(
            "Suppressed large positive faces: "
            f"{suppressed_count} (relativeArea > {max_relative_area:g})"
        )

    return filtered_labels


def visualize_cad_results(step_path, pred_labels, labels_path):
    from OCC.Core.Quantity import (
        Quantity_NOC_GRAY,
        Quantity_NOC_GREEN,
        Quantity_NOC_RED,
        Quantity_NOC_YELLOW,
    )
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Display.SimpleGui import init_display

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    display, start_display, _, _ = init_display()
    truth_labels = load_truth_labels(labels_path)
    print(f"Displaying comparison: {step_path}")
    print("green=TP, red=FP, yellow=FN, gray=TN")

    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    num_faces = face_map.Size()
    print(f"Model faces: {num_faces}, Predicted labels: {len(pred_labels)}")

    label_counts = {}
    for label in pred_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")

    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        label = pred_labels[i - 1] if (i - 1) < len(pred_labels) else 0

        truth_label = truth_labels.get(i, 0)
        if label == 1 and truth_label == 1:
            color = Quantity_NOC_GREEN
            confusion["tp"] += 1
        elif label == 1 and truth_label == 0:
            color = Quantity_NOC_RED
            confusion["fp"] += 1
        elif label == 0 and truth_label == 1:
            color = Quantity_NOC_YELLOW
            confusion["fn"] += 1
        else:
            color = Quantity_NOC_GRAY
            confusion["tn"] += 1
        display.DisplayShape(face, color=color, update=False)

    print(f"Comparison stats: {confusion}")

    display.FitAll()
    start_display()


def parse_args():
    parser = argparse.ArgumentParser(description="Run RivetGNN inference and visualize face labels on a STEP model.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help=f"Detector executable path. Default: {DEFAULT_DETECTOR}")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INFERENCE_CSV, help=f"Inference CSV path. Default: {DEFAULT_INFERENCE_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help=f"Trained model path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats path. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--pred-out", type=Path, help="Optional prediction CSV output path.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension used during training. Default: 64")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing inference CSV instead of calling Detector.")
    parser.add_argument("--no-display", action="store_true", help="Only export predictions; do not open the OCC viewer.")
    parser.add_argument("--compare-labels", type=Path, required=True, help="labels.json for true-vs-prediction comparison visualization.")
    parser.add_argument("--inference-mode", choices=["full", "window"], default="full", help="Run full-graph inference or per-face k-hop window inference. Default: full")
    parser.add_argument("--window-hop", type=int, default=2, help="k-hop radius for window inference. Default: 2")
    parser.add_argument("--inference-batch-size", type=int, default=128, help="Window inference batch size. Default: 128")
    parser.add_argument("--max-rivet-relative-area", type=float, default=1e-4, help="Suppress predicted rivet faces larger than this relative area. Use <=0 to disable. Default: 1e-4")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_export:
        export_inference_csv(args.step_model, args.detector, args.csv)

    predicted_labels, num_classes = run_inference(
        args.csv,
        args.model,
        args.stats,
        args.hidden_dim,
        inference_mode=args.inference_mode,
        window_hop=args.window_hop,
        inference_batch_size=args.inference_batch_size,
    )
    print("Inference finished.")
    predicted_labels = suppress_large_positive_faces(args.csv, predicted_labels, args.max_rivet_relative_area)
    if args.pred_out:
        write_predictions_csv(args.pred_out, predicted_labels)
    if not args.no_display:
        visualize_cad_results(args.step_model, predicted_labels, labels_path=args.compare_labels)


if __name__ == "__main__":
    main()
