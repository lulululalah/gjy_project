import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CSV = PROJECT_ROOT / "data" / "current_inference.csv"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Debug" / "Detector.exe"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "rivet_gnn_no_centerz_split19_5.pth"
DEFAULT_STATS_PATH = PROJECT_ROOT / "rivet_gnn_no_centerz_split19_5_stats.npz"

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


def run_inference(
    csv_path,
    model_path,
    stats_path,
    hidden_dim,
    inference_mode="full",
    window_hop=2,
    inference_batch_size=1,
):
    import torch
    from torch_geometric.loader import DataLoader
    from train_rivet_gcn import (
        RivetGNN,
        build_window_cache,
        build_window_graph,
        load_cad_data,
    )

    data = load_cad_data(csv_path, stats_path=stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    stats = np.load(stats_path, allow_pickle=True)
    num_layers = int(stats["num_layers"]) if "num_layers" in stats.files else 3
    classifier_weight = state_dict.get("classifier.weight")
    num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 2
    model = RivetGNN(
        node_features=data.num_node_features,
        edge_features=data.edge_attr.size(1),
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    if inference_mode == "window":
        data = data.cpu()
        window_cache = build_window_cache(data)
        windows = [
            build_window_graph(data, center_idx, window_hop, window_cache)
            for center_idx in range(data.num_nodes)
        ]
        loader = DataLoader(windows, batch_size=inference_batch_size, shuffle=False)
        pred = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                target_log_probabilities = out[batch.target_mask]
                pred.extend(target_log_probabilities.argmax(dim=1).cpu().tolist())
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


def read_predictions_csv(prediction_path):
    pred_labels = []
    with Path(prediction_path).open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            face_id = int(row["face_id"])
            while len(pred_labels) < face_id:
                pred_labels.append(0)
            pred_labels[face_id - 1] = int(row["pred_label"])
    return pred_labels


def visualize_cad_results(
    step_path,
    pred_labels,
    context_transparency=0.82,
    marker_radius=0.0,
):
    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_TOC_RGB
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt
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
    prediction_colors = {
        1: Quantity_Color(0.0, 0.8, 0.0, Quantity_TOC_RGB),
        2: Quantity_Color(0.1, 0.3, 1.0, Quantity_TOC_RGB),
        3: Quantity_Color(0.0, 0.8, 0.8, Quantity_TOC_RGB),
    }
    print(f"Displaying raw model predictions: {step_path}")
    print("green=predicted rivet, blue=predicted decal, cyan=predicted window")
    print("transparent gray=predicted background/model context")
    print(f"transparent gray=model context (transparency={context_transparency:g})")
    display.DisplayShape(
        shape,
        color=Quantity_NOC_GRAY,
        transparency=context_transparency,
        update=False,
    )

    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    num_faces = face_map.Size()
    print(f"Model faces: {num_faces}, Predicted labels: {len(pred_labels)}")

    model_box = Bnd_Box()
    brepbndlib.Add(shape, model_box)
    x_min, y_min, z_min, x_max, y_max, z_max = model_box.Get()
    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5
    auto_marker_radius = min(max(diagonal * 0.001, 2.0), 12.0)
    use_markers = marker_radius and marker_radius > 0
    if use_markers:
        print(f"Marker radius: {marker_radius:g}")
    else:
        print(f"Marker radius: disabled (auto suggestion: {auto_marker_radius:g})")

    label_counts = {}
    for label in pred_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")

    highlight_markers = []
    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        label = pred_labels[i - 1] if (i - 1) < len(pred_labels) else 0
        if label == 0:
            continue
        color = prediction_colors.get(label, Quantity_NOC_GRAY)
        display.DisplayShape(face, color=color, update=False)

        if use_markers:
            face_box = Bnd_Box()
            brepbndlib.Add(face, face_box)
            fx_min, fy_min, fz_min, fx_max, fy_max, fz_max = face_box.Get()
            center = gp_Pnt(
                (fx_min + fx_max) * 0.5,
                (fy_min + fy_max) * 0.5,
                (fz_min + fz_max) * 0.5,
            )
            highlight_markers.append((center, color))

    for center, color in highlight_markers:
        marker = BRepPrimAPI_MakeSphere(center, marker_radius).Shape()
        display.DisplayShape(marker, color=color, update=False)

    display.FitAll()
    start_display()


def parse_args():
    parser = argparse.ArgumentParser(description="Run RivetGNN inference and visualize face labels on a STEP model.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help=f"Detector executable path. Default: {DEFAULT_DETECTOR}")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INFERENCE_CSV, help=f"Inference CSV path. Default: {DEFAULT_INFERENCE_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help=f"Trained model path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats path. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--pred-in", type=Path, help="Read an existing prediction CSV instead of running model inference.")
    parser.add_argument("--pred-out", type=Path, help="Optional prediction CSV output path.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension used during training. Default: 64")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing inference CSV instead of calling Detector.")
    parser.add_argument("--no-display", action="store_true", help="Only export predictions; do not open the OCC viewer.")
    parser.add_argument("--context-transparency", type=float, default=0.82, help="Transparency for the full-model context. Default: 0.82")
    parser.add_argument("--marker-radius", type=float, default=0.0, help="Overlay marker sphere radius. Use 0 to disable markers. Default: 0")
    parser.add_argument("--inference-mode", choices=["full", "window"], default="full", help="Run full-graph inference or per-face k-hop window inference. Default: full")
    parser.add_argument("--window-hop", type=int, default=2, help="k-hop radius for window inference. Default: 2")
    parser.add_argument("--inference-batch-size", type=int, default=1, help="Window inference batch size. Keep 1 because batched global pooling changes window predictions. Default: 1")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pred_in:
        predicted_labels = read_predictions_csv(args.pred_in)
        print(f"Loaded predictions: {args.pred_in}")
    else:
        if not args.skip_export:
            export_inference_csv(args.step_model, args.detector, args.csv)

        predicted_labels, _ = run_inference(
            args.csv,
            args.model,
            args.stats,
            args.hidden_dim,
            inference_mode=args.inference_mode,
            window_hop=args.window_hop,
            inference_batch_size=args.inference_batch_size,
        )
        print("Inference finished.")

    if args.pred_out:
        write_predictions_csv(args.pred_out, predicted_labels)

    if not args.no_display:
        visualize_cad_results(
            args.step_model,
            predicted_labels,
            context_transparency=args.context_transparency,
            marker_radius=args.marker_radius,
        )


if __name__ == "__main__":
    main()

