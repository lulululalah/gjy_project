import argparse
import subprocess
from pathlib import Path

import torch
from OCC.Core.Quantity import Quantity_NOC_GRAY, Quantity_NOC_GREEN, Quantity_NOC_RED
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Display.SimpleGui import init_display

from train_rivet_gcn import DEFAULT_MODEL_PATH, DEFAULT_STATS_PATH, RivetGNN, load_cad_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CSV = PROJECT_ROOT / "data" / "current_inference.csv"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Release" / "Detector.exe"

LABEL_COLORS = {
    0: None,
    1: Quantity_NOC_GREEN,
    2: Quantity_NOC_RED,
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


def run_inference(csv_path, model_path, stats_path, hidden_dim):
    data = load_cad_data(csv_path, stats_path=stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RivetGNN(
        node_features=data.num_node_features,
        edge_features=1,
        hidden_dim=hidden_dim,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        out = model(data.to(device))
        pred = out.argmax(dim=1).cpu().tolist()

    return pred


def visualize_cad_results(step_path, pred_labels):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    display, start_display, _, _ = init_display()
    print(f"Displaying model: {step_path}")

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

    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        label = pred_labels[i - 1] if (i - 1) < len(pred_labels) else 0
        color = LABEL_COLORS.get(label)

        if color is None:
            display.DisplayShape(face, color=Quantity_NOC_GRAY, update=False)
        else:
            display.DisplayShape(face, color=color, update=False)

    display.FitAll()
    start_display()


def parse_args():
    parser = argparse.ArgumentParser(description="Run RivetGNN inference and visualize face labels on a STEP model.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help=f"Detector executable path. Default: {DEFAULT_DETECTOR}")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INFERENCE_CSV, help=f"Inference CSV path. Default: {DEFAULT_INFERENCE_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help=f"Trained model path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats path. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension used during training. Default: 64")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing inference CSV instead of calling Detector.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_export:
        export_inference_csv(args.step_model, args.detector, args.csv)

    predicted_labels = run_inference(args.csv, args.model, args.stats, args.hidden_dim)
    print("Inference finished.")
    visualize_cad_results(args.step_model, predicted_labels)


if __name__ == "__main__":
    main()
