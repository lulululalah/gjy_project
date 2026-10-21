"""Inspect the retained three-class predictions and truth on native CAD faces."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INFERENCE_CSV = PROJECT_ROOT / "data" / "current_inference.csv"

DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Release" / "Detector.exe"

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "model" / "final_detection_model"
    / "model.pth"
)

DEFAULT_STATS_PATH = (
    PROJECT_ROOT / "model" / "final_detection_model"
    / "stats.npz"
)


def export_inference_csv(step_path, detector_path, csv_path):
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector executable not found: {detector_path}")

    output_csv = DEFAULT_INFERENCE_CSV
    previous_mtime = output_csv.stat().st_mtime_ns if output_csv.exists() else 0
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

    if not output_csv.exists() or output_csv.stat().st_mtime_ns <= previous_mtime:
        raise RuntimeError(
            "Detector completed without producing a fresh inference CSV: "
            f"{output_csv}"
        )
    csv_path = Path(csv_path)
    if csv_path.resolve() != output_csv.resolve():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_csv, csv_path)

def run_inference(step_path, csv_path, model_path, stats_path):
    import json
    import torch
    from recognition.graph import load_cad_data
    from recognition.predict import LABEL_NAMES, create_bundle_model, bundle_decision, screen_predictions, sha256
    from recognition.recovery import DC10WindowRecovery

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(stats_path, allow_pickle=True) as stats:
        model = create_bundle_model(stats, dict(model_architecture='three-class-mlp', encoder_sharing='shared', decision='argmax')).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
        model.eval()
        data = load_cad_data(csv_path, stats_path=stats_path)
        face_ids = data.face_id.cpu().tolist()
        with torch.no_grad():
            labels, scores = bundle_decision(model(data.to(device)), stats)
        probabilities = scores.cpu().numpy()
        labels = labels.cpu().numpy()
    features = pd.read_csv(csv_path)
    if features.id.astype(int).tolist() != face_ids:
        raise ValueError('Feature and prediction face identifiers do not match')
    raw = pd.DataFrame(dict(face_id=face_ids, pred_label=labels,
        pred_name=[LABEL_NAMES[int(k)] for k in labels],
        pred_confidence=probabilities[np.arange(len(labels)), labels],
        prob_background=probabilities[:, 0], prob_rivet=probabilities[:, 1], prob_surface_feature=probabilities[:, 2]))
    profile = Path(stats_path).parent/'recognition.recovery.json'
    recovery = None
    if profile.is_file():
        configuration = json.loads(profile.read_text(encoding='utf-8'))
        if configuration.get('checkpoint_sha256') != sha256(Path(model_path)):
            raise ValueError('Geometry recovery configuration and checkpoint differ')
        if 'model_name' not in features:
            features['model_name'] = Path(step_path).name
        recovery = DC10WindowRecovery(profile, {str(name): Path(step_path) for name in features.model_name.unique()})
    final = screen_predictions(raw, features, Path(step_path), PROJECT_ROOT/'work/recognition_geometry_cache', recovery=recovery)
    return final.pred_label.tolist(), list(LABEL_NAMES), face_ids, probabilities

def write_predictions_csv(prediction_path, face_ids, pred_labels, probabilities=None, label_names=None):
    prediction_path = Path(prediction_path)
    face_ids = list(face_ids)
    pred_labels = list(pred_labels)
    if len(face_ids) != len(pred_labels):
        raise ValueError(
            f"Prediction/face ID count mismatch: predictions={len(pred_labels)}, face_ids={len(face_ids)}"
        )
    if len(set(face_ids)) != len(face_ids):
        raise ValueError("Prediction face IDs must be unique.")
    probability_array = None if probabilities is None else np.asarray(probabilities, dtype=float)
    if probability_array is not None and probability_array.shape != (len(pred_labels), len(label_names or [])):
        raise ValueError(
            "Probability matrix shape does not match predictions and label names: "
            f"shape={probability_array.shape}, predictions={len(pred_labels)}, labels={len(label_names or [])}"
        )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        probability_headers = [f"prob_{name}" for name in (label_names or [])]
        writer.writerow(["face_id", "pred_label", "pred_name", "pred_confidence", *probability_headers])
        for row_index, (face_id, label) in enumerate(zip(face_ids, pred_labels)):
            label_name = label_names[label] if label_names else str(label)
            if probability_array is None:
                writer.writerow([face_id, label, label_name, ""])
                continue
            row_probabilities = probability_array[row_index]
            writer.writerow(
                [face_id, label, label_name, float(row_probabilities[label]), *row_probabilities.tolist()]
            )
    print(f"Prediction CSV ready: {prediction_path}")

def read_predictions_csv(prediction_path):
    face_ids = []
    pred_labels = []
    with Path(prediction_path).open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            face_ids.append(int(row["face_id"]))
            pred_labels.append(int(row["pred_label"]))
    if not face_ids:
        raise ValueError(f"Prediction CSV is empty: {prediction_path}")
    if len(set(face_ids)) != len(face_ids):
        raise ValueError("Prediction CSV contains duplicate face IDs.")
    return face_ids, pred_labels

def read_truth_csv(truth_path, model_name):
    """Read ordered training labels for one complete STEP model."""
    labels_by_id = {}
    with Path(truth_path).open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if str(row.get("model_name")) != model_name:
                continue
            labels_by_id[int(row["id"])] = int(row["label"])
    if not labels_by_id:
        raise ValueError(f"No rows for {model_name!r} in truth CSV: {truth_path}")
    max_id = max(labels_by_id)
    missing = [face_id for face_id in range(1, max_id + 1) if face_id not in labels_by_id]
    if missing:
        raise ValueError(f"Truth CSV is missing face IDs for {model_name}: {missing[:10]}")
    return [labels_by_id[face_id] for face_id in range(1, max_id + 1)]

def visualize_cad_results(
    step_path,
    pred_labels,
    label_names=None,
    truth_labels=None,
    context_transparency=0.82,
    marker_radius=0.0,
    pred_face_ids=None,
    screenshot_path=None,
    simple_viewer=False,
    foreground_only=False,
):
    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_TOC_RGB
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise RuntimeError(f"Unable to read STEP file: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    root = None
    canvas = None
    selected_face_text = None
    if screenshot_path or simple_viewer:
        from OCC.Display.SimpleGui import init_display

        display, start_display, _, _ = init_display()
    else:
        import tkinter as tk
        from OCC.Display.tkDisplay import tkViewer3d

        root = tk.Tk()
        root.title(f"Prediction/truth comparison: {Path(step_path).name}")
        selected_face_text = tk.StringVar(value="Right-click a face to show its F-number here.")
        status = tk.Label(root, textvariable=selected_face_text, anchor="w", padx=8)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        canvas = tkViewer3d(root)
        canvas.pack(fill=tk.BOTH, expand=True)
        root.update_idletasks()
        root.update()
        display = canvas._display
        start_display = root.mainloop
    prediction_colors = {
        1: Quantity_Color(0.0, 0.8, 0.0, Quantity_TOC_RGB),
        2: Quantity_Color(0.1, 0.3, 1.0, Quantity_TOC_RGB),
    }
    error_colors = {
        (0, 1): Quantity_Color(0.75, 0.1, 0.85, Quantity_TOC_RGB),
        (0, 2): Quantity_Color(1.0, 0.05, 0.05, Quantity_TOC_RGB),
        (1, 0): Quantity_Color(0.75, 0.1, 0.85, Quantity_TOC_RGB),
        (2, 0): Quantity_Color(1.0, 0.9, 0.05, Quantity_TOC_RGB),
        (1, 2): Quantity_Color(0.75, 0.1, 0.85, Quantity_TOC_RGB),
        (2, 1): Quantity_Color(1.0, 0.9, 0.05, Quantity_TOC_RGB),
    }
    if label_names != ["background", "rivet", "surface_feature"]:
        raise ValueError(f"Unsupported label schema for visualization: {label_names}")
    if truth_labels is None:
        print(f"Displaying raw model predictions: {step_path}")
        print("green=predicted rivet, blue=predicted surface feature")
    else:
        print(f"Displaying prediction/truth comparison: {step_path}")
        print("green=correct rivet, blue=correct surface feature")
        print("purple=false-positive rivet, red=false-positive surface feature")
        print("purple=missed/wrong rivet, yellow=missed/wrong surface feature")
    print("transparent gray=predicted background/model context")
    print(f"transparent gray=model context (transparency={context_transparency:g})")
    if not foreground_only:
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
    if pred_face_ids is None:
        pred_face_ids = range(1, len(pred_labels) + 1)
    pred_face_ids = list(pred_face_ids)
    if len(pred_face_ids) != len(pred_labels):
        raise ValueError(
            f"Prediction/face ID count mismatch: predictions={len(pred_labels)}, "
            f"face_ids={len(pred_face_ids)}"
        )
    predictions_by_face_id = dict(zip(pred_face_ids, pred_labels))
    if len(predictions_by_face_id) != len(pred_face_ids):
        raise ValueError("Prediction face IDs must be unique.")
    expected_face_ids = set(range(1, num_faces + 1))
    actual_face_ids = set(predictions_by_face_id)
    if actual_face_ids != expected_face_ids:
        missing = sorted(expected_face_ids - actual_face_ids)
        unexpected = sorted(actual_face_ids - expected_face_ids)
        raise ValueError(
            "Prediction face IDs do not match the STEP face map: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if truth_labels is not None and len(truth_labels) != num_faces:
        raise ValueError(f"Truth labels ({len(truth_labels)}) do not match STEP faces ({num_faces})")

    model_box = Bnd_Box()
    brepbndlib.Add(shape, model_box)
    x_min, y_min, z_min, x_max, y_max, z_max = model_box.Get()
    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5
    auto_marker_radius = min(max(diagonal * 0.001, 2.0), 12.0)
    # Enable enlarged face-center markers when the caller explicitly requests one.
    use_markers = marker_radius > 0
    if use_markers:
        print(f"Marker radius: {marker_radius:g}")
    else:
        print(f"Marker radius: disabled (auto suggestion: {auto_marker_radius:g})")

    label_counts = {}
    for label in pred_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")

    error_counts = {}
    highlight_markers = []
    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        predicted = predictions_by_face_id[i]
        truth = truth_labels[i - 1] if truth_labels is not None else None
        if truth_labels is None:
            if predicted == 0:
                continue
            color = prediction_colors.get(predicted, Quantity_NOC_GRAY)
        elif predicted == truth:
            if predicted == 0:
                continue
            color = prediction_colors.get(predicted, Quantity_NOC_GRAY)
        else:
            color = error_colors[(truth, predicted)]
            error_counts[(truth, predicted)] = error_counts.get((truth, predicted), 0) + 1

        if color is None:
            continue
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

    if truth_labels is not None:
        print(f"Error counts by truth->prediction: {error_counts}")

    def report_selected_face(selected_shapes, x, y):
        if not selected_shapes:
            print(f"No face selected at ({x}, {y}); click directly on a visible surface.")
            return

        for selected in selected_shapes:
            face_id = face_map.FindIndex(selected)
            if not face_id:
                for candidate_id in range(1, num_faces + 1):
                    candidate = face_map.FindKey(candidate_id)
                    if selected.IsSame(candidate) or selected.IsEqual(candidate):
                        face_id = candidate_id
                        break
            if not face_id:
                continue

            message = f"Selected face: F{face_id}"
            if selected_face_text is not None:
                selected_face_text.set(message)
            print(message)
            return

        print("Selected shape did not match a STEP face; click directly on a visible surface.")

    display.SetSelectionModeFace()
    display.register_select_callback(report_selected_face)
    if canvas is not None:

        def select_face(event):
            display.MoveTo(event.x, event.y)
            display.Select(event.x, event.y)

        canvas.bind("<Button-3>", select_face)
        print("Right-click a face to display its F-number in the status bar.")

    display.FitAll()
    display.Repaint()
    if root is not None:
        root.update_idletasks()
        root.after(200, lambda: (display.FitAll(), display.Repaint()))
    if screenshot_path:
        if not display.View.Dump(str(screenshot_path)):
            raise RuntimeError(f"Unable to write visualization screenshot: {screenshot_path}")
        print(f"Visualization screenshot written: {screenshot_path}")
    else:
        start_display()

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect current three-class predictions against STEP face labels.")
    parser.add_argument("step_model", type=Path, help="STEP model to analyze.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help=f"Detector executable path. Default: {DEFAULT_DETECTOR}")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INFERENCE_CSV, help=f"Inference CSV path. Default: {DEFAULT_INFERENCE_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help=f"Trained model path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH, help=f"Normalization stats path. Default: {DEFAULT_STATS_PATH}")
    parser.add_argument("--pred-in", type=Path, help="Read an existing prediction CSV instead of running model inference.")
    parser.add_argument("--pred-out", type=Path, help="Optional prediction CSV output path.")
    parser.add_argument("--truth-csv", type=Path, help="Optional labeled CSV used to color correct predictions and errors.")
    parser.add_argument("--truth-model-name", help="Exact model_name in --truth-csv. Required with --truth-csv.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing inference CSV instead of calling Detector.")
    parser.add_argument("--no-display", action="store_true", help="Only export predictions; do not open the OCC viewer.")
    parser.add_argument("--screenshot", type=Path, help="Write an OCC PNG screenshot instead of opening the interactive viewer.")
    parser.add_argument("--context-transparency", type=float, default=0.0, help="Transparency for the full-model context. Default: 0 (opaque)")
    parser.add_argument("--marker-radius", type=float, default=0.0, help="Overlay marker sphere radius. Use 0 to disable markers. Default: 0")
    parser.add_argument(
        "--simple-viewer",
        action="store_true",
        help="Use the OCC SimpleGui viewer instead of the Tk embedded viewer.",
    )
    parser.add_argument(
        "--foreground-only",
        action="store_true",
        help="Display only predicted/truth foreground faces; omit the full-model context.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    stats = np.load(args.stats, allow_pickle=True)
    if "label_names" not in stats.files:
        raise ValueError("Stats file is missing the required three-class label_names metadata.")
    label_names = [str(value) for value in stats["label_names"].tolist()]

    if args.pred_in:
        prediction_face_ids, predicted_labels = read_predictions_csv(args.pred_in)
        prediction_probabilities = None
        print(f"Loaded predictions: {args.pred_in}")
    else:
        if not args.skip_export:
            export_inference_csv(args.step_model, args.detector, args.csv)

        predicted_labels, label_names, prediction_face_ids, prediction_probabilities = run_inference(
            args.step_model, args.csv, args.model, args.stats)
        print("Inference finished.")


    if (args.truth_csv is None) != (args.truth_model_name is None):
        raise ValueError("Use --truth-csv and --truth-model-name together.")
    truth_labels = None
    if args.truth_csv is not None:
        truth_labels = read_truth_csv(args.truth_csv, args.truth_model_name)

    if args.pred_out:
        write_predictions_csv(
            args.pred_out,
            prediction_face_ids,
            predicted_labels,
            probabilities=prediction_probabilities,
            label_names=label_names,
        )

    if not args.no_display:
        visualize_cad_results(
            args.step_model,
            predicted_labels,
            label_names=label_names,
            truth_labels=truth_labels,
            context_transparency=args.context_transparency,
            marker_radius=args.marker_radius,
            pred_face_ids=prediction_face_ids,
            screenshot_path=args.screenshot,
            simple_viewer=args.simple_viewer,
            foreground_only=args.foreground_only,
        )
