import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CSV = PROJECT_ROOT / "data" / "current_inference.csv"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Release" / "Detector.exe"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "work" / "rivet_gnn_xian20_train_simpletest_50ep.pth"
DEFAULT_STATS_PATH = PROJECT_ROOT / "work" / "rivet_gnn_xian20_train_simpletest_50ep_stats.npz"
LEGACY_INFERENCE_MODE = "full"
LEGACY_WINDOW_HOP = 2


def optional_stats_value(stats, key, default=None):
    available = stats.files if hasattr(stats, "files") else stats
    if key not in available:
        return default
    value = stats[key]
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def infer_hidden_dim(state_dict, num_layers):
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        classifier_weight = state_dict.get("rivet_classifier.weight")
    if classifier_weight is None:
        raise ValueError("Checkpoint is missing classifier or dual-head weights.")
    classifier_input = int(classifier_weight.shape[1])
    if num_layers == 3:
        return classifier_input
    if classifier_input % 3 != 0:
        raise ValueError(
            "Cannot infer hidden_dim from classifier.weight: "
            f"input_features={classifier_input}, num_layers={num_layers}"
        )
    return classifier_input // 3


def resolve_inference_contract(
    stats,
    requested_mode=None,
    requested_window_hop=None,
    requested_hidden_dim=None,
    inferred_hidden_dim=None,
    allow_override=False,
):
    saved_mode = optional_stats_value(stats, "inference_mode")
    saved_training_mode = optional_stats_value(stats, "training_mode")
    if saved_mode is None and saved_training_mode is not None:
        saved_mode = "window" if saved_training_mode == "window" else "full"
    saved_window_hop = optional_stats_value(stats, "window_hop")
    saved_hidden_dim = optional_stats_value(stats, "hidden_dim")
    checks = [
        ("inference mode", requested_mode, saved_mode),
        ("window hop", requested_window_hop, saved_window_hop),
        ("hidden dimension", requested_hidden_dim, saved_hidden_dim),
    ]
    mismatches = [
        f"{name}: requested={requested}, checkpoint={saved}"
        for name, requested, saved in checks
        if requested is not None and saved is not None and requested != saved
    ]
    if mismatches and not allow_override:
        raise ValueError(
            "Inference arguments do not match the checkpoint contract: "
            + "; ".join(mismatches)
            + ". Use --allow-contract-override only for a controlled experiment."
        )

    mode = next(
        value for value in (requested_mode, saved_mode, LEGACY_INFERENCE_MODE)
        if value is not None
    )
    window_hop = int(next(
        value for value in (requested_window_hop, saved_window_hop, LEGACY_WINDOW_HOP)
        if value is not None
    ))
    hidden_dim = int(next(
        value for value in (requested_hidden_dim, saved_hidden_dim, inferred_hidden_dim, 64)
        if value is not None
    ))
    if mode not in {"full", "window"}:
        raise ValueError(f"Unsupported inference mode in checkpoint stats: {mode}")
    if window_hop <= 0:
        raise ValueError(f"window_hop must be positive, got {window_hop}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    return mode, window_hop, hidden_dim

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


def run_inference(
    csv_path,
    model_path,
    stats_path,
    hidden_dim=None,
    inference_mode=None,
    window_hop=None,
    inference_batch_size=1,
    allow_contract_override=False,
    rivet_threshold_override=None,
    surface_threshold_override=None,
):
    import torch
    from torch_geometric.loader import DataLoader
    from train_rivet_gcn import (
        DualHeadRivetGNN,
        RivetGNN,
        build_window_cache,
        build_window_graph,
        dual_head_decision,
        apply_smooth_shell_surface_guard,
        label_names_from_stats,
        load_cad_data,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    stats = np.load(stats_path, allow_pickle=True)
    num_layers = int(stats["num_layers"]) if "num_layers" in stats.files else 3
    model_architecture = str(optional_stats_value(stats, "model_architecture", "three-class"))
    classifier_weight = state_dict.get("classifier.weight")
    num_classes = 3 if model_architecture == "dual-head" else (
        int(classifier_weight.shape[0]) if classifier_weight is not None else 2
    )
    label_names = label_names_from_stats(stats, num_classes)
    inference_mode, window_hop, hidden_dim = resolve_inference_contract(
        stats,
        requested_mode=inference_mode,
        requested_window_hop=window_hop,
        requested_hidden_dim=hidden_dim,
        inferred_hidden_dim=infer_hidden_dim(state_dict, num_layers),
        allow_override=allow_contract_override,
    )
    data = load_cad_data(csv_path, stats_path=stats_path)
    model_kwargs = {
        "node_features": data.num_node_features,
        "edge_features": data.edge_attr.size(1),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    model = (
        DualHeadRivetGNN(**model_kwargs)
        if model_architecture == "dual-head"
        else RivetGNN(**model_kwargs, num_classes=num_classes)
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    face_ids = data.face_id.cpu().tolist()
    probabilities = []
    predictions = []
    rivet_threshold = float(optional_stats_value(stats, "rivet_threshold", 0.5))
    surface_threshold = float(optional_stats_value(stats, "surface_threshold", 0.5))
    if rivet_threshold_override is not None or surface_threshold_override is not None:
        if not allow_contract_override:
            raise ValueError(
                "Decision-threshold overrides require --allow-contract-override."
            )
        rivet_threshold = (
            float(rivet_threshold_override)
            if rivet_threshold_override is not None
            else rivet_threshold
        )
        surface_threshold = (
            float(surface_threshold_override)
            if surface_threshold_override is not None
            else surface_threshold
        )
    if not 0.0 < rivet_threshold < 1.0:
        raise ValueError("rivet_threshold must be between 0 and 1.")
    if not 0.0 < surface_threshold < 1.0:
        raise ValueError("surface_threshold must be between 0 and 1.")
    print(
        "Inference contract: "
        f"mode={inference_mode}, window_hop={window_hop}, hidden_dim={hidden_dim}, "
        f"num_layers={num_layers}, architecture={model_architecture}"
    )
    feature_cols = {str(value) for value in stats["feature_cols"].tolist()}
    smooth_shell_guard = bool(
        optional_stats_value(stats, "smooth_shell_surface_guard", False)
    )
    smooth_shell_guard_min_component_area = float(
        optional_stats_value(stats, "smooth_shell_guard_min_component_area", 0.25)
    )
    smooth_shell_guard_min_face_area_ratio = float(
        optional_stats_value(stats, "smooth_shell_guard_min_face_area_ratio", 0.4)
    )

    def collect_output(output, target_mask, graph):
        target_output = output[target_mask]
        if model_architecture == "dual-head":
            target_predictions, head_probabilities = dual_head_decision(
                target_output,
                rivet_threshold=rivet_threshold,
                surface_threshold=surface_threshold,
            )
            if smooth_shell_guard:
                target_predictions = apply_smooth_shell_surface_guard(
                    target_predictions,
                    graph.smooth_component_normalized_area[target_mask],
                    graph.smooth_component_face_count[target_mask],
                    graph.smooth_component_face_area_ratio[target_mask],
                    minimum_component_area=smooth_shell_guard_min_component_area,
                    minimum_face_area_ratio=smooth_shell_guard_min_face_area_ratio,
                )
            rivet_probability = head_probabilities[:, 0]
            surface_probability = head_probabilities[:, 1]
            background_probability = (1.0 - rivet_probability) * (1.0 - surface_probability)
            output_probabilities = torch.stack(
                [background_probability, rivet_probability, surface_probability], dim=1
            )
        else:
            output_probabilities = target_output.exp()
            target_predictions = output_probabilities.argmax(dim=1)
        predictions.extend(target_predictions.cpu().tolist())
        probabilities.extend(output_probabilities.cpu().tolist())

    if inference_mode == "window":
        data = data.cpu()
        window_cache = build_window_cache(data)
        windows = [
            build_window_graph(data, center_idx, window_hop, window_cache)
            for center_idx in range(data.num_nodes)
        ]
        loader = DataLoader(windows, batch_size=inference_batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                collect_output(model(batch), batch.target_mask, batch)
    else:
        with torch.no_grad():
            output = model(data.to(device))
            collect_output(
                output,
                torch.ones(output.shape[0], dtype=torch.bool, device=device),
                data.to(device),
            )
    probability_array = np.asarray(probabilities, dtype=float)
    pred = predictions
    if len(face_ids) != len(pred):
        raise RuntimeError(
            f"Prediction/face ID count mismatch: predictions={len(pred)}, face_ids={len(face_ids)}"
        )
    return pred, label_names, face_ids, probability_array


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
    use_markers = marker_radius and marker_radius > 0
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

    display.FitAll()
    if screenshot_path:
        if not display.View.Dump(str(screenshot_path)):
            raise RuntimeError(f"Unable to write visualization screenshot: {screenshot_path}")
        print(f"Visualization screenshot written: {screenshot_path}")
    else:
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
    parser.add_argument("--truth-csv", type=Path, help="Optional labeled CSV used to color correct predictions and errors.")
    parser.add_argument("--truth-model-name", help="Exact model_name in --truth-csv. Required with --truth-csv.")
    parser.add_argument("--hidden-dim", type=int, help="Override checkpoint hidden dimension.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing inference CSV instead of calling Detector.")
    parser.add_argument("--no-display", action="store_true", help="Only export predictions; do not open the OCC viewer.")
    parser.add_argument("--screenshot", type=Path, help="Write an OCC PNG screenshot instead of opening the interactive viewer.")
    parser.add_argument("--context-transparency", type=float, default=0.82, help="Transparency for the full-model context. Default: 0.82")
    parser.add_argument("--marker-radius", type=float, default=0.0, help="Overlay marker sphere radius. Use 0 to disable markers. Default: 0")
    parser.add_argument("--inference-mode", choices=["full", "window"], help="Override checkpoint inference mode. Legacy v7 defaults to window.")
    parser.add_argument("--window-hop", type=int, help="Override checkpoint window hop. Legacy v7 defaults to 2.")
    parser.add_argument("--inference-batch-size", type=int, default=1, help="Window inference batch size. Keep 1 because batched global pooling changes window predictions. Default: 1")
    parser.add_argument("--allow-contract-override", action="store_true", help="Allow explicit inference arguments to override checkpoint metadata.")
    parser.add_argument("--rivet-threshold", type=float, help="Override dual-head rivet threshold; requires --allow-contract-override.")
    parser.add_argument("--surface-threshold", type=float, help="Override dual-head surface threshold; requires --allow-contract-override.")
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
            args.csv,
            args.model,
            args.stats,
            args.hidden_dim,
            inference_mode=args.inference_mode,
            window_hop=args.window_hop,
            inference_batch_size=args.inference_batch_size,
            allow_contract_override=args.allow_contract_override,
            rivet_threshold_override=args.rivet_threshold,
            surface_threshold_override=args.surface_threshold,
        )
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
        )


if __name__ == "__main__":
    main()

