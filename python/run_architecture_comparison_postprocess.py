"""Evaluate archived architecture-comparison checkpoints after CAD guards.

The script keeps model inference separate from CAD post-processing.  It writes
raw predictions only as intermediate audit artifacts, then applies the same
planar-patch rivet guard and surface-visibility guard recorded by the formal
nine-aircraft evaluation snapshot.  The reported summary is computed solely
from the final guarded prediction CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rivet_topology_guard import suppress_planar_patch_rivet_predictions
from surface_visibility_guard import postprocess_prediction_csv
from summarize_evaluation_snapshot import main as summarize_snapshot_main


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_model_name(name: str) -> str:
    for suffix in ("_decals",):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def feature_paths(feature_dir: Path) -> dict[str, Path]:
    paths = {}
    for path in feature_dir.glob("*.features.csv"):
        stem = path.name[: -len(".features.csv")]
        key = canonical_model_name(stem)
        if key in paths:
            raise ValueError(f"Duplicate feature CSVs for {key}: {paths[key]}, {path}")
        paths[key] = path
    return paths


def step_path(step_dir: Path, model_name: str) -> Path:
    candidates = [model_name, f"{model_name}_decals"]
    matches = [
        path for path in step_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
        and path.stem in candidates
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one STEP model for {model_name!r}; found {matches}."
        )
    return matches[0]


def raw_predictions(step_path: Path, feature_path: Path, checkpoint: Path, stats_path: Path):
    """Return direct classifier outputs, without any heuristic or CAD guard."""
    import torch
    from torch_geometric.loader import DataLoader
    from train_rivet_gcn import (
        DualHeadRivetGNN,
        RivetGNN,
        build_window_cache,
        build_window_graph,
        dual_head_decision,
        label_names_from_stats,
        load_cad_data,
    )
    from visualize_rivets import (
        infer_hidden_dim,
        optional_stats_value,
        resolve_inference_contract,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint, map_location=device)
    stats = np.load(stats_path, allow_pickle=True)
    num_layers = int(stats["num_layers"]) if "num_layers" in stats.files else 3
    architecture = str(optional_stats_value(stats, "model_architecture", "three-class"))
    classifier_weight = state_dict.get("classifier.weight")
    num_classes = 3 if architecture == "dual-head" else int(classifier_weight.shape[0])
    inference_mode, window_hop, hidden_dim = resolve_inference_contract(
        stats,
        inferred_hidden_dim=infer_hidden_dim(state_dict, num_layers),
    )
    data = load_cad_data(feature_path, stats_path=stats_path)
    model_kwargs = {
        "node_features": data.num_node_features,
        "edge_features": data.edge_attr.size(1),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    model = (
        DualHeadRivetGNN(**model_kwargs)
        if architecture == "dual-head"
        else RivetGNN(**model_kwargs, num_classes=num_classes)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    rivet_threshold = float(optional_stats_value(stats, "rivet_threshold", 0.5))
    surface_threshold = float(optional_stats_value(stats, "surface_threshold", 0.5))
    predictions, probabilities = [], []

    def collect(output, mask):
        target = output[mask]
        if architecture == "dual-head":
            labels, head_probabilities = dual_head_decision(
                target,
                rivet_threshold=rivet_threshold,
                surface_threshold=surface_threshold,
            )
            rivet_probability = head_probabilities[:, 0]
            surface_probability = head_probabilities[:, 1]
            output_probabilities = torch.stack(
                [
                    (1.0 - rivet_probability) * (1.0 - surface_probability),
                    rivet_probability,
                    surface_probability,
                ],
                dim=1,
            )
        else:
            output_probabilities = target.exp()
            labels = output_probabilities.argmax(dim=1)
        predictions.extend(labels.cpu().tolist())
        probabilities.extend(output_probabilities.cpu().tolist())

    with torch.no_grad():
        if inference_mode == "full":
            output = model(data.to(device))
            collect(output, torch.ones(output.shape[0], dtype=torch.bool, device=device))
        else:
            data = data.cpu()
            cache = build_window_cache(data)
            windows = [
                build_window_graph(data, center_idx, window_hop, cache)
                for center_idx in range(data.num_nodes)
            ]
            for batch in DataLoader(windows, batch_size=1, shuffle=False):
                batch = batch.to(device)
                collect(model(batch), batch.target_mask)
    face_ids = data.face_id.cpu().tolist()
    if len(face_ids) != len(predictions):
        raise RuntimeError("Classifier output and face identifiers do not align.")
    labels = label_names_from_stats(stats, 3)
    return face_ids, predictions, np.asarray(probabilities), labels


def write_prediction(path: Path, face_ids, labels, probabilities, label_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "face_id", "pred_label", "pred_name", "pred_confidence",
            "prob_background", "prob_rivet", "prob_surface_feature",
        ])
        for face_id, label, probability in zip(face_ids, labels, probabilities):
            writer.writerow([
                int(face_id), int(label), label_names[int(label)], float(probability[int(label)]),
                *[float(value) for value in probability],
            ])


def summarize(prediction_dir: Path, truth_dir: Path, output_dir: Path) -> None:
    original = sys.argv
    try:
        sys.argv = [
            "summarize_evaluation_snapshot.py",
            "--truth-dir", str(truth_dir),
            "--prediction-dir", str(prediction_dir),
            "--output-dir", str(output_dir),
        ]
        summarize_snapshot_main()
    finally:
        sys.argv = original


def run_metrics(summary_path: Path) -> dict[str, float]:
    rows = pd.read_csv(summary_path.parent / "face_level_metrics.csv").set_index("class")
    risk = pd.read_csv(summary_path.parent / "per_aircraft_risk.csv")
    return {
        "rivet_f1": float(rows.at["Rivet", "f1"]),
        "surface_feature_f1": float(rows.at["Surface-feature", "f1"]),
        "foreground_macro_f1": float(rows.loc[["Rivet", "Surface-feature"], "f1"].mean()),
        "miou": float(rows["iou"].mean()),
        "background_fp": float(risk["total_background_false_positives"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path,
                        default=PROJECT_ROOT / "results/e1_detection/current_test_snapshot/manifest.json")
    parser.add_argument("--feature-dir", type=Path,
                        default=PROJECT_ROOT / "model/predictions/postprocess_current_20260909")
    parser.add_argument("--step-dir", type=Path,
                        default=PROJECT_ROOT / "data/plane_model/after_two")
    parser.add_argument("--truth-dir", type=Path,
                        default=PROJECT_ROOT / "model/evaluation/test_eval_models")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir.resolve()
    output_dir = (args.output_dir or comparison_dir / "postprocessed").resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    experiment = json.loads((comparison_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
    model_names = [item["model"] for item in snapshot["models"]]
    features = feature_paths(args.feature_dir)
    missing = [name for name in model_names if name not in features]
    if missing:
        raise FileNotFoundError(f"No feature CSV for snapshot model(s): {missing}")
    output_dir.mkdir(parents=True)
    all_metrics, run_records = [], []

    for run in experiment["runs"]:
        run_id = f"{run['architecture']}_h{run['hidden_dim']}_seed{run['seed']}"
        checkpoint = PROJECT_ROOT / run["checkpoint"] if "checkpoint" in run else comparison_dir / f"{run_id}.pth"
        stats = PROJECT_ROOT / run["stats"] if "stats_root_relative" in run else comparison_dir / run["stats"]
        if not checkpoint.is_file() or not stats.is_file():
            raise FileNotFoundError(f"Missing checkpoint/statistics pair for {run_id}")
        run_dir = output_dir / run_id
        raw_dir, rivet_dir, final_dir = run_dir / "raw", run_dir / "rivet_guard", run_dir / "predictions"
        adjustment_rows = []
        print(f"\n=== {run_id} ===", flush=True)
        for model_name in model_names:
            feature_path = features[model_name]
            source_step = step_path(args.step_dir, model_name)
            output_name = feature_path.name.replace(".features.csv", ".pred.csv")
            raw_path, rivet_path, final_path = raw_dir / output_name, rivet_dir / output_name, final_dir / output_name
            print(f"  {model_name}", flush=True)
            face_ids, labels, probabilities, label_names = raw_predictions(
                source_step, feature_path, checkpoint, stats
            )
            write_prediction(raw_path, face_ids, labels, probabilities, label_names)
            rivet_guarded, rivet_ids = suppress_planar_patch_rivet_predictions(
                pd.read_csv(raw_path), pd.read_csv(feature_path)
            )
            rivet_path.parent.mkdir(parents=True, exist_ok=True)
            rivet_guarded.to_csv(rivet_path, index=False)
            visibility_diagnostics = postprocess_prediction_csv(
                source_step, feature_path, rivet_path, final_path
            )
            adjustment_rows.append({
                "model": model_name,
                "rivet_topology_suppressed": len(rivet_ids),
                "surface_visibility_or_density_suppressed": len(visibility_diagnostics),
            })
        metrics_dir = run_dir / "metrics"
        summarize(final_dir, args.truth_dir, metrics_dir)
        metrics = run_metrics(metrics_dir / "summary.json")
        metrics.update({"architecture": run["architecture"], "seed": run["seed"], "run_id": run_id})
        all_metrics.append(metrics)
        pd.DataFrame(adjustment_rows).to_csv(run_dir / "postprocess_adjustments.csv", index=False)
        run_records.append({
            **run,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "postprocessed_prediction_dir": str(final_dir.relative_to(comparison_dir)),
            "metrics_dir": str(metrics_dir.relative_to(comparison_dir)),
        })

    run_frame = pd.DataFrame(all_metrics)
    run_frame.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summaries = []
    for architecture, group in run_frame.groupby("architecture", sort=False):
        first = next(item for item in experiment["runs"] if item["architecture"] == architecture)
        row = {
            "architecture": architecture,
            "hidden_dim": first["hidden_dim"],
            "parameters": first["parameters"],
            "seeds": ";".join(str(seed) for seed in group["seed"].tolist()),
        }
        for metric in ("rivet_f1", "surface_feature_f1", "foreground_macro_f1", "miou", "background_fp"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(comparison_dir / "summary.csv", index=False, float_format="%.6f")
    provenance = {
        "postprocess": ["rivet_topology_guard", "surface_visibility_guard"],
        "snapshot_manifest": str(args.snapshot_manifest.relative_to(PROJECT_ROOT)),
        "feature_dir": str(args.feature_dir.relative_to(PROJECT_ROOT)),
        "step_dir": str(args.step_dir.relative_to(PROJECT_ROOT)),
        "truth_dir": str(args.truth_dir.relative_to(PROJECT_ROOT)),
        "runs": run_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
