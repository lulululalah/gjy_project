"""Frozen-bundle prediction, aligned metrics and reproducibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BUNDLE = ROOT / "model/final_detection_model"

LABEL_NAMES = ("background", "rivet", "surface_feature")

def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_model(name: str) -> str:
    stem = Path(name).name
    for suffix in (".csv", ".features", ".pred", ".step", ".stp", ".STEP", ".STP", "_decals"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem

def validate_datasets(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required = {"graph_id", "model_name", "id", "label"}
    for role, frame in (("train", train), ("test", test)):
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{role} data is empty or missing identity/label columns")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{role} data has missing identity/label values")
        if frame.duplicated(["model_name", "id"]).any():
            raise ValueError(f"{role} data has duplicate model/face identifiers")
        if not frame.label.isin(range(3)).all():
            raise ValueError(f"{role} labels must be background=0, rivet=1, surface_feature=2")
        if (frame.id < 1).any() or not (frame.id == frame.id.astype(int)).all():
            raise ValueError(f"{role} face identifiers must be positive integers")
        if (frame.groupby("graph_id").model_name.nunique() != 1).any():
            raise ValueError(f"{role} graph identifiers must each refer to one aircraft")
        if (frame.groupby("model_name").graph_id.nunique() != 1).any():
            raise ValueError(f"{role} aircraft must each have one complete graph")
        canonical = frame.model_name.drop_duplicates().map(canonical_model)
        if canonical.duplicated().any():
            raise ValueError(f"{role} contains ambiguous aircraft filenames")
    if set(train.model_name.map(canonical_model)) & set(test.model_name.map(canonical_model)):
        raise ValueError("Training and test aircraft overlap")

def validate_predictions(test: pd.DataFrame, predictions: pd.DataFrame, recovery_threshold: float | None = None) -> None:
    required = {"model_name", "face_id", "true_label", "pred_label"}
    if not required.issubset(predictions.columns):
        raise ValueError("Predictions are missing model/face/truth/prediction columns")
    if predictions.duplicated(["model_name", "face_id"]).any():
        raise ValueError("Predictions contain duplicate model/face identifiers")
    truth = test.rename(columns={"id": "face_id"}).set_index(["model_name", "face_id"]).sort_index()
    output = predictions.set_index(["model_name", "face_id"]).sort_index()
    if not truth.index.equals(output.index):
        raise ValueError("Prediction model/face identifiers do not match the test data")
    if not np.array_equal(truth.label.to_numpy(), output.true_label.to_numpy()):
        raise ValueError("Prediction truth labels differ from the authoritative test data")
    for column in ("pred_label", "raw_label"):
        if column in output and not output[column].isin(range(3)).all():
            raise ValueError(f"{column} contains invalid class labels")
    for column in ("prob_background", "prob_rivet", "prob_surface_feature"):
        if column in output and not output[column].between(0, 1).all():
            raise ValueError(f"{column} contains invalid scores")
    if "raw_label" in output:
        allowed = (output.pred_label == 0) | (output.pred_label == output.raw_label)
        if recovery_threshold is not None:
            if not {"recovery_candidate", "recovery_score", "surface_recovered"}.issubset(output.columns):
                raise ValueError("Recovery changes must have complete provenance columns")
            for column in ("recovery_candidate", "surface_recovered"):
                if not output[column].isin([True, False]).all():
                    raise ValueError(f"Invalid recovery flag: {column}")
            if not output.recovery_score.between(0, 1).all():
                raise ValueError("Invalid recovery scores")
            recovered = (output.raw_label == 0) & (output.pred_label == 2)
            if not np.array_equal(recovered.to_numpy(), output.surface_recovered.astype(bool).to_numpy()):
                raise ValueError("Recovered-face flags disagree with labels")
            allowed |= recovered & output.recovery_candidate.astype(bool) & (output.recovery_score >= recovery_threshold)
        if not allowed.all():
            raise ValueError("Screening changed a label into another target class")

def compute_metrics(truth, predictions):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    if truth.shape != predictions.shape or not np.isin(truth, range(3)).all() or not np.isin(predictions, range(3)).all():
        raise ValueError("Metrics require aligned three-class labels")
    confusion = np.bincount(3 * truth + predictions, minlength=9).reshape(3, 3)
    rows = []
    for label, name in enumerate(LABEL_NAMES):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        fp, fn = int(confusion[:, label].sum()) - tp, support - tp
        rows.append({"class_id": label, "class_name": name, "support": support,
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / support if support else 0.0,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                     "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0})
    summary = {"faces": len(truth), "confusion": confusion.tolist(),
               "accuracy": float(np.trace(confusion) / len(truth)) if len(truth) else 0.0,
               "rivet_f1": rows[1]["f1"], "surface_feature_f1": rows[2]["f1"],
               "foreground_macro_f1": (rows[1]["f1"] + rows[2]["f1"]) / 2,
               "background_fp": int(confusion[0, 1:].sum())}
    return rows, summary

def load_bundle(bundle: Path):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["input_sha256"].items():
        path = bundle / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundle input is missing or changed: {relative}")
    train, test = pd.read_csv(bundle / manifest["train_csv"]), pd.read_csv(bundle / manifest["test_csv"])
    validate_datasets(train, test)
    if len(train) != manifest["train_faces"] or len(test) != manifest["test_faces"]:
        raise ValueError("Bundle face counts disagree with the manifest")
    if set(manifest["geometry"]) != set(test.model_name.unique()):
        raise ValueError("Geometry inventory differs from test aircraft")
    for entry in manifest["geometry"].values():
        path = (bundle / entry["path"]).resolve()
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Geometry is missing or changed: {path}")
    return manifest, train, test

def screen_predictions(raw: pd.DataFrame, features: pd.DataFrame, step: Path, cache: Path, recovery=None):
    from recognition.postprocess import apply_base_guards
    from recognition.postprocess import suppress_planar_patch_rivet_predictions
    from recognition.postprocess import apply_exterior_visibility_surface_guard, suppress_low_confidence_dense_surface_predictions

    candidates = recovery.apply(raw, features) if recovery is not None else raw
    result, _ = apply_base_guards(candidates, features)
    result, _ = suppress_planar_patch_rivet_predictions(result, features)
    labels, _ = apply_exterior_visibility_surface_guard(
        step, result.pred_label.tolist(), result.face_id.tolist(), features,
        exposure_cache_dir=cache,
    )
    labels, _, _ = suppress_low_confidence_dense_surface_predictions(labels, result.prob_surface_feature.to_numpy())
    result["pred_label"] = labels
    result["pred_name"] = [LABEL_NAMES[int(label)] for label in labels]
    suppressed = result.pred_label == 0
    result.loc[suppressed, "pred_confidence"] = result.loc[suppressed, "prob_background"]
    if recovery is not None:
        recovered = (raw.pred_label == 0) & (result.pred_label == 2)
        result["surface_recovered"] = recovered
        result.loc[recovered, "pred_confidence"] = result.loc[recovered, "recovery_score"]
    return result

def recovery_threshold(bundle, manifest):
    entry = manifest.get("surface_recovery", {})
    if not entry.get("enabled", False):
        return None
    configuration = json.loads((bundle / entry["configuration"]).read_text(encoding="utf-8"))
    if not configuration["enabled"]:
        raise ValueError("Manifest enables a recovery classifier that failed calibration")
    if configuration["train_sha256"] != manifest["input_sha256"][manifest["train_csv"]]:
        raise ValueError("Recovery classifier was not fitted to this bundle's training data")
    if configuration.get("checkpoint_sha256") != manifest["input_sha256"][manifest["checkpoint"]]:
        raise ValueError("Recovery gates were not calibrated with this bundle's checkpoint")
    return float(configuration["minimum_score"])

def create_bundle_model(stats, manifest):

    architecture = str(stats['model_architecture'].item())
    if architecture != manifest['model_architecture']:
        raise ValueError('Model architecture metadata differs from manifest')
    if str(stats['encoder_sharing'].item()) != manifest['encoder_sharing']:
        raise ValueError('Model sharing metadata differs from manifest')
    if str(stats['inference_mode'].item()) != 'full':
        raise ValueError('The published bundle requires full-graph inference')
    kwargs = dict(node_features=len(stats['feature_cols']), edge_features=len(stats['edge_attr_cols']),
                  hidden_dim=int(stats['hidden_dim']), num_layers=int(stats['num_layers']))
    if architecture == 'three-class-mlp':
        from recognition.model import ThreeClassGNN
        if manifest['encoder_sharing'] != 'shared' or manifest.get('decision') != 'argmax':
            raise ValueError('Categorical bundle requires one shared encoder and argmax')
        return ThreeClassGNN(adapter_dim=int(stats['adapter_dim']), dropout=float(stats['dropout']), **kwargs)
    raise ValueError(f'Unsupported bundle architecture: {architecture}')

def bundle_decision(logits, stats):
    if str(stats['model_architecture'].item()) != 'three-class-mlp':
        raise ValueError('Only the deployed single-three-class architecture is supported')
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError('Categorical model requires three output logits')
    probabilities = logits.softmax(dim=1)
    return probabilities.argmax(dim=1), probabilities

def predict_bundle(bundle: Path, manifest, test, cache: Path):
    import torch
    from recognition.graph import build_graph
    from recognition.postprocess import _load_step_faces

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = recovery_threshold(bundle, manifest)
    recovery = None
    if threshold is not None:
        entry = manifest["surface_recovery"]
        if entry.get("method") == "dc10-window-topology":
            from recognition.recovery import DC10WindowRecovery
            geometry = {name: (bundle / item["path"]).resolve() for name, item in manifest["geometry"].items()}
            recovery = DC10WindowRecovery(bundle / entry["configuration"], geometry)
        else:
            raise ValueError("Unsupported recovery method for the deployed bundle")
    with np.load(bundle / manifest["stats"], allow_pickle=True) as stats:
        model = create_bundle_model(stats, manifest).to(device)
        model.load_state_dict(torch.load(bundle / manifest["checkpoint"], map_location=device, weights_only=True), strict=True)
        model.eval()
        output = []
        for _, group in test.groupby("graph_id", sort=True):
            group = group.reset_index(drop=True)
            name = str(group.model_name.iloc[0])
            print(f"Predicting {name}: {len(group)} faces", flush=True)
            step = (bundle / manifest["geometry"][name]["path"]).resolve()
            _, face_map = _load_step_faces(step)
            if sorted(group.id.astype(int).tolist()) != list(range(1, face_map.Size() + 1)):
                raise ValueError(f"STEP face identifiers do not match test CSV: {name}")
            graph = build_graph(group, stats["mean"], stats["std"], stats["edge_mean"], stats["edge_std"], stats["feature_cols"].tolist()).to(device)
            with torch.no_grad():
                labels, scores = bundle_decision(model(graph), stats)
            labels = labels.cpu().numpy()
            probabilities = scores.cpu().numpy()
            raw = pd.DataFrame({"face_id": group.id.astype(int), "pred_label": labels,
                        "pred_name": [LABEL_NAMES[int(label)] for label in labels],
                        "pred_confidence": probabilities[np.arange(len(labels)), labels],
                        "prob_background": probabilities[:, 0], "prob_rivet": probabilities[:, 1], "prob_surface_feature": probabilities[:, 2]})
            final = screen_predictions(raw, group, step, cache, recovery=recovery)
            final.insert(0, "model_name", name)
            final.insert(1, "graph_id", int(group.graph_id.iloc[0]))
            final["true_label"] = group.label.astype(int)
            final["raw_label"] = raw.pred_label
            output.append(final)
    predictions = pd.concat(output, ignore_index=True)
    validate_predictions(test, predictions, recovery_threshold=threshold)
    return predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, help="Must not exist; defaults to bundle/evaluation")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "work/recognition_geometry_cache")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest, train, test = load_bundle(bundle)
    output = (args.output_dir or bundle / "evaluation").resolve()
    if args.verify_only:
        predictions = pd.read_csv(output / "predictions.csv", low_memory=False)
        validate_predictions(test, predictions, recovery_threshold=recovery_threshold(bundle, manifest))
        _, computed = compute_metrics(predictions.true_label, predictions.pred_label)
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        for key in ("rivet_f1", "surface_feature_f1", "foreground_macro_f1", "background_fp", "faces", "confusion"):
            if saved[key] != computed[key]:
                raise ValueError(f"Saved metrics disagree with predictions: {key}")
        if saved["input_sha256"] != manifest["input_sha256"]:
            raise ValueError("Prediction input fingerprint differs from the selected bundle")
        for relative, expected in saved["output_sha256"].items():
            path = output / relative
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Prediction artifact is missing or changed: {relative}")
        print(json.dumps({"status": "PASS", "test_faces": len(test), "model": manifest["model_id"], **computed}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to mix with existing results: {output}; use --verify-only or a new --output-dir")
    predictions = predict_bundle(bundle, manifest, test, args.cache_dir)
    rows, summary = compute_metrics(predictions.true_label, predictions.pred_label)
    _, raw_summary = compute_metrics(predictions.true_label, predictions.raw_label)
    summary.update(model_id=manifest["model_id"], encoder_sharing=manifest["encoder_sharing"],
                   input_sha256=manifest["input_sha256"], raw=raw_summary)
    summary["pipeline_id"] = manifest.get("pipeline_id", manifest["model_id"])
    if "surface_recovered" in predictions:
        summary["recovery_candidates"] = int(predictions.recovery_candidate.sum())
        summary["surface_recovered"] = int(predictions.surface_recovered.sum())
    output.mkdir(parents=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "face_level_metrics.csv", index=False)
    per_aircraft = []
    for name, group in predictions.groupby("model_name", sort=True):
        stem = canonical_model(name)
        for folder in ("inputs", "predictions", "raw"):
            (output / folder).mkdir(exist_ok=True)
        test.loc[test.model_name == name].to_csv(output / "inputs" / f"{stem}.features.csv", index=False)
        columns = ["face_id", "pred_label", "pred_name", "pred_confidence", "prob_background", "prob_rivet", "prob_surface_feature"]
        diagnostic_columns = [c for c in ("raw_label", "recovery_candidate", "recovery_score", "surface_recovered",
                              "recovery_reason", "recovery_host_face_id", "recovery_mirror_face_id") if c in group]
        group[columns + diagnostic_columns].to_csv(output / "predictions" / f"{stem}.pred.csv", index=False)
        raw_group = group[columns].copy()
        raw_group["pred_label"] = group.raw_label
        raw_group["pred_name"] = [LABEL_NAMES[int(label)] for label in group.raw_label]
        probabilities = raw_group[["prob_background", "prob_rivet", "prob_surface_feature"]].to_numpy()
        raw_group["pred_confidence"] = probabilities[np.arange(len(group)), group.raw_label.to_numpy()]
        raw_group.to_csv(output / "raw" / f"{stem}.pred.csv", index=False)
        class_metrics, metrics = compute_metrics(group.true_label, group.pred_label)
        per_aircraft.append({"model_name": name,
                             "rivet_support": class_metrics[1]["support"],
                             "rivet_missed": class_metrics[1]["fn"],
                             "surface_feature_support": class_metrics[2]["support"],
                             "surface_feature_missed": class_metrics[2]["fn"],
                             "surface_recovered": int(group.surface_recovered.sum()) if "surface_recovered" in group else 0,
                             **{k: v for k, v in metrics.items() if k != "confusion"}})
    pd.DataFrame(per_aircraft).to_csv(output / "per_aircraft_metrics.csv", index=False)
    summary["output_sha256"] = {str(path.relative_to(output)): sha256(path)
                                for path in sorted(output.rglob("*.csv"))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
