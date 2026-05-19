from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from train_hole_candidate_gcn import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    FEATURE_COLS,
    HoleCandidateMLP,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "current_hole_candidates.csv"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "Debug" / "Detector.exe"


def export_candidate_csv(step_path, detector_path=DEFAULT_DETECTOR, csv_path=DEFAULT_INPUT_CSV):
    detector_path = Path(detector_path)
    csv_path = Path(csv_path)

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector executable not found: {detector_path}")

    result = subprocess.run(
        [str(detector_path), "--predict-hole", str(step_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Detector failed to export hole candidates.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"Hole candidate CSV was not generated: {csv_path}")

    return csv_path


def load_stats(stats_path=DEFAULT_STATS_PATH):
    stats = np.load(stats_path, allow_pickle=True)
    return stats["mean"], stats["std"]


def load_candidate_dataframe(csv_path):
    return pd.read_csv(csv_path)


def build_model(hidden_dim, model_path=DEFAULT_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoleCandidateMLP(len(FEATURE_COLS), hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def run_inference(csv_path, model_path=DEFAULT_MODEL_PATH, stats_path=DEFAULT_STATS_PATH, hidden_dim=64):
    df = load_candidate_dataframe(csv_path)
    if df.empty:
        return df

    mean, std = load_stats(stats_path)
    features = df[FEATURE_COLS].astype(float).to_numpy()
    features = (features - mean) / std

    x = torch.tensor(features, dtype=torch.float)
    model, device = build_model(hidden_dim=hidden_dim, model_path=model_path)

    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    result = df.copy()
    result["pred_label"] = probs.argmax(axis=1)
    result["pred_score"] = probs[:, 1]
    return result
