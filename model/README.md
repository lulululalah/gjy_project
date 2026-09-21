# Model artifacts

This directory contains retained model artifacts and model-derived outputs. Runtime geometry-processing intermediates remain under `work`.

## Current formal model

`final_detection_model/` contains the retained checkpoint, paired normalization statistics, evaluation CSV, training/test snapshots, and the predictions generated from this model.

## Model-derived outputs

- `predictions/postprocess_current_20260909/`: current nine-aircraft predictions used by the formal detection snapshot.
- `predictions/post_rivet_predictions/`: predictions after the rivet-removal stage.
- `predictions/test_predictions_all/`: historical prediction outputs.
- `postprocess/`: topology and surface-visibility post-processing experiments.
- `visualizations/`: prediction screenshots and related visual artifacts.
- `evaluation/test_eval_models/`: face-level truth CSVs aligned with the nine formal test STEP models.
- `label_audits/`: retained label-audit and truth-review artifacts.

The checkpoint and its statistics file must be used as a pair. The formal snapshot manifest is maintained under `results/e1_detection/current_test_snapshot/manifest.json`.
