# Model artifacts

This directory contains the single current three-class model and its complete training and inference assets. Runtime geometry-processing intermediates remain under `work`.

## Current formal model

`final_detection_model/` contains the retained checkpoint, paired normalization statistics, evaluation CSV, training/test snapshots, and the predictions generated from this model.

## Current dependencies

- `final_detection_model/training_assets/`: encoder initialization, train-only auxiliary logits, normalization and identity index required for retraining.
- `final_detection_model/training_record/`: the selected model's actual training history, settings and provenance.
- `final_detection_model/evaluation/`: the only formal predictions and metrics, paired with the retained model.
- Project `python/` and `data/plane_model/after_two/`: executable runtime and required geometry, listed by hash in the formal manifest.

The checkpoint and its statistics file must be used as a pair. The formal manifest is `final_detection_model/manifest.json`; commands are in its README. Obsolete candidate models and experiments were removed by explicit user request; see `work/current_three_class_cleanup_20261019/`. Original truth audits and compact manuscript evidence remain as historical records.
