# Codebase Guide

This file is a compact map for day-to-day development.

## Runtime Entry Points

- Train: `src/run.py`
- Trainer logic: `src/trainer.py`
- Full evaluation: `src/utils/run_evaluation.py`
- Inference helper: `src/utils/inference.py`

## Model / Loss Boundaries

- `src/model.py`
  - network structure
  - style injection paths
  - integration/inference stepping
- `src/losses.py`
  - objective composition
  - style/content balancing terms
  - train-time sampling for step size and style strength

## Config Ownership

- `src/config.json`: main large-run config
- `src/overfit50.json`: small-data debug config

Recommended practice:
- keep one config per experiment family
- avoid ad-hoc edits in historical configs
- prefer copying config and assigning new `checkpoint.save_dir`

## Output Ownership

Each run should write under its own `checkpoint.save_dir`:
- `epoch_XXXX.pt`
- `logs/training_*.csv`
- `full_eval/epoch_XXXX/summary.json`

## Scripts

- `run_full_eval_epochs.sh`: evaluate selected checkpoints
- `scripts/style_ablation.py`: style-focused ablation automation (core)
- `scripts/style_ablation.sh`: thin wrapper for bash users

## Hygiene Rules

- Do not commit generated run artifacts.
- Keep docs synchronized when adding new loss/model switches.
- When adding config keys, wire them through docs + script templates together.
