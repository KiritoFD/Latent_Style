# SchrodingerBridge

This directory contains the SchrodingerBridge branch of the latent style-transfer project.

## Start Here

- Current working index:
  `docs/aaai2027_working_index_20260602.md`
- Current paper source:
  `aaai_submission/paper_aaai2026.tex`
- Current paper PDF:
  `aaai_submission/paper_aaai2026.pdf`
- Current experiment ledger:
  `docs/experiments/aaai2027_master_experiment_log.csv`
- Current review lane:
  `docs/reviews/README.md`
- Theory / design notes:
  `docs/maths/`

## Important Status Notes

- `aaai_submission/` is the canonical manuscript location.
- `archives/old_paper_workspaces/` contains superseded paper workspaces kept
  only for provenance and recovery.
- Root `config.json` is not the trusted baseline for the current OMF
  conclusions.
- The current paper-facing experiment surface is centered on:
  - `docs/experiments/`
  - `docs/reviews/`
  - `configs/aaai2027/`
  - `exp/aaai2027_*`

## Code Layout

- `src/model.py`: bridge model wrapper
- `src/style_tokenizer.py`: style-side representation module
- `src/lancet_runtime.py`: time-conditioned execution path
- `src/losses.py`: training objectives and regularizers
- `src/ot_cost.py`: SWD computation
- `src/trainer.py`: training loop
- `src/utils/`: evaluation and inference helpers

## Cleanup Rule

The root should contain only active project entry points, stable evidence, and
clearly labeled historical folders. For the current cleanup and retention
policy, start from:

- `docs/experiments/2026-06-03-repo-cleanup-and-archive-pass.md`
- `docs/experiments/2026-06-03-exp-surface-classification.md`
- `docs/cleanup/worktree_triage_20260603.md`
