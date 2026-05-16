# SchrodingerBridge

This directory contains the SchrodingerBridge branch of the latent style-transfer project.

## Start Here

- Current paper:
  `paper_orchestra_workspace/aaai_submission/paper_aaai2026.pdf`
- Project map:
  `PROJECT_OVERVIEW.md`
- Experiment history:
  `EXPERIMENT_LOG.md`
- Theory:
  `maths/`

## Important Status Notes

- `paper_orchestra_workspace/aaai_submission/` is the canonical manuscript location.
- `paper_refine_v2/` is a legacy refinement workspace kept for traceability.
- Root `config.json` is not the trusted baseline for the current OMF conclusions.

## Code Layout

- `src/model.py`: bridge model wrapper
- `src/losses.py`: training objectives and regularizers
- `src/ot_cost.py`: SWD computation
- `src/trainer.py`: training loop
- `src/utils/`: evaluation and inference helpers

## Cleanup Rule

The root should contain only active project entry points, stable evidence, and clearly labeled historical folders. See `DIRECTORY_CLEANUP_LOG.md` for the current cleanup record.
