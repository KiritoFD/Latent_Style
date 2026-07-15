# Active Submission Scripts

Only two launchers are active:

- `run_submission_repro.ps1`: train from `config.json`, then evaluate every checkpoint with `inference.json`.
- `batch_eval_all.py`: evaluate selected or all epochs with CLIP-S, LPIPS, DINO-S, and DINO-C.

Both scripts resolve paths from the project root and contain no machine-specific dataset paths. Historical launchers are preserved under `archives/legacy-scripts-20260715/` and must not be used for submission experiments.
