# Vertical Target Round 1

Date: 2026-06-16

## Scope

First `docs/616/design.md` training-side experiment.

- Parent checkpoint:
  `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Fixed:
  - tokenizer `pure_latent_spatial`
  - `solver_i2sb`
  - `bridge_sigma=0.02`
  - endpoint parameterization `orthogonal_lowhigh`
  - endpoint low anchor `0.5`
  - terminal SWD contract
  - fast10 transfer eval contract
- Changed only:
  - training target projection:
    - `training_target_projection_mode=source_low_target_high`
    - `training_target_projection_kernel=5`
    - `training_target_projection_low_anchor=1.0`

## Hypothesis

Current retained line still learns against the raw OT-matched endpoint and relies on endpoint geometry plus solver structure control to pull LPIPS back.

This round tests the cleaner alternative:

- keep the model family fixed
- keep the solver fixed
- keep the parent fixed
- project the training target itself to `source_low + target_high`

Expected signal:

- if the theory is right, early checkpoints should show lower structural drift and a flatter LPIPS curve without collapsing style immediately
- if style collapses sharply while LPIPS barely improves, then the projection is too hard and we should follow with a softer `low_anchor < 1.0` point

## Instrumentation

New runtime metrics added in this round:

- `training_target_projection_active`
- `training_target_projection_mode_source_low_target_high`
- `training_target_projection_low_anchor`
- `training_target_projection_low_drift`
- `training_target_projection_target_delta`
- `training_target_projection_high_energy_ratio`

These are intended to tell us whether the objective itself is now injecting style only through the projected high-frequency branch and how much the projected target departs from the raw OT match.

## Launcher

- Config:
  [phase616_i2sb_vertical_target_a100_lowanchor050e9_b8a2_vlen010.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_i2sb_vertical_target_a100_lowanchor050e9_b8a2_vlen010.json)
- Script:
  [run_phase616_vertical_target_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_vertical_target_round1.sh)

## Status

- `2026-06-16`: config, launcher, and codepath prepared locally.
- Next:
  - sync to remote WSL EXT4 workspace
  - launch the run from EXT4
  - record first-epoch transfer curve and new projection observability
