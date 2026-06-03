# Distinct5 Same-Family Path-Stability Probe

Date: 2026-06-03

This directory mirrors the retained local copy of the matched same-family
Distinct5 `H`-family path-stability probe.

## Packet scope

- dataset:
  - `Distinct5-512`
- family:
  - current `H` packet
- matched checkpoint policy:
  - `epoch_0001` for `H_base`, `H_k025`, and `H_k000`
- rollout mode:
  - `field`
- split emphasized in the paper:
  - `transfer`

## Retained files

- `summary.json`
- `run_summary.csv`
- `per_time_stats.csv`
- `fig_velocity_over_time.pdf`

## Main read

The probe closes the path-stability packet at the narrow mechanism level.
Under matched `epoch_0001` checkpoints, weakening or removing kinetic
regularization sharply increases transfer-direction executed motion:

- `H_base`
  - endpoint/path/peak = `80.24 / 80.28 / 80.41`
- `H_k025`
  - endpoint/path/peak = `111.71 / 111.72 / 111.83`
- `H_k000`
  - endpoint/path/peak = `122.42 / 122.40 / 122.51`

The same-family training/eval rows show the corresponding quality cost:

- `H_base`
  - `clip_style = 0.6891`
  - `content_lpips = 0.4272`
- `H_k025`
  - `clip_style = 0.6825`
  - `content_lpips = 0.4600`
- `H_k000`
  - `clip_style = 0.6790`
  - `content_lpips = 0.4862`

This packet therefore supports only the bounded paper claim:

- kinetic regularization acts as a practical path stabilizer in the current
  OMF / field regime on the same-family Distinct5 packet;
- it does not justify a broader theorem or cross-family generalization claim.
