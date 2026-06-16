# Eval-Only Orthogonal Fiber-SDE Projection Plan

Date: 2026-06-16

## Goal

Test the hard-projection hypothesis without retraining. Use the current best
target-facing endpoint-trained checkpoint and restrict both the predicted I2SB
endpoint and Brownian noise to the high-frequency fiber subspace during
inference.

## Parent Checkpoint

- Checkpoint:
  `exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Baseline transfer:
  `0.701429 / 0.372203`
- Reason:
  This is the current best target-facing point. It is endpoint-trained, unlike
  the velocity-only k070 parent, so its endpoint head can be safely used by the
  I2SB posterior.

## Mechanism

- `i2sb_fiber_project_endpoint=true`
- `i2sb_fiber_project_noise=true`
- `i2sb_fiber_project_kernel=5`
- `num_steps=8`
- `save_generated_images=false`
- transfer-only fast10 eval

The endpoint projection is:

`x1_projected = lowpass(source) + highpass(x1_pred)`

The noise projection is:

`fiber_noise = noise - lowpass(noise)`

## Sigma Scan

| config | sigma |
|---|---:|
| `phase2_eval_fiber_project_sigma0p5_lowanchor050e9.json` | `0.5` |
| `phase2_eval_fiber_project_sigma1p0_lowanchor050e9.json` | `1.0` |
| `phase2_eval_fiber_project_sigma1p5_lowanchor050e9.json` | `1.5` |

## Decision Rule

- Positive:
  any sigma improves style over `0.701429` without LPIPS exceeding `0.40`, or
  reaches LPIPS `<=0.35` with style at least `0.700`.
- Strong positive:
  any sigma reaches style `>=0.72` with LPIPS `<=0.35`.
- Negative:
  high-frequency noise does not improve style, or hard projection destroys
  endpoint style while only improving LPIPS.

## Artifact Targets

- Eval mirrors:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_lowanchor050e9_sigma*/`
- Consolidated CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_lowanchor050e9_sigma_scan.csv`

## Launch Log

- Pending remote WSL eval-only launch.
