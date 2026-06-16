# I2SB Orthogonal Channel-Mean Lowpass Closure

Date: 2026-06-16

## Status

`closed_negative_structure_unstable`

## Run

- Config:
  `configs/aaai2027/phase2_i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010.json`
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Curve:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_chmeanlow_k070_e3_fast10_curve.csv`
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010/`

## Transfer Curve Read

| epoch | transfer CLIP-S | transfer LPIPS | read |
|---|---:|---:|---|
| e1 | `0.697062` | `0.482289` | style below band, structure loose |
| e2 | `0.702899` | `0.513291` | style recovers but structure explodes |
| e5 | `0.701429` | `0.410212` | target-facing but worse than low-anchor0.50 e9 |
| e6 | `0.690869` | `0.448003` | collapse confirmation |

## Decision

Do not promote `channel_mean` lowpass anchoring.

The idea preserved low-frequency style/color too aggressively. It confirms the
diagnosis that low-frequency style cannot be released globally without a harder
base/fiber separation during the actual SDE step.

## Next

Switch from training-time endpoint parameterization changes to eval-only
Orthogonal Fiber-SDE projection: project the predicted endpoint and Brownian
noise into the high-frequency fiber subspace during `solver_i2sb` inference.
