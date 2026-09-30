# Eval-Only Orthogonal Fiber-SDE Projection On Slerp e2 Closure

Date: 2026-06-16

## Status

`closed_negative_structure_unsafe`

## Parent

- Checkpoint:
  `exp/aaai2027_phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010/epoch_0002.pt`
- Baseline:
  `0.712038 / 0.476511`

## Results

| sigma | transfer CLIP-S | transfer LPIPS | delta vs baseline |
|---:|---:|---:|---|
| `0.0` | `0.693441` | `0.435260` | `-0.018597` style, `-0.041251` LPIPS |
| `0.5` | `0.719065` | `0.568915` | `+0.007027` style, `+0.092404` LPIPS |

## Decision

Do not promote raw latent avg-pool Orthogonal Fiber-SDE projection.

The stronger-style parent shows the same tradeoff: endpoint-only projection
improves structure by suppressing style, while highpass Brownian noise raises
style only by blowing up LPIPS. This closes the naive projector family for now.

## Next

The next hard-projection mechanism must use a better base/fiber split than
latent avg-pool. The cleanest next candidates are:

- mask-aware projection using tokenizer/topogate routing;
- decoder-aware projection/post-decode low-frequency correction as eval-only
  diagnostic;
- train a small projector/head to predict a structure mask, with the main
  backbone frozen.
