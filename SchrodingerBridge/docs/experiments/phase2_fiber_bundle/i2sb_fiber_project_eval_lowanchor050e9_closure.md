# Eval-Only Orthogonal Fiber-SDE Projection Closure

Date: 2026-06-16

## Status

`closed_negative_current_latent_highpass`

## Parent

- Checkpoint:
  `exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Baseline:
  `0.701429 / 0.372203`

## Results

| sigma | transfer CLIP-S | transfer LPIPS | delta vs baseline |
|---:|---:|---:|---|
| `0.0` | `0.687711` | `0.358167` | `-0.013718` style, `-0.014036` LPIPS |
| `0.5` | `0.703560` | `0.592224` | `+0.002131` style, `+0.220021` LPIPS |
| `1.0` | `0.676268` | `0.684136` | `-0.025161` style, `+0.311934` LPIPS |
| `1.5` | `0.673637` | `0.708145` | `-0.027791` style, `+0.335943` LPIPS |

## Observability

All eval summaries report:

- `i2sb_fiber_project_endpoint_active=1`
- `i2sb_fiber_project_noise_active=1`
- `i2sb_fiber_project_kernel=5`

So this is a valid negative result for the implemented hard projector, not a
missed-switch artifact.

## Decision

Do not promote the naive latent avg-pool Orthogonal Fiber-SDE projector.

The result falsifies the direct assumption that latent avg-pool lowpass is a
safe proxy for decoded-image base geometry. The `sigma=0.0` diagnostic shows
the endpoint projector can lower LPIPS but suppresses style. The high-noise
scan shows highpass latent Brownian motion does not create useful brushstroke
style in the current decoder path; it mainly increases decoded LPIPS.

## Next

Keep the switch for reproducibility, default off. The next hard-projection
variant should not use raw latent avg-pool as the base projector. Candidate
directions:

- project only after decoding or through a decoder-aware latent Jacobian proxy;
- use tokenizer/topogate masks to restrict where the highpass projector acts;
- use lower sigma with stochastic corrector on a stronger style checkpoint
  such as latent-slerp e2, but only as a matched diagnostic.
