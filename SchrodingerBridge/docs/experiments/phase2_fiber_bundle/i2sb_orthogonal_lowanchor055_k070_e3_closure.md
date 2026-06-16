# I2SB Orthogonal Low-Anchor 0.55 Closure

Date: 2026-06-16

## Status

`closed_negative_lpips_only_tail`

## Run

- Config:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010.json`
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Curve:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor055_k070_e3_fast10_curve.csv`
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010/`

## Transfer Curve Read

| epoch | transfer CLIP-S | transfer LPIPS | read |
|---|---:|---:|---|
| e1 | `0.711863` | `0.457232` | strong style impulse |
| e4 | `0.704881` | `0.405001` | best target-facing point in this run |
| e8 | `0.697144` | `0.379475` | near structure band, style below target |
| e11 | `0.688107` | `0.353115` | LPIPS-only tail |
| e12 | `0.689145` | `0.365202` | closure confirmation |

## Matched Decision

- Compared with low-anchor0.50 e9 (`0.701429 / 0.372203`), low-anchor0.55
  does not create a replacement point.
- e4 keeps style healthy, but LPIPS is still too high.
- e8-e12 improve structure only after transfer style has fallen below `0.700`.
- The low-anchor strength sweep now brackets the behavior:
  `0.50` is the current best target-facing point, while `0.55` and `0.65`
  primarily produce LPIPS-only tails.

## Decision

Do not promote `lowanchor055`.

Keep low-anchor0.50 e9 as the current target-facing candidate:
`transfer_clip_style=0.701429`, `transfer_content_lpips=0.372203`.

## Next

Stop scanning scalar lowpass anchor strength for now. The next mechanism should
preserve more low-frequency style/color information while still constraining
structure, rather than increasing or decreasing the same global lowpass anchor.
