# I2SB Orthogonal Low-Anchor 0.65 Closure

Date: 2026-06-16

## Status

`closed_negative_over_anchored`

## Run

- Config:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010.json`
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Curve:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor065_k070_e3_fast10_curve.csv`
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010/`

## Transfer Curve Read

| epoch | transfer CLIP-S | transfer LPIPS | read |
|---|---:|---:|---|
| e1 | `0.709417` | `0.449507` | strong style, structure too loose |
| e4 | `0.706564` | `0.395071` | best target-facing point in this run |
| e9 | `0.692446` | `0.358758` | LPIPS improves only after style falls below target band |
| e10 | `0.690043` | `0.362955` | style remains collapsed |
| e11 | `0.687332` | `0.359799` | closure confirmation |

## Matched Decision

- Compared with low-anchor0.50 e9 (`0.701429 / 0.372203`), low-anchor0.65
  does not create a replacement point.
- e4 keeps style healthy but LPIPS is still too high (`0.395071`).
- e9/e11 enter the better LPIPS band, but only after style drops to
  `0.69x/0.68x`.
- This is not evidence against endpoint orthogonal anchoring in general. It is
  evidence that `endpoint_orthogonal_low_anchor=0.65` is too strong for the
  current absolute I2SB parent.

## Decision

Do not promote `lowanchor065`.

Keep low-anchor0.50 e9 as the current target-facing candidate:
`transfer_clip_style=0.701429`, `transfer_content_lpips=0.372203`.

## Next

Scan a milder anchor (`0.55` or `0.58`) from the same parent before trying new
mechanisms. The hypothesis is that `0.50` preserves more style, `0.65` cools
structure too aggressively, and a middle value may move the e7-e9 point closer
to the desired `0.70+ / <=0.36` band.
