# Actuation Delta Basis R16 S030 Probe

Date: 2026-06-15

## Goal

Test the fiber-bundle diagnosis that the model is learning nearly collinear
style residual directions because the output actuation path has insufficient
fiber-section freedom. S030 showed that more scale alone is not enough, so this
lane increases only the output-side style delta basis rank.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config:
  `configs/aaai2027/phase2_actuation_delta_basis_r16_s030_k070_e3_b32bf16_vlen010.json`
- Only mechanism/parameter delta from S030:
  `model.style_delta_rank=16` instead of `4`.
- Held fixed: `style_delta_scale=0.30`, `style_delta_force_highpass=true`,
  tokenizer, solver, losses, TopoGate, appearance alignment, batch,
  transfer-only training eval, and `freeze_mode=injection_only`.

## Decision Rule

- Primary: transfer `CLIP-S`, style-first toward Seedream.
- Budget: transfer `LPIPS <= ~0.36` is acceptable if style is clearly rising.
- If rank16 improves transfer style over S030 by less than `+0.002` after
  convergence, the output-side basis capacity alone is not the missing
  mechanism.
- If early epochs sharply worsen LPIPS without style gain, stop and inspect
  `style_delta_side_abs/rms` before interpreting the result as a theory failure.

## Launch Log

- 2026-06-15 20:37 remote WSL formal run started.
- PID: `3746`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_delta_basis_r16_s030_k070_e3_b32bf16_vlen010`.
- Remote log:
  `logs/phase2_actuation_delta_basis_r16_s030_k070_e3_b32bf16_vlen010.launch.log`.
- 30s health check: active training, GPU sample about `3186 MiB`, `97%`
  utilization, `115 W`.
- Config load check: `style_delta_rank=16`, `style_delta_scale=0.30`,
  `freeze_mode=injection_only`, `full_eval_transfer_only=true`.
- Resume check: loaded same k070 e3 parent with `loaded=282`, `missing=8`.
  The missing keys are expected because the rank16 delta-basis head is new.

## Running Eval Curve

Curve CSV:
`docs/experiments/phase2_fiber_bundle/eval/actuation_delta_basis_r16_s030_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | transfer-only |
|---|---:|---:|---:|---:|
| 1 | 0.674387 | 0.344712 | 93.92s | yes |
| 2 | 0.674525 | 0.352222 | 93.44s | yes |
| 3 | 0.673936 | 0.349550 | 93.56s | yes |

Interim read: e1 is already above the S030 full-board closure point
(`0.674200 / 0.353881`) and above S030 e20 transfer-only style
(`0.674338`) while keeping LPIPS much lower (`0.344712`). This is a useful
early positive for the "fiber section capacity" hypothesis. e2 improves style
again to `0.674525` but with LPIPS rising to `0.352222`; e3 then regresses to
`0.673936 / 0.349550`. Continue to formal convergence; the rank16 signal is
not yet stable enough for promotion or closure.

## Closure Decision

Pending.
