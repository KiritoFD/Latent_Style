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

Pending.

## Closure Decision

Pending.
