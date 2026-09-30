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
| 4 | 0.674041 | 0.352357 | 93.46s | yes |
| 5 | 0.674215 | 0.353640 | 97.08s | yes |
| 6 | 0.674339 | 0.355674 | 93.79s | yes |
| 7 | 0.674366 | 0.354956 | 93.66s | yes |
| 8 | 0.674252 | 0.355417 | 93.38s | yes |

Curve read: e2 is the best transfer style point (`0.674525 / 0.352222`).
The next six retained checkpoints never exceed e2 and generally pay more
LPIPS. e6/e7 approach the e2 style value but remain below it while moving
LPIPS to `0.355-0.356`. This satisfies the controlled stop condition for a
weak/negative output-side rank expansion: the extra basis capacity is active,
but the generated style residual direction still does not open a new fiber
section.

## Full-Board Closure Eval

Best checkpoint: `epoch_0002`.

Full-board eval root:
`docs/experiments/phase2_fiber_bundle/eval/actuation_delta_basis_r16_s030_k070_e3_b32bf16_vlen010/fullboard_epoch_0002/`

| scope | CLIP-S | LPIPS | rows | wall | style - IDT |
|---|---:|---:|---:|---:|---:|
| transfer | 0.674395 | 0.352223 | 600 | 110.90s | -0.144026 |
| all pairs | 0.703201 | 0.349966 | 750 | 110.90s | -0.115221 |
| IDT | 0.818422 | 0.340940 | 150 | 110.90s | 0.000000 |

Matched-control read:

- Versus S030 full-board closure (`0.674200 / 0.353881`), R16 improves
  transfer CLIP-S by only `+0.000195` and improves LPIPS by `-0.001658`.
- The style gain is far below the `+0.002` decision threshold.
- Runtime/observability confirms the new head was active:
  `style_delta_basis_rank=16`, `style_delta_scale=0.30`,
  `style_delta_side_abs=0.042745`, `style_delta_side_rms=0.055562`.
  The mechanism runs, but its actuation is not enough.

## Closure Decision

Closed rejected / weak negative.

Decision: do not promote R16 and do not continue scaling output-side
`style_delta_rank` or `style_delta_scale` as the next controlled move. The
fiber-bundle diagnosis now points more strongly at where style is injected:
style must enter the internal transport/decoder feature evolution before the
final `dec_out` bottleneck, not as a larger late residual basis alone.

Next controlled experiment: internal `mixed` body+decoder style injection from
the same k070 e3 parent, with output delta basis disabled. This changes the
actuation location while keeping tokenizer, solver, loss, TopoGate, schedule,
dataset, and transfer-only eval contract fixed.
