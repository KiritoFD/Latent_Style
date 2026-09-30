# Actuation Delta Basis S030 Probe

Date: 2026-06-15

## Goal

Run a Seedream-oriented style-first follow-up after the `style_delta_scale=0.15`
lane showed weak positive actuation but plateaued around transfer
`CLIP-S=0.674`.

This lane keeps the same mechanism and parent, then changes only
`style_delta_scale` from `0.15` to `0.30`.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config:
  `configs/aaai2027/phase2_actuation_delta_basis_s030_k070_e3_b32bf16_vlen010.json`
- Only mechanism/parameter delta from S015:
  `model.style_delta_scale=0.30`.
- Unchanged: `style_delta_rank=4`, `style_delta_hidden_dim=64`,
  `style_delta_force_highpass=true`, tokenizer, solver, losses, TopoGate,
  appearance alignment, batch/eval contract, and `freeze_mode=injection_only`.
- Logging fix: `style_delta_*` observability columns are present in the training
  CSV before launch.

## Decision Rule

- Primary metric: transfer `CLIP-S`.
- Structure metric: transfer `LPIPS` is a budget, not the ranking target.
- Seedream-oriented tolerance: do not stop merely because LPIPS exceeds the old
  0.31 structure-preserving target. Treat `0.35` as an in-band reference, and
  keep watching if transfer CLIP-S is still improving without visual/metric
  collapse.
- Convergence read: evaluate every epoch and close only when the transfer
  CLIP-S curve is flat/regressing for multiple retained checkpoints, or when
  LPIPS rises without style gain.

## Launch Log

- 2026-06-15 16:31 remote WSL formal run started.
- PID: `1328`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_delta_basis_s030_k070_e3_b32bf16_vlen010`.
- Remote log:
  `logs/phase2_actuation_delta_basis_s030_k070_e3_b32bf16_vlen010.launch.log`.
- Fresh save dir was created; this lane does not resume from the S015 e5
  checkpoint and does not reuse S015 optimizer state.
- 30s health check: active training, GPU sample about `8795 MiB`, `94%`
  utilization, `134 W`.
- CSV header check: `style_delta_basis_active`,
  `style_delta_basis_rank`, `style_delta_basis_abs`,
  `style_delta_weight_abs`, `style_delta_side_abs`,
  `style_delta_side_rms`, and `style_delta_scale` are present.
- 2026-06-15 eval speed audit: old formal eval used
  `batch=1,target_chunk=2,decode=4` and spent about `200-230s` per checkpoint.
  Timing breakdown showed generation and VAE decode dominate:
  `lancet_generation ~=105-109s`, `vae_decode ~=54s`, metrics loop only
  `~=21s`.
- Fast eval smoke on `epoch_0009.pt`:
  `batch=4,target_chunk=5,decode=20` completed in `102.09s`
  (`lancet_generation=16.60s`, `vae_decode=51.14s`) with the same
  LPIPS+CLIP-S metric surface.
- Larger smoke `batch=8,target_chunk=5,decode=40` was slower (`108.46s`) due
  to VAE decode cost, so the formal continuation uses the `batch=4` profile.
- From the continuation after `epoch_0009`, full eval is switched to
  `full_eval_batch_size=4`, `full_eval_target_chunk_size=5`,
  `full_eval_vae_decode_batch_size=20`.
- 2026-06-15 eval optimization follow-up: added
  `full_eval_transfer_only=true` for training-time convergence eval. This skips
  identity pairs only; per-transfer generated samples and CLIP-S/LPIPS formulas
  are unchanged. Final closure must still run a full-board eval if
  `style minus IDT` is needed.
- After restart from `epoch_0013.pt`, `epoch_0014` transfer-only eval completed
  in `86.66s` (`lancet_generation=15.11s`, `vae_decode=41.07s`), versus
  `~102s` for the prior batched full-board profile and `~203-207s` for the
  original profile.

## Running Eval Curve

Curve CSV:
`docs/experiments/phase2_fiber_bundle/eval/actuation_delta_basis_s030_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | style minus IDT | eval wall | transfer-only |
|---|---:|---:|---:|---:|---:|
| 1 | 0.672659 | 0.333805 | -0.150195 | 203.04s | no |
| 2 | 0.674053 | 0.346618 | -0.145910 | 206.40s | no |
| 3 | 0.673827 | 0.346089 | -0.145851 | 206.72s | no |
| 4 | 0.674136 | 0.350179 | -0.144175 | 207.28s | no |
| 5 | 0.673752 | 0.348534 | -0.145378 | 201.79s | no |
| 6 | 0.673968 | 0.349402 | -0.144596 | 204.28s | no |
| 7 | 0.674119 | 0.350776 | -0.144325 | 200.35s | no |
| 8 | 0.673917 | 0.350305 | -0.144514 | 204.18s | no |
| 9 | 0.673790 | 0.349841 | -0.144800 | 207.17s | no |
| 10 | 0.674079 | 0.351139 | -0.144023 | 103.13s | no |
| 11 | 0.673980 | 0.351319 | -0.143997 | 102.23s | no |
| 12 | 0.674017 | 0.352630 | -0.143231 | 101.81s | no |
| 13 | 0.674097 | 0.353023 | -0.143039 | 102.31s | no |
| 14 | 0.673868 | 0.353363 | n/a | 86.66s | yes |
| 15 | 0.673968 | 0.351537 | n/a | 86.62s | yes |
| 16 | 0.674081 | 0.353087 | n/a | 86.92s | yes |
| 17 | 0.674263 | 0.353245 | n/a | 86.54s | yes |
| 18 | 0.674194 | 0.352582 | n/a | 86.74s | yes |
| 19 | 0.674306 | 0.353175 | n/a | 86.65s | yes |
| 20 | 0.674338 | 0.353857 | n/a | 86.57s | yes |

Closure full-board e20 eval:
`docs/experiments/phase2_fiber_bundle/eval/actuation_delta_basis_s030_k070_e3_b32bf16_vlen010/fullboard_epoch_0020/summary.json`

| checkpoint | transfer CLIP-S | transfer LPIPS | IDT CLIP-S | IDT LPIPS | style minus IDT | eval wall |
|---|---:|---:|---:|---:|---:|---:|
| e20 full-board | 0.674200 | 0.353881 | 0.816896 | 0.342996 | -0.142696 | 110.46s |

Training observability:

| epoch | style_delta_weight_abs | style_delta_side_abs | style_delta_side_rms |
|---|---:|---:|---:|
| 1 | 0.136375 | 0.005921 | 0.008003 |
| 2 | 0.664804 | 0.031740 | 0.041855 |
| 3 | 0.794532 | 0.038386 | 0.050010 |

Interim read: e1 is not better than S015 e1, but the actuator magnitude grows
substantially by the e2 training row. e2 transfer CLIP-S surpasses the best S015
point (`0.674053` vs `0.673966`). e4 remains the current best full-board point at
`0.674136 / 0.350179`; e17-e20 produce tiny transfer-only style gains, ending at
`0.674338 / 0.353857`, but the closure full-board e20 read is only
`0.674200 / 0.353881` with `style_minus_idt=-0.142696`. This confirms that
raising style_delta_scale from `0.15` to `0.30` mostly increases structure cost
and does not break the style actuation bottleneck.

## Closure Decision

Closed negative/weak-positive. Keep the implementation switch because it is
clean and observable, but do not promote S030. Next controlled test should
change fiber actuation capacity, not just scale: increase `style_delta_rank`
while keeping parent, scale, highpass, tokenizer, solver, loss, eval, and
freeze mode fixed.
