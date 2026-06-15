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

## Running Eval Curve

Pending.

## Closure Decision

Pending.
