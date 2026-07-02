# I2SB Slerp + Orthogonal Low/High Plan

Date: 2026-06-16

## Goal

Test the first controlled integration after two partial positives:

- `latent_slerp` gave a matched e2 style/LPIPS gain, but later cooled into
  low-style LPIPS-only points.
- `orthogonal_lowhigh` restrained structure better than content-anchor, but
  lost too much style when used alone.

The hypothesis is that latent-slerp can keep the early style shock while
orthogonal endpoint reconstruction prevents the worst absolute-endpoint
structure drift.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Candidate config:
  `configs/aaai2027/phase2_i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Matched controls:
  `phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010`
  and
  `phase2_i2sb_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  transport, `bridge_sigma=0.02`, exact Brownian bridge schedule, terminal
  SWD, no style delta adapter, no proximal path, b8 accumulation-2, vlen
  `0.10`, and fast10 transfer-only in-loop eval.
- Only integration delta:
  `bridge.bridge_path_mode=latent_slerp` plus
  `model.endpoint_parameterization=orthogonal_lowhigh`.

## Decision Rule

- Primary metric:
  transfer CLIP-S, style-first toward `0.74`.
- LPIPS budget:
  `0.35` is acceptable only if style stays above the clean/SDE style front;
  LPIPS-only improvements below `0.69` style do not count as target progress.
- Positive evidence:
  a checkpoint that beats latent-slerp e2 on LPIPS while keeping comparable
  style, or beats orthogonal-lowhigh e1/e4 style at comparable LPIPS.
- Negative evidence:
  if the curve behaves like orthogonal-lowhigh alone, the endpoint projection
  is still suppressing style; if it behaves like latent-slerp alone, the
  projection is too weak.

## Runtime Observability

Required in summaries:

- `bridge_path_slerp_active=1`.
- `i2sb_endpoint_orthogonal_active=1`.
- `i2sb_endpoint_orthogonal_kernel=5`.
- `i2sb_endpoint_orthogonal_high_scale=1`.
- endpoint/base/final magnitudes.
- fast10 eval wall time.

## Artifacts

- Local curve target:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_slerp_orthogonal_lowhigh_k070_e3_fast10_curve.csv`.
- Local eval mirror target:
  `docs/experiments/phase2_fiber_bundle/eval/i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010/`.

## Launch Log

- 2026-06-16 10:34 first remote launch was stopped after e1 because training
  CSV instrumentation showed `bridge_path_slerp_active=0.0` despite config and
  numeric debug proving `latent_slerp` was active. Root cause: the training log
  column existed, but `append_training_log()` did not write the row-map value
  and therefore filled the column with default `0.0`.
- Infra fix:
  `src/utils/training.py` now writes `bridge_path_slerp_active` into the CSV
  row map. Direct log-writer smoke confirms the value is preserved as `1.0`.
- The invalid first launch directory was removed from remote and is not used as
  evidence.
- 2026-06-16 10:47 formal clean remote WSL launch restarted from the shared
  parent. Parent load confirmed with `loaded=272 skipped=0 missing=0
  unexpected=0`.
- Health check:
  GPU memory about `3086 MiB`, accepted because low memory is not a stop
  condition and GPU utilization is active.

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010/full_eval_fast10/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | read |
|---|---:|---:|---:|---|
| e1 | 0.704828 | 0.446676 | 37.57s | structure better than slerp e1, but style below slerp shock |
| e2 | 0.697412 | 0.419297 | 24.93s | style retreats; not the desired combo behavior |
| e3 | 0.698710 | 0.392849 | 25.91s | LPIPS improves, style remains below 0.70 |
| e4 | 0.694292 | 0.408310 | 24.98s | no style recovery |
| e5 | 0.689088 | 0.396008 | 24.94s | further style decay |
| e6 | 0.688025 | 0.361792 | 24.97s | structure-side point; style has fallen below target band |
| e7 | 0.682186 | 0.368300 | 24.90s | low-style tail |
| e8 | 0.685025 | 0.352807 | 24.92s | LPIPS-side Pareto only |
| e9 | 0.682724 | 0.381770 | 24.94s | no recovery |
| e10 | 0.682408 | 0.368596 | 24.89s | no recovery |
| e11 | 0.683137 | 0.355313 | 25.02s | low-style structure point |
| e12 | 0.683382 | 0.358237 | 25.03s | low-style structure point |
| e13 | 0.679402 | 0.377409 | 24.95s | style cools further |
| e14 | 0.682620 | 0.362356 | 26.51s | no target-facing rebound |
| e15 | 0.678109 | 0.350421 | 25.31s | best LPIPS; style is too low |
| e16 | 0.678957 | 0.371222 | 24.99s | stopped after confirming no style rebound |

Closure decision:

- `closed_negative_style_suppressed_not_promoted`.
- Runtime observability is now clean: training CSV reports
  `bridge_path_slerp_active=1.0`, and summaries report
  `i2sb_endpoint_orthogonal_active=1.0`.
- The desired effect would be to keep latent-slerp e2 style near `0.712` while
  reducing LPIPS. The final e1-e16 curve instead follows the
  orthogonal-lowhigh pattern: LPIPS improves, but style falls into the
  `0.678-0.705` band and never recovers after e1.
- The automatic joint Pareto tracker is misleading here because e15 creates a
  low-style LPIPS-only point. Under the active Seedream/style-first target, this
  is a negative closure, not a promotable frontier update.
- Matched read:
  e1 versus latent-slerp e2 is `-0.007210` CLIP-S and `-0.029835` LPIPS; e15
  versus latent-slerp e28 is `-0.004529` CLIP-S and `-0.002305` LPIPS. The
  combination buys structure by paying style, which is exactly the coupling we
  need to break.
