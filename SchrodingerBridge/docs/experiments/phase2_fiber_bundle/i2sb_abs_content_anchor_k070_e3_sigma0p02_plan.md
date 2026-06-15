# I2SB Absolute + Content Anchor: k070 e3 sigma0p02

## Purpose

Clean follow-up to the `fiber.md` diagnosis after absolute I2SB showed the
strongest style actuation but unacceptable structure drift. This stage keeps
the I2SB absolute endpoint path intact and changes only the loss-side content
topology anchor.

## Controlled Change

- Parent checkpoint:
  `../exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Base config:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate config:
  `configs/aaai2027/phase2_i2sb_abs_content_anchor_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Unchanged:
  `solver_family=solver_i2sb`, `transport_prediction_mode=endpoint`,
  `endpoint_parameterization=absolute`, `bridge.objective_mode=i2sb_endpoint`,
  `bridge_sigma=0.02`, `i2sb_predictor_time_floor=0.10`, parent checkpoint,
  pure latent tokenizer, TopoGate, legacy semantic cross-attention, terminal
  SWD, dataset, fast10 transfer eval, and vlen `0.10`.
- Changed only:
  `bridge.w_content_lowpass_anchor=0.35`,
  `bridge.w_content_edge_anchor=0.10`,
  `bridge.content_anchor_lowpass_kernel=9`.
- Explicit eval isolation:
  `training.full_eval_in_process=false`,
  `training.full_eval_runtime_model_cache=false`.

## Decision Rule

- Positive:
  retain transfer style above the current clean-I2SB late point while reducing
  LPIPS toward the `0.35` tolerance band.
- Strong positive:
  transfer style `>=0.700` with LPIPS `<=0.38`, or style `>=0.690` with LPIPS
  `<=0.35`.
- Negative:
  style falls below `0.690` while LPIPS remains above `0.38`, or the curve
  becomes style-flat around the predec/latent-affine frontier.
- Convergence:
  use all retained checkpoint `CLIP-S + LPIPS`; do not stop while the best
  checkpoint is among the newest two retained checkpoints.

## Matched Controls

- Clean absolute I2SB sigma0p02:
  best e2 transfer `0.709094 / 0.490233`, e5 `0.704671 / 0.408530`.
- Blend0p25 I2SB:
  best e2 transfer `0.694567 / 0.415258`, e6 `0.690439 / 0.382179`.
- Current in-band diagnostic frontier:
  latent-affine s0.75 transfer `0.685444 / 0.344580`.

## Runtime Notes

- Remote WSL path:
  `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- Batch/eval inherited from base: b8/a2 bf16, fast10 transfer-only eval every
  retained checkpoint.
- Low VRAM is acceptable; OOM or eval-process accumulation is not.

## Live Log

- 2026-06-16: config and plan created. Next action is local config smoke,
  remote sync, and remote WSL launch.
- 2026-06-16 07:24 CST: remote WSL launch started with PID `18759`.
- 2026-06-16 07:25 CST health check: process alive in epoch `1/24`, dataset
  cache loaded from `/mnt/i/wikiarts_5_full_notest_latents_ema/train`, parent
  checkpoint loaded from
  `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- First batch observability confirms the new anchors are active:
  `cla=0.4473` for content lowpass anchor and `cea=0.0504` for content edge
  anchor. GPU read was approximately `8% util / 2883 MiB / 45.6 W`; this low
  memory is expected for b8/a2 bf16 and is not a stop condition.
- Pending: pull the first training-time fast10 `CLIP-S + LPIPS` summary after
  `epoch_0001`, append transfer rows to `plot_points.csv`, regenerate the
  AAAI2027 page-1 figure, and decide whether to continue from the matched
  delta.
