# Geometry-Aware I2SB: Orthogonal Low/High Endpoint

## Purpose

Test the geometric correction suggested by the I2SB failure analysis: do not
shorten the endpoint vector with scalar `blend`; instead preserve the absolute
endpoint high-frequency style residual while replacing the endpoint low
frequency with the current content lowpass.

## Controlled Change

- Base:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Parent:
  `../exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Changed only:
  `model.endpoint_parameterization=orthogonal_lowhigh`,
  `model.endpoint_orthogonal_kernel=5`,
  `model.endpoint_orthogonal_high_scale=1.0`.
- Endpoint formula:
  `endpoint = lowpass(content) + highpass(raw_absolute_endpoint)`.
- Explicitly not changed:
  no content-anchor loss, no `blend`, no residual endpoint, no gated SDE noise,
  no PC corrector, no tokenizer/backbone/loss schedule change.

## Matched Controls

- Clean absolute I2SB sigma0p02:
  best e2 transfer `0.709094 / 0.490233`, e5 `0.704671 / 0.408530`.
- Blend0p25:
  best e2 transfer `0.694567 / 0.415258`, e6 `0.690439 / 0.382179`.
- Content-anchor e1-e3 early curve:
  e1 `0.691647 / 0.517638`, e2 `0.701186 / 0.528431`, e3
  `0.696943 / 0.500977`.

## Decision Rule

- Positive:
  transfer style stays near clean-I2SB strength while LPIPS improves more than
  the content-anchor loss probe.
- Strong positive:
  `CLIP-S >= 0.700` and LPIPS `<=0.38`, or `CLIP-S >=0.690` and LPIPS
  `<=0.35`.
- Negative:
  style drops to the `0.68-0.69` band without bringing LPIPS below blend0p25,
  or highpass-only reconstruction creates unstable artifacts/metric collapse.
- Closure:
  use all retained checkpoint `CLIP-S + LPIPS`; do not stop while the best
  checkpoint is among the newest two retained checkpoints.

## Runtime Observability

- Eval summaries must expose:
  `i2sb_endpoint_orthogonal_active=1`,
  `i2sb_endpoint_orthogonal_kernel=5`,
  `i2sb_endpoint_orthogonal_high_scale=1.0`.

## Live Log

- 2026-06-16: code switch, config, and plan created locally. Next actions:
  local smoke, remote sync, then launch after the content-anchor probe is
  stopped or closed.
- 2026-06-16 07:51 CST: content-anchor probe stopped after e8 and this run
  launched on remote WSL, PID `19428`.
- 2026-06-16 07:52 CST health check: process alive in epoch `1/24`, parent
  checkpoint loaded, training progressing at roughly `3.3 GiB` VRAM. This is
  normal for b8/a2 bf16 and not a stop condition.
- 2026-06-16 07:55 CST e1 eval:
  transfer `0.705847 / 0.451386`, eval wall `26.12s`, metric eval `9.66s`,
  generation `5.38s`, VAE decode `8.33s`.
- Runtime summary confirms the mechanism is active:
  `i2sb_endpoint_orthogonal_active=1.0`,
  `i2sb_endpoint_orthogonal_kernel=5.0`,
  `i2sb_endpoint_orthogonal_high_scale=1.0`.
- 2026-06-16 07:58 CST e2 eval:
  transfer `0.699997 / 0.420951`, eval wall `28.71s`, metric eval `11.49s`,
  generation `5.63s`, VAE decode `8.45s`.
- Matched read:
  orthogonal e1 improves over content-anchor e1 by `+0.014200` CLIP-S and
  `-0.066253` LPIPS, and improves over blend0p25 e1 by `+0.015801` CLIP-S
  with only `+0.003906` worse LPIPS. e2 then trades style down for structure,
  landing near the blend0p25 structure band but still with stronger style than
  blend0p25 e6.
- Current decision:
  continue to at least e3 because e1 is the best style point and is still in
  the newest two retained checkpoints. If e3 continues the style retreat
  without moving LPIPS toward `<=0.38`, stop this line as a partial positive
  but not a promoted frontier.
- Local artifacts:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowhigh_k070_e3_sigma0p02_fast10_curve.csv`
  and
  `docs/experiments/phase2_fiber_bundle/eval/i2sb_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010/`.
- Plot update:
  appended e1-e2 transfer points to `plot_points.csv` and regenerated the
  AAAI2027 WikiArt-5 page-1 figure.
