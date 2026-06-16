# I2SB Orthogonal Low-Anchor 0.50 Plan

Date: 2026-06-16

## Goal

Test whether hard low-frequency replacement is the reason
`orthogonal_lowhigh` and `slerp+orthogonal` suppress style. This run keeps the
absolute endpoint style force and weakens only the low-frequency content anchor.

## Controlled Delta

- Base:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  transport, `bridge_sigma=0.02`, exact Brownian bridge schedule, terminal
  SWD, no latent-slerp path, no DINO/VLM, b8 accumulation-2, vlen `0.10`, and
  fast10 transfer-only in-loop eval.
- Only candidate delta:
  `endpoint_orthogonal_low_anchor=0.5`.

Endpoint reconstruction:

```text
raw_low = lowpass(raw_absolute_endpoint)
x_low = lowpass(content)
anchored_low = lerp(raw_low, x_low, 0.5)
endpoint = anchored_low + highpass(raw_absolute_endpoint)
```

The default value is `1.0`, which exactly preserves the previous hard
orthogonal behavior.

## Matched Controls

- Clean absolute I2SB sigma0p02:
  best e2 transfer `0.709094 / 0.490233`.
- Hard orthogonal-lowhigh:
  e1 `0.705847 / 0.451386`, e4 `0.698245 / 0.390826`.
- Slerp+orthogonal:
  e1 `0.704828 / 0.446676`, e15 `0.678109 / 0.350421`.

## Decision Rule

- Positive:
  style is closer to clean absolute I2SB than hard orthogonal e1 while LPIPS is
  lower than clean absolute.
- Strong positive:
  `CLIP-S >= 0.710` with LPIPS below `0.45`, or `CLIP-S >= 0.700` with LPIPS
  below `0.38`.
- Negative:
  it follows the hard projection curve and falls into the low `0.68-0.69`
  style band, or it restores style only by returning to clean absolute LPIPS.
- Closure:
  style-first. Do not treat LPIPS-only tail points as convergence progress
  toward the Seedream target.

## Runtime Observability

Required summary/debug keys:

- `i2sb_endpoint_orthogonal_active=1`.
- `i2sb_endpoint_orthogonal_kernel=5`.
- `i2sb_endpoint_orthogonal_high_scale=1`.
- `i2sb_endpoint_orthogonal_low_anchor=0.5`.

## Artifact Targets

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor050_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/`.

## Launch Log

- 2026-06-16 11:48 first remote launch was stopped and archived as invalid
  because the log showed `No checkpoint found, start from scratch`. This
  violates the matched-parent rule and is not evidence.
- Config hardening:
  the candidate now explicitly sets `training.resume_checkpoint` to the k070 e3
  parent and disables optimizer/training-state resume, instead of relying only
  on inherited base config fields.
