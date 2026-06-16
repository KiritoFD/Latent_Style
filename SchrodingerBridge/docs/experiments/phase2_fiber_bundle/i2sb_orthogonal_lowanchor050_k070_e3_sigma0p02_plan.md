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
- 2026-06-16 11:54 second remote launch was also stopped and archived as
  invalid because the explicit parent path used `../exp/...`, which is wrong
  for the remote working directory `/mnt/i/Github/Latent_Style`. The config is
  corrected to `./exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- 2026-06-16 11:59 third launch is valid. Log confirms parent load:
  `Partially loaded resume ... epoch_0003.pt | loaded=272 skipped=0 missing=0
  unexpected=0`.
- 2026-06-16 12:02 e1 eval:
  transfer `0.711470 / 0.472991`, eval wall `37.59s`. This nearly preserves
  latent-slerp e2 style while improving clean-I2SB e2 LPIPS, but LPIPS remains
  high.
- 2026-06-16 12:05 e2 eval:
  transfer `0.704958 / 0.429371`, eval wall `24.83s`. Compared with hard
  orthogonal e2 (`0.699997 / 0.420951`), style is higher by about `+0.00496`
  at a modest LPIPS cost. Continue; this is the cleanest positive sign since
  latent-slerp e2, but it still needs an in-band structure point.
- 2026-06-16 12:07 e3 eval:
  transfer `0.705415 / 0.430264`, eval wall `24.86s`. Style remains stable
  near e2 instead of collapsing into the `0.68` band.
- 2026-06-16 12:10 e4 eval:
  transfer `0.705008 / 0.412302`, eval wall `24.95s`. This is the current
  low-anchor structure point: it is much more style-preserving than the
  slerp+hard-orthogonal tail, but LPIPS is still outside the target band.
- 2026-06-16 12:13 e5 eval:
  transfer `0.702532 / 0.393892`, eval wall `24.84s`. This is the best
  balanced point so far: it is close to the `>=0.700 / <=0.38` short gate but
  does not reach it.
- 2026-06-16 12:16 e6 eval:
  transfer `0.692812 / 0.413901`, eval wall `24.92s`. Style retreat without a
  structure gain.
- 2026-06-16 12:18 e7 eval:
  transfer `0.696491 / 0.391731`, eval wall `24.91s`. This is a joint Pareto
  point only under the automatic tracker; under style-first reading it is below
  the useful style band.
- 2026-06-16 12:21 e8 eval:
  transfer `0.698460 / 0.384314`, eval wall `24.98s`. Structure is close to
  the short gate but style remains slightly below `0.700`.
- 2026-06-16 12:24 e9 eval:
  transfer `0.701429 / 0.372203`, eval wall `24.88s`. This is the first
  low-anchor point that clears the short style-first gate
  (`CLIP-S >= 0.700`, LPIPS `<= 0.38`).
- 2026-06-16 12:26 e10 eval:
  transfer `0.690436 / 0.369871`, eval wall `25.75s`. LPIPS improves but style
  falls out of the target band.
- 2026-06-16 12:29 e11 eval:
  transfer `0.686964 / 0.368482`, eval wall `25.02s`. This is LPIPS-only under
  the active target and should not replace e9.

## Interim Read

- `running_positive_in_band_not_closed`.
- The mechanism is doing what the hypothesis predicted: reducing hard lowpass
  anchoring restores style relative to hard orthogonal projection.
- e9 is now the current target-facing candidate:
  `0.701429 / 0.372203`. It improves both style and LPIPS over hard
  orthogonal-lowhigh e4 (`0.698245 / 0.390826`), and recovers `+0.02332`
  CLIP-S versus the slerp+hard-orthogonal LPIPS floor.
- Continue to formal tail because e10/e11 are newer LPIPS-side Pareto points.
  Do not promote a later LPIPS-only point if style remains below `0.700`.
