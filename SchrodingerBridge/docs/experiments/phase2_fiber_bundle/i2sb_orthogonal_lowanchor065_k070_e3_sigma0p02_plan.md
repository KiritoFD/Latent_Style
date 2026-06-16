# I2SB Orthogonal Low-Anchor 0.65 Plan

Date: 2026-06-16

## Goal

Follow up the low-anchor0.50 partial positive with one controlled strength
change. The aim is to reduce LPIPS beyond the e9 `0.372203` point while
keeping transfer CLIP-S at or above `0.700`.

## Controlled Delta

- Base:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  transport, `bridge_sigma=0.02`, exact Brownian bridge schedule, terminal
  SWD, no latent-slerp path, no DINO/VLM, b8 accumulation-2, vlen `0.10`, and
  fast10 transfer-only in-loop eval.
- Only candidate delta:
  `endpoint_orthogonal_low_anchor=0.65` instead of `0.50`.

## Controls

- Low-anchor0.50 e9:
  `0.701429 / 0.372203`.
- Low-anchor0.50 e14 LPIPS floor:
  `0.686635 / 0.348625`, not promoted.
- Hard orthogonal-lowhigh e4:
  `0.698245 / 0.390826`.

## Decision Rule

- Positive:
  reaches `CLIP-S >= 0.700` with LPIPS below low-anchor0.50 e9.
- Strong positive:
  reaches `CLIP-S >= 0.705` with LPIPS `<= 0.37`, or `CLIP-S >= 0.700` with
  LPIPS `<= 0.35`.
- Negative:
  it follows the low-anchor0.50 tail and improves LPIPS only by falling below
  `0.700` style.
- Closure:
  style-first. Later LPIPS-only points do not replace the best target-facing
  checkpoint.

## Runtime Observability

- `i2sb_endpoint_orthogonal_active=1`.
- `i2sb_endpoint_orthogonal_kernel=5`.
- `i2sb_endpoint_orthogonal_high_scale=1`.
- `i2sb_endpoint_orthogonal_low_anchor=0.65`.

## Artifact Targets

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor065_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010/`.

## Launch Log

- 2026-06-16 12:49 remote WSL launch valid. Log confirms parent load:
  `Partially loaded resume ... epoch_0003.pt | loaded=272 skipped=0 missing=0
  unexpected=0`.
- 2026-06-16 12:52 e1 eval:
  transfer `0.709417 / 0.449507`, eval wall `39.61s`. Compared with
  low-anchor0.50 e1, style drops slightly but LPIPS improves by about `0.0235`.
- 2026-06-16 12:55 e2 eval:
  transfer `0.707335 / 0.434842`, eval wall `24.91s`.
- 2026-06-16 12:57 e3 eval:
  transfer `0.704287 / 0.417983`, eval wall `25.66s`.

## Interim Read

- `running_promising_not_in_band`.
- The strength scan behaves monotonically in the intended direction: stronger
  lowpass anchoring improves LPIPS early while paying a controlled style cost.
- It has not yet beaten the low-anchor0.50 e9 candidate
  (`0.701429 / 0.372203`). Continue until either an in-band point appears or
  the tail becomes LPIPS-only below `0.700` style.
