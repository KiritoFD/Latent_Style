# I2SB Orthogonal Low-Anchor 0.55 Plan

Date: 2026-06-16

## Goal

Scan the middle of the low-anchor strength bracket after:

- low-anchor0.50: partial positive; e9 reaches `0.701429 / 0.372203`.
- low-anchor0.65: negative over-anchor; e9/e11 reach `~0.36` LPIPS only after
  transfer style falls below `0.700`.

The target for this controlled run is to preserve the `0.700+` transfer style
band while moving LPIPS below the low-anchor0.50 e9 value.

## Controlled Delta

- Base:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  transport, `bridge_sigma=0.02`, exact Brownian bridge schedule, terminal
  SWD, no latent-slerp path, no DINO/VLM, b8 accumulation-2, vlen `0.10`, and
  fast10 transfer-only in-loop eval.
- Only candidate delta:
  `endpoint_orthogonal_low_anchor=0.55`.

## Controls

- Low-anchor0.50 e9:
  `0.701429 / 0.372203`.
- Low-anchor0.65 e4:
  `0.706564 / 0.395071`.
- Low-anchor0.65 e9:
  `0.692446 / 0.358758`, LPIPS-only not promoted.

## Decision Rule

- Positive:
  reaches `CLIP-S >= 0.700` with LPIPS below `0.372203`.
- Strong positive:
  reaches `CLIP-S >= 0.705` with LPIPS `<= 0.37`, or `CLIP-S >= 0.700` with
  LPIPS `<= 0.35`.
- Negative:
  repeats the `0.65` pattern: LPIPS improves only after transfer style falls
  below `0.700`.
- Closure:
  style-first. Later LPIPS-only points do not replace the best target-facing
  checkpoint.

## Runtime Observability

- `i2sb_endpoint_orthogonal_active=1`.
- `i2sb_endpoint_orthogonal_kernel=5`.
- `i2sb_endpoint_orthogonal_high_scale=1`.
- `i2sb_endpoint_orthogonal_low_anchor=0.55`.

## Artifact Targets

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor055_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010/`.

## Launch Log

- 2026-06-16 13:30 remote WSL launch. `git pull` was blocked by a transient
  GitHub TLS failure, so the already-committed config/plan were copied directly
  to the remote workspace before launch.
- 2026-06-16 13:30 health check: parent checkpoint load confirmed:
  `Partially loaded resume ... epoch_0003.pt | loaded=272 skipped=0 missing=0
  unexpected=0`.
- 2026-06-16 13:31 training entered epoch 1. GPU observed around `3.1 GiB`
  with high utilization; low VRAM is acceptable here because the goal is stable
  throughput and in-loop fast10 eval rather than filling memory.
- 2026-06-16 13:33 e1 eval:
  transfer `0.711863 / 0.457232`, eval wall `39.94s`. This is style-healthier
  than low-anchor0.65 e1 (`0.709417 / 0.449507`) and slightly above
  low-anchor0.50 e1 (`0.711470 / 0.472991`) while improving LPIPS by `0.0158`.
- 2026-06-16 13:36 e2 eval:
  transfer `0.703175 / 0.433346`, eval wall `24.95s`.
- 2026-06-16 13:38 e3 eval:
  transfer `0.698490 / 0.419661`, eval wall `24.88s`. This single point is
  below the style band, but the next checkpoint recovers.
- 2026-06-16 13:41 e4 eval:
  transfer `0.704881 / 0.405001`, eval wall `24.98s`. This is not yet in-band,
  but it keeps style above `0.700` while cooling LPIPS toward the low-anchor0.65
  e4 point (`0.706564 / 0.395071`).
- 2026-06-16 13:44 e5 eval:
  transfer `0.698690 / 0.419045`, eval wall `24.95s`. This loses the `0.700`
  style band and LPIPS worsens, so it is not a usable cooling step.
- 2026-06-16 13:46 e6 eval:
  transfer `0.696947 / 0.423814`, eval wall `24.91s`.
- 2026-06-16 13:49 e7 eval:
  transfer `0.690097 / 0.391744`, eval wall `24.94s`.
- 2026-06-16 13:52 e8 eval:
  transfer `0.697144 / 0.379475`, eval wall `24.93s`.
- 2026-06-16 13:54 e9 eval:
  transfer `0.695546 / 0.382781`, eval wall `24.95s`.
- 2026-06-16 13:57 e10 eval:
  transfer `0.694382 / 0.397952`, eval wall `25.83s`.
- 2026-06-16 14:00 e11 eval:
  transfer `0.688107 / 0.353115`, eval wall `24.97s`.
- 2026-06-16 14:03 e12 eval:
  transfer `0.689145 / 0.365202`, eval wall `24.94s`.
- 2026-06-16 14:04 remote run stopped. GPU returned to idle.

## Interim Read

- `closed_negative_lpips_only_tail`.
- e1 supports the bracket hypothesis: `0.55` preserves the style impulse better
  than `0.65` and anchors structure more than `0.50`.
- The decision point remains the cooling tail. e4 is a useful target-facing
  checkpoint (`0.704881 / 0.405001`) but still too high in LPIPS. Continue to
  e7-e9 to test whether it can beat low-anchor0.50 e9
  (`0.701429 / 0.372203`) without repeating the low-anchor0.65 sub-`0.700`
  style collapse.
- e5/e6 are negative evidence: style falls below `0.700` before LPIPS improves
  into the low-anchor0.50 e9 band. Continue a few more retained checkpoints for
  formal closure rather than stopping on the first bad tail.
- e7-e12 close the run: LPIPS reaches `0.353115` at e11, but only with
  transfer style `0.688107`. The best target-facing point remains e4
  (`0.704881 / 0.405001`), which does not replace low-anchor0.50 e9
  (`0.701429 / 0.372203`).
