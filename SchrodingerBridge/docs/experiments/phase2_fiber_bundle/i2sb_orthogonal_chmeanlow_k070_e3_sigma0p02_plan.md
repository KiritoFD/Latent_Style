# I2SB Orthogonal Channel-Mean Lowpass Plan

Date: 2026-06-16

## Goal

The scalar low-anchor sweep showed a consistent failure mode: stronger
lowpass anchoring improves LPIPS only after transfer style falls below `0.700`.
The next mechanism therefore changes what is anchored rather than how strongly
all channels are anchored.

This run anchors only the channel-mean lowpass component while preserving the
channel-relative low-frequency residuals. In latent space this is a conservative
proxy for keeping shared structure stable while allowing low-frequency
style/color actuation to survive.

## Controlled Delta

- Base:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  transport, `bridge_sigma=0.02`, exact Brownian bridge schedule, terminal
  SWD, no latent-slerp path, no DINO/VLM, b8 accumulation-2, vlen `0.10`, and
  fast10 transfer-only in-loop eval.
- Only candidate mechanism:
  `endpoint_orthogonal_low_mode=channel_mean` with low anchor `1.0`.

## Controls

- Hard all-channel lowhigh e4:
  `0.698245 / 0.390826`.
- Low-anchor0.50 e9:
  `0.701429 / 0.372203`.
- Low-anchor0.55 e4:
  `0.704881 / 0.405001`; e11 LPIPS-only `0.688107 / 0.353115`.
- Low-anchor0.65 e4:
  `0.706564 / 0.395071`; e9 LPIPS-only `0.692446 / 0.358758`.

## Decision Rule

- Positive:
  reaches `CLIP-S >= 0.700` with LPIPS below `0.372203`.
- Strong positive:
  reaches `CLIP-S >= 0.705` with LPIPS `<= 0.37`, or `CLIP-S >= 0.700` with
  LPIPS `<= 0.35`.
- Negative:
  behaves like the scalar anchor tails: LPIPS improves only after transfer
  style falls below `0.700`.
- Closure:
  style-first. Later LPIPS-only points do not replace the best target-facing
  checkpoint.

## Runtime Observability

- `i2sb_endpoint_orthogonal_active=1`.
- `i2sb_endpoint_orthogonal_kernel=5`.
- `i2sb_endpoint_orthogonal_high_scale=1`.
- `i2sb_endpoint_orthogonal_low_anchor=1`.
- `i2sb_endpoint_orthogonal_low_mode_channel_mean=1`.

## Artifact Targets

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_chmeanlow_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010/`.

## Launch Log

- 2026-06-16 local smoke passed:
  `endpoint_orthogonal_low_mode=channel_mean` and legacy `all` mode both
  execute `_endpoint_delta_from_raw` with finite outputs.
- 2026-06-16 14:20 remote WSL launch. `git pull` was blocked by a transient
  GitHub TLS failure, so the committed source/config/plan files were copied
  directly to the remote workspace before launch.
- 2026-06-16 14:20 health check: parent checkpoint load confirmed:
  `Partially loaded resume ... epoch_0003.pt | loaded=272 skipped=0 missing=0
  unexpected=0`.
- 2026-06-16 14:21 training entered epoch 1. GPU observed around `3.1 GiB`
  with high utilization. Early train loss/kinetic are higher than scalar
  low-anchor runs, so first eval is required before interpreting stability.
- 2026-06-16 14:23 e1 eval:
  transfer `0.697062 / 0.482289`, eval wall `42.34s`.
- 2026-06-16 14:25 e2 eval:
  transfer `0.702899 / 0.513291`, eval wall `24.96s`.

## Interim Read

- `running_negative_structure_unstable`.
- Channel-mean lowpass anchoring preserves too much low-frequency freedom in
  the first two checkpoints: e2 recovers style above `0.700`, but LPIPS rises
  to `0.513291`, worse than clean absolute I2SB sigma0p02 and far outside the
  target structure band.
- Continue to e4 before closure because this family may cool later, but it must
  show a sharp LPIPS correction while preserving `0.700+` style to remain
  viable.
