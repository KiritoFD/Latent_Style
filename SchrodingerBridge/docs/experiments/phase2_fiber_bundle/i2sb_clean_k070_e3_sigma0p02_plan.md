# Clean I2SB Endpoint Path: k070 e3 sigma0p02

## Purpose

Test the `fiber.md` path/topology diagnosis after several actuation-only probes
failed to produce material style lift. This experiment changes the bridge/path
family, not the tokenizer, attention family, loss extras, or output actuation
modules.

## Controlled Change

- Parent checkpoint:
  `../exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Config:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Changed:
  `solver_family=solver_i2sb`,
  `transport_prediction_mode=endpoint`,
  `bridge.objective_mode=i2sb_endpoint`,
  `bridge_sigma=0.02`.
- Stabilizers tied to endpoint/I2SB only:
  `endpoint_velocity_time_floor=0.10`,
  `i2sb_predictor_time_floor=0.10`,
  `bridge_noise_schedule=exact_brownian`.
- Unchanged:
  `tokenizer_family=pure_latent_spatial`, `num_clusters=32`,
  `semantic_self_topology_blend=0.7`, tokenizer-guided appearance alignment,
  legacy semantic cross-attention, terminal SWD family/weight, dataset, pairing
  cache, and fast10 eval surface.
- Explicitly disabled:
  PnP self-injection, SMoE, fiberwise SWD, style-delta basis, pre-decoder
  section, proximal residual, and PC/lowpass corrector.

## Eval Contract

- Training-time eval subdir: `full_eval_fast10`.
- Transfer-only, `10` source samples per style.
- ONNX VAE decoder: `../eval_cache/vae_onnx/ema_b16_32/decoder.onnx`.
- Source latent cache enabled.
- `CLIP-S + LPIPS` every retained checkpoint.
- Generated-delta observability enabled for every checkpoint.
- Plot data must be appended to
  `docs/experiments/phase2_fiber_bundle/plot_points.csv` after each stage
  closure.

## Decision Rule

- Style priority: continue while transfer CLIP-S is improving and LPIPS remains
  at or below the Seedream-like `~0.35` tolerance band.
- Hard negative:
  if transfer LPIPS jumps above `0.38` before style exceeds the current I2SB
  mixed-screen transfer style (`~0.684`), stop and archive as endpoint-cost
  negative.
- Positive signal:
  style above the R16/proximal/predec frontier with LPIPS not materially worse
  than the accepted Seedream band.
- Formal closure:
  use all-ckpt fast10 curve for convergence; do not promote without selected
  full-transfer confirmation.

## Remote Launch Plan

- Remote WSL repo:
  `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- Batch: `8`, accumulation: `2`, AMP bf16, channels_last, no torch.compile.
- Virtual length: `0.10` for finer epoch-level reads and faster feedback.
- First health check within `30s`; low VRAM is acceptable if explained by batch
  and mechanism, but any actual OOM/exploded memory stops the run.

## Initial Decision Context

- Pre-decoder section closed flat:
  best fast10 e3 `0.680121 / 0.315600`, final e18
  `0.680047 / 0.315528`.
- Mixed I2SB + PnP + fiberwise + SMoE was style-positive but immediately
  out-of-band:
  e1 transfer `0.684073 / 0.394578`.
- This clean run exists to separate endpoint/I2SB path effects from PnP,
  fiberwise SWD, and SMoE confounders.

## Live Result And Closure

- Remote run was stopped after e5 because style peaked at e2 and then declined
  for three retained checkpoints while LPIPS remained out of the accepted
  Seedream-like band.
- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_clean_k070_e3_sigma0p02_fast10_curve.csv`.

| epoch | transfer CLIP-S | transfer LPIPS | eval wall |
|---:|---:|---:|---:|
| 1 | `0.700250` | `0.543142` | `27.54s` |
| 2 | `0.709094` | `0.490233` | `26.36s` |
| 3 | `0.708411` | `0.496648` | `26.36s` |
| 4 | `0.705868` | `0.442701` | `26.13s` |
| 5 | `0.704671` | `0.408530` | `26.37s` |

- Decision: `closed_negative_style_reversal_high_lpips`.
- Interpretation:
  endpoint I2SB is a real style actuator compared with the k070/predec
  frontier, but the absolute endpoint target is too destructive under
  `sigma=0.02` and current anchoring. Continue from this evidence with a
  lower-cost endpoint/residual variant, not by extending this exact lane.
- Eval infra note:
  the same e5 checkpoint was used for runtime-cache eval A/B. The hot cached
  path reduced exact fast10 outer wall from `43.99s` to `28.41s`; see
  `docs/experiments/phase2_fiber_bundle/eval/speed_probes/i2sb_e5_runtime_cache_speed_probe.json`.
