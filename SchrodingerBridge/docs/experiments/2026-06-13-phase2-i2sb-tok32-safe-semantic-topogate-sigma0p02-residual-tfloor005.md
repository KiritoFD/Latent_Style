# Phase 2: i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005

Date: 2026-06-13

## Role

- current preferred exact-I2SB diagnostic follow-on
- keeps the same safe-tokenizer profile, residual `sigma=0.02`, and promoted structure-side parent as the plain safe I2SB packet
- changes only one theory-facing detail:
  - floor the predictor time passed into the `x_1` estimator to `0.05` on the earliest step

## Why This Exists

- the current exact-I2SB implementation already uses:
  - `solver_family = solver_i2sb`
  - `objective_mode = i2sb_endpoint`
  - `bridge_noise_schedule = exact_brownian`
- but the first inference step still queried the endpoint predictor at exact `t=0`
- training does not sample exact `t=0`; it samples `t in [eps, 1-eps]`
- so the clean theory question is:
  - if exact-I2SB is still weak after removing that earliest-step timestamp mismatch,
  - does the failure still look like a fundamental Distinct5 safe-band limitation rather than a trivial runtime misalignment

## Config

- config:
  - [phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005_seed42_b20a1.json)
- control packet:
  - [phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1.json)

## Deltas

- keep unchanged from the safe residual exact-I2SB diagnostic:
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `objective_mode = i2sb_endpoint`
  - `bridge_noise_schedule = exact_brownian`
  - `bridge_sigma = 0.02`
  - `endpoint_parameterization = residual`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `proximal_mode = crossattn_texture`
  - `tokenizer_structured_temperature = 0.075`
  - `tokenizer_global_gate_scale = 1.15`
- new runtime-alignment probe:
  - `i2sb_predictor_time_floor = 0.05`
  - this only changes the time stamp fed into the endpoint predictor on very early steps
  - it does not change the exact posterior coefficients used by `solver_i2sb`

## Intended Read

- positive diagnostic:
  - LPIPS improves relative to the plain safe I2SB control
  - style does not collapse
  - tokenizer observability remains alive
- negative diagnostic:
  - the packet still exits the safe band quickly
  - then the remaining weakness is less likely to be caused by the trivial `t=0` predictor mismatch

## Queue Position

- this packet is now the preferred `i2sb_diagnostic_only` successor
- the plain safe residual exact-I2SB packet remains as the immediate no-floor control
- it remains diagnostic-only and must not preempt the active structure-side velocity lane

## Parent Refresh

- Source packet: `vel_tok32_safe_semantic_topogate_k085`
- Selection policy: `latest`
- Selected parent epoch: `epoch_0003`
- Selected parent checkpoint: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1/epoch_0003.pt`
- Selected parent metrics: transfer `0.675388 / 0.375598`, all-pairs `0.702936 / 0.371762`
