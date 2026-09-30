# Phase 2: i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005

Date: 2026-06-13

## Role

- current preferred exact-I2SB diagnostic follow-on
- keeps the same safe-tokenizer profile and residual `sigma=0.02`
- now inherits the highest-style recovered `appalign epoch_0001` parent instead of the older promoted `topogate epoch_0003`
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
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json)

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
  - and less likely to be caused by "the parent still was not style-strong enough inside the recovered structure band"

## Queue Position

- this packet is now the preferred `i2sb_diagnostic_only` successor
- the plain safe residual exact-I2SB packet remains as the immediate no-floor control
- current parent choice follows the guide read:
  - once `appalign` stayed below `transfer style = 0.68` while keeping LPIPS near `0.31`,
  - the next exact-I2SB probe should inherit the stronger style-side settled point from the recovered line
- it remains diagnostic-only and must not preempt the active structure-side velocity lane

## Launch Read

- the first auto-pivot attempt from `appalign` exposed one infra bug:
  - the remote launcher synced only the top-level config and not the `_base` chain
  - that made the first remote load fail before training started
- this has now been repaired in the launcher:
  - future remote launches sync the full config dependency chain
  - the required safe-I2SB base configs have also already been synced to the remote workspace
- current meaning:
  - the queued `i2sb_diagnostic_only` packet is still pending on the `appalign` close gate
  - but the earlier missing-config failure is no longer the expected blocker

## Parent Refresh

- Source packet: `vel_tok32_safe_semantic_topogate_k085_appalign`
- Selection policy: `best_transfer_style_within_recovered_structure_band`
- Selected parent epoch: `epoch_0001`
- Selected parent checkpoint: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt`
- Selected parent metrics: transfer `0.672604 / 0.336357`, all-pairs `0.703506 / 0.332992`
