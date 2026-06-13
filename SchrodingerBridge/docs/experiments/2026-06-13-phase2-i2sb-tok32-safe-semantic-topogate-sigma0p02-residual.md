# Phase 2: i2sb_tok32_safe_semantic_topogate_sigma0p02_residual

Date: 2026-06-13

## Role

- preferred current exact-Brownian diagnostic successor
- same residual `sigma=0.02` theory check as the earlier `tok32_refresh` packet
- upgraded to the current `tok32_safe_rescan` tokenizer profile and the cleaner `epoch_0004` in-band parent

## Why This Exists

- the older `i2sb_tok32_semantic_topogate_sigma0p02_residual` packet already answered:
  - refreshed tokenizer is better than legacy64
- but it still inherited the older `tok32_refresh` tokenizer settings
- now that `safe_rescan_r2` has produced a lower-LPIPS in-band tokenizer state, the clean theory question becomes:
  - if exact-I2SB still fails even after inheriting the safer tokenizer profile,
  - then the remaining weakness is even less likely to be “tokenizer still not clean enough”

## Config

- config:
  - [phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- parent packets:
  - [phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
  - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)

## Deltas

- keep unchanged from the residual exact-Brownian diagnostic:
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `objective_mode = i2sb_endpoint`
  - `bridge_noise_schedule = exact_brownian`
  - `bridge_sigma = 0.02`
  - `endpoint_parameterization = residual`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `proximal_mode = crossattn_texture`
- safe-tokenizer upgrade:
  - `tokenizer_structured_temperature: 0.08 -> 0.075`
  - `tokenizer_global_gate_scale: 1.10 -> 1.15`
- provisional warm start:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1/epoch_0004.pt`

## Intended Read

- positive diagnostic:
  - LPIPS improves relative to the older `tok32_refresh` I2SB packet
  - without collapsing tokenizer routing observability
- negative diagnostic:
  - exact-I2SB remains far outside the safe band
  - even after inheriting the stronger safe-tokenizer profile

## Queue Position

- this packet supersedes the older `i2sb_tok32_semantic_topogate_sigma0p02_residual` packet as the preferred exact-I2SB theory check
- it remains diagnostic-only and must not preempt the formal velocity lane
