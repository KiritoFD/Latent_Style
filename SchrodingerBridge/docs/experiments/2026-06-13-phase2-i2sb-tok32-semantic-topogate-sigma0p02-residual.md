# Phase 2: i2sb_tok32_semantic_topogate_sigma0p02_residual

Date: 2026-06-13

## Role

- diagnostic-only exact-I2SB candidate
- not part of the formal remote lane
- first packet that combines:
  - the refreshed `tok32` pure-latent tokenizer stack
  - exact-Brownian residual I2SB
  - attention-side topology gating
- this is now the preferred current I2SB theory-check packet

## Why This Exists

- the earlier I2SB diagnostic packets were already true exact-Brownian runs
- but they still used the older pure-tokenizer parameterization:
  - `query_dim = 64`
  - `query_num_blocks = 4`
  - `pe_temperature = 1.0`
  - `global_gate_scale = 1.0`
- Phase 2 has since established that the cleaner tokenizer mainline is the refreshed `tok32` stack from `vel_tok32_pos_refresh`
- so this packet closes an important evidence gap:
  - if true I2SB still fails even after receiving the current stronger tokenizer, then the remaining problem is much less likely to be “tokenizer too weak”

## Parents

- I2SB contract parent:
  - [phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1.json)
- tokenizer capability parent:
  - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)
  - strongest in-band point:
    - `epoch_0004`
    - transfer `0.673399 / 0.376463`
    - all-pairs `0.701161 / 0.374695`

## Config

- config:
  - [phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)

## Deltas

- keep unchanged from the residual exact-Brownian diagnostic:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `objective_mode = i2sb_endpoint`
  - `bridge_noise_schedule = exact_brownian`
  - `bridge_sigma = 0.02`
  - `endpoint_parameterization = residual`
  - `proximal_mode = crossattn_texture`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `w_content_lowpass_anchor = 0.9`
  - `w_content_edge_anchor = 0.3`
- refreshed tokenizer overrides:
  - `tokenizer_query_dim = 96`
  - `tokenizer_query_num_blocks = 5`
  - `tokenizer_pe_temperature = 0.75`
  - `tokenizer_global_gate_hidden_dim = 192`
  - `tokenizer_global_gate_scale = 1.1`
  - `tokenizer_structured_temperature = 0.08`
- launch hygiene:
  - `batch_size = 20`
  - warm-start target:
    - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1/epoch_0004.pt`

## Intended Read

- positive diagnostic:
  - compared with the older `64/4` I2SB diagnostic, LPIPS moves downward
  - style does not collapse harder than before
  - topology and tokenizer observability remain clearly active
- negative diagnostic:
  - the stronger tokenizer changes little or pushes LPIPS even higher
  - meaning the current I2SB weakness is downstream of tokenizer capacity

## Required Observability

- smoke / runtime contract must expose:
  - `bridge_noise_schedule_exact_metric`
  - `semantic_topology_attn_entropy_metric`
  - `semantic_topology_attn_active_metric`
  - `structured_style_tokenizer_debug`
- token-capacity-specific read:
  - `query_dim`
  - `query_num_blocks`
  - `global_gate_scale`
  - tokenizer attention entropy / max

## Smoke

- local synthetic smoke:
  - [phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1_smoke.json)
  - status `ok`
  - `objective_mode = i2sb_endpoint`
  - `solver_family = solver_i2sb`
  - `transport_prediction_mode = endpoint`
  - `bridge_sigma = 0.02`
  - `bridge_noise_schedule_exact_metric = 1.0`
  - refreshed tokenizer contract:
    - `query_dim = 96`
    - `query_num_blocks = 5`
    - `pe_temp = 0.75`
    - `global_gate_hidden_dim = 192`
    - `global_gate_scale = 1.1`
    - `structured_temperature = 0.08`
    - `batch_size = 20`
    - warm-start target:
      - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1/epoch_0004.pt`
  - tensor shapes:
    - forward `[1, 4, 32, 32]`
    - endpoint `[1, 4, 32, 32]`
    - integrated `[1, 4, 32, 32]`
    - `semantic_attn_shape = [1, 256, 256]`
    - `semantic_topology_attn_shape = [1, 256, 256]`
  - loss read:
    - `loss = 3.708116`
    - `flow = 2.642122`
    - `terminal_swd = 0.026628`
    - `t_mean = 0.268873`
    - `semantic_attn_mean_metric = 0.003906`
    - `semantic_k_abs_metric = 0.070595`
    - `semantic_topology_attn_entropy_metric = 2.828322`
    - `semantic_topology_attn_active_metric = 1.0`
  - tokenizer debug snapshot:
    - `attn_entropy = 2.914052`
    - `attn_max = 0.514943`
    - `num_clusters = 32`
    - `query_dim = 96`
    - `query_num_blocks = 5`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.082069`

## Queue Position

- still diagnostic-only
- no change to the live queue:
  - `safe_rescan_r2`
  - then structure-side velocity reentry
  - then diagnostic-only I2SB
- this packet is the preferred next I2SB theory check after remote recovery
