# Phase 2: i2sb_semantic_topogate_sigma0p02_residual

Date: 2026-06-13

## Role

- diagnostic-only exact-I2SB candidate
- not part of the formal remote lane
- prepared so that, once remote WSL2 recovers, the next I2SB theory check is not another broad heuristic sweep

## Why This Exists

- the closed exact-I2SB fallback ladder already established:
  - pure endpoint topology anchors alone were insufficient
  - `crossattn_texture` proximal rescue was necessary
  - `sigma = 0.02` plus residual endpoint parameterization gave the least-bad archival point
- the new question is narrower:
  - if we keep that same residual exact-Brownian packet almost unchanged,
  - does an attention-side topology constraint improve structure at all?
- this is intentionally diagnostic-only:
  - it does not challenge the current Phase-2 queue order
  - it only asks whether the new `semantic_self_topology_gate` deserves future coexistence with true I2SB

## Parent

- direct parent:
  - [phase2_i2sb_pattn_topo_anchor_sigma0p02_residual_warm_vel2_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_pattn_topo_anchor_sigma0p02_residual_warm_vel2_seed42_b22a1.json)
- parent read:
  - transfer `0.688376 / 0.571735`
  - all-pairs `0.697686 / 0.569086`
  - interpretation:
    - still archival-only
    - but this was the cleanest endpoint-side structural rescue among the exact-Brownian retries

## Config

- config:
  - [phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1.json)

## Deltas

- keep unchanged:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `objective_mode = i2sb_endpoint`
  - `bridge_noise_schedule = exact_brownian`
  - `bridge_sigma = 0.02`
  - `endpoint_parameterization = residual`
  - `proximal_mode = crossattn_texture`
  - `w_content_lowpass_anchor = 0.9`
  - `w_content_edge_anchor = 0.3`
  - `batch_size = 22`
- new diagnostic variable:
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`

## Intended Read

- positive diagnostic:
  - LPIPS moves downward relative to the residual archival parent
  - while style does not collapse catastrophically
  - and the new topology signal is clearly active in the training record
- negative diagnostic:
  - structure remains in the same `0.57+` zone
  - or style collapses with no meaningful structural benefit
  - or the topology signal is effectively inactive

## Required Observability

- smoke / runtime contract must expose:
  - `bridge_noise_schedule_exact_metric`
  - `semantic_topology_attn_entropy_metric`
  - `semantic_topology_attn_active_metric`
  - `structured_style_tokenizer_debug`
- reason:
  - this packet is supposed to be both a true-I2SB contract check and a topology-gate theory check

## Smoke

- local synthetic smoke:
  - [phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_semantic_topogate_sigma0p02_residual_warm_vel2_seed42_b22a1_smoke.json)
  - status `ok`
  - `objective_mode = i2sb_endpoint`
  - `solver_family = solver_i2sb`
  - `transport_prediction_mode = endpoint`
  - `bridge_sigma = 0.02`
  - `bridge_noise_schedule_exact_metric = 1.0`
  - tensor shapes:
    - forward `[1, 4, 32, 32]`
    - endpoint `[1, 4, 32, 32]`
    - integrated `[1, 4, 32, 32]`
    - `semantic_attn_shape = [1, 256, 256]`
    - `semantic_topology_attn_shape = [1, 256, 256]`
  - loss read:
    - `loss = 4.301777`
    - `flow = 2.999270`
    - `terminal_swd = 0.037306`
    - `t_mean = 0.100112`
    - `semantic_attn_mean_metric = 0.003906`
    - `semantic_k_abs_metric = 0.070865`
    - `semantic_topology_attn_entropy_metric = 2.966969`
    - `semantic_topology_attn_active_metric = 1.0`
  - tokenizer debug snapshot:
    - `attn_entropy = 2.941125`
    - `attn_max = 0.418621`
    - `num_clusters = 32`
    - `query_dim = 64`
    - `query_num_blocks = 4`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.098917`

## Queue Position

- no change to the live queue
- current order remains:
  - `safe_rescan_r2`
  - then structure-side velocity reentry
  - then diagnostic-only I2SB
- this packet only becomes relevant after the remote host is healthy again and there is idle room for a diagnostic I2SB read
