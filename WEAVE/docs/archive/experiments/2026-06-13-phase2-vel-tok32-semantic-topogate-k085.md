# Phase 2: vel_tok32_semantic_topogate_k085

Date: 2026-06-13

> 2026-06-13 supersession note:
> This packet remains a valid `tok32_refresh`-parent structure-side reference.
> But the preferred successor is now:
> [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md)
> because it inherits the current `tok32_safe_rescan` tokenizer profile and the cleaner in-band `epoch_0004` parent instead of the older `pos_refresh` parent.

## Role

- queued structure-side candidate after the safe-family tokenizer sweep
- not launched yet
- cleaner than reopening a full attention-family branch
- only becomes eligible if the formal lane leaves tokenizer-only safe rescan

## Why This Exists

- `docs/plan/attn.md` already argued that the cleanest structure-preserving move is to keep content self-attention topology while injecting style through the value path
- round-1 `attn_sa_mod` showed that the mechanism is implementable, but not strong enough as a standalone promotion family
- phase-2 therefore uses the lighter version:
  - keep `legacy_semantic_crossattn`
  - turn on `semantic_self_topology_gate`
  - blend style-routing logits toward content self-affinity
- this asks a narrower question than `topo_anchor` loss reentry:
  - can attention-side topology anchoring buy structure safety without paying the full style tax of a stronger loss-side anchor?

## Config

- config:
  - [phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1.json)
- parent packet:
  - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)

## Deltas

- keep unchanged:
  - `tokenizer_family = pure_latent_spatial`
  - refreshed tokenizer dimensions / position encoding / global-spatial coupling
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
  - `batch_size = 20`
- structure reentry overrides:
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `w_kinetic: 1.0 -> 0.85`
- queued warm start:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1/epoch_0004.pt`

## Intended Read

- success condition:
  - style rises above the current `0.701161 / 0.374695` shelf
  - while staying in-band under `LPIPS < 0.40`
  - and while the new topology signal is visibly active in epoch logs
- failure condition:
  - style still plateaus under the shelf
  - or LPIPS still leaks into `0.40+`
  - or the topology signal stays effectively inactive despite the switch being on

## Required Observability

- epoch CSV and epoch log must persist:
  - `semantic_topology_attn_entropy`
  - `semantic_topology_attn_active`
- reason:
  - this packet is defined by an attention-side topology constraint
  - so the training record must show whether that route was actually active

## Launch Rule

- do not launch while `safe_rescan_r2` still owns the next formal slot
- before launch:
  - rerun local smoke on the exact JSON
  - if a stronger in-band parent than `epoch_0004` emerges, replace the warm-start checkpoint

## Smoke

- local synthetic smoke:
  - [phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1_smoke.json)
  - status `ok`
  - `objective_mode = bridge_velocity`
  - `tokenizer_family = pure_latent_spatial`
  - `solver_family = euler_legacy`
  - `transport_prediction_mode = velocity`
  - no DINO runtime required
  - tensor shapes:
    - forward `[1, 4, 32, 32]`
    - endpoint `[1, 4, 32, 32]`
    - integrated `[1, 4, 32, 32]`
    - `semantic_attn_shape = [1, 256, 256]`
    - `semantic_topology_attn_shape = [1, 256, 256]`
  - loss read:
    - `loss = 2.298454`
    - `flow = 2.069974`
    - `terminal_swd = 0.009923`
    - `t_mean = 0.456406`
    - `semantic_attn_mean_metric = 0.003906`
    - `semantic_k_abs_metric = 0.070684`
    - `semantic_topology_attn_entropy_metric = 2.805402`
    - `semantic_topology_attn_active_metric = 1.0`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.004326`
  - tokenizer debug snapshot:
    - `attn_entropy = 2.914052`
    - `attn_max = 0.514943`
    - `num_clusters = 32`
    - `query_dim = 96`
    - `query_num_blocks = 5`
- local gate-off/on probe:
  - mean absolute output delta `0.0189447`
  - topology attention entropy `2.755764`
  - `topology_active = true`
- local objective-metrics probe:
  - `semantic_topology_attn_entropy = 2.837168`
  - `semantic_topology_attn_active = 1.0`
- local log-column check:
  - epoch CSV now persists:
    - `semantic_topology_attn_entropy`
    - `semantic_topology_attn_active`

## Queue Position

- current formal queue is unchanged:
  - `safe_rescan_r2`
  - then structure-side reentry
  - then diagnostic-only I2SB
- inside structure-side reentry, this packet is now available as the cleaner attention-side alternative to the heavier loss-side `topo_anchor` retry
