# Phase 2: vel_tok32_safe_semantic_topogate_k085

Date: 2026-06-13

## Role

- preferred structure-side reentry successor once `safe_rescan_r2` closes
- same structure-control question as the earlier `tok32_refresh` topology-gate packet
- upgraded to the current `tok32_safe_rescan` tokenizer profile and a cleaner in-band parent

## Why This Exists

- `vel_tok32_semantic_topogate_k085` was prepared on the older `tok32_refresh` parent
- since then the formal lane established a cleaner safe-band point at:
  - `safe_rescan_r2 epoch_0004`
  - transfer `0.672377 / 0.369065`
  - all-pairs `0.700490 / 0.367229`
- if structure-side reentry is going to try to break the safe shelf cleanly, it should inherit:
  - the current safer tokenizer profile
  - the current lower-LPIPS parent
  - not the older `pos_refresh` packet

## Config

- preferred relaunch config:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1.json)
- first launch config:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)

## Deltas

- keep from the current safe tokenizer profile:
  - `tokenizer_query_dim = 96`
  - `tokenizer_query_num_blocks = 5`
  - `tokenizer_pe_temperature = 0.75`
  - `tokenizer_global_gate_hidden_dim = 192`
  - `tokenizer_global_gate_scale = 1.15`
  - `tokenizer_structured_temperature = 0.075`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
- structure reentry override:
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `w_kinetic: 0.95 -> 0.85`
- provisional warm start:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1/epoch_0004.pt`

## Launch Read

- first `b20a1` launch did start correctly on the remote host
- but the runtime guard stopped it before the first epoch closed:
  - observed memory `11093 MiB`
  - guard cap `11000 MiB`
- preferred relaunch:
  - switch to the `b16a1` packet above on the same safe parent
  - keep the model family unchanged
  - change only the remote memory fit
- `b16a1` relaunch at `2026-06-13 17:41 +08:00`:
  - remote live state:
    - `training_before_first_settled_eval`
  - first remote health read:
    - roughly `9001 MiB / 12288 MiB`
  - interpretation:
    - the relaunch is back inside the formal memory band
    - now the scientific question returns to board behavior rather than infrastructure failure
- first settled authority point at `2026-06-13 18:08:07 +08:00`:
  - `epoch_0001`
  - transfer `0.672300 / 0.361248`
  - all-pairs `0.701787 / 0.357462`
  - identity `0.819733 / 0.342317`
  - interpretation:
    - this already beats the old all-pairs safe shelf `0.701666 / 0.381724`
    - transfer style is still slightly below the transfer shelf `0.673934`, but LPIPS is much cleaner
    - so the packet is already a real in-band structure-side recovery candidate on its first retained checkpoint
- latest settled authority point:
  - `epoch_0002`
  - transfer `0.671915 / 0.361009`
  - all-pairs `0.700605 / 0.357866`
  - interpretation:
    - LPIPS remains comfortably clean
    - but the board has slipped back below the formal safe shelf on both transfer and all-pairs style
    - therefore the lane stays open as an active structure-side candidate, not yet as a promoted recovery
- queued low-intrusion follow-on:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json)
  - [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085-appalign.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085-appalign.md)
  - rationale:
    - if qualitative review still shows brightness / contrast mismatch, the next clean question is low-order appearance alignment
    - not another large structure-family jump yet

## Intended Read

- success condition:
  - style rises past the current safe shelf
  - while staying under `LPIPS < 0.40`
  - and tokenizer observability remains clearly active rather than collapsing into a dead route
- required tokenizer reads:
  - `structured_style_tokenizer_attn_entropy`
  - `structured_style_tokenizer_attn_effective_count`
  - `structured_style_tokenizer_gate_mean`
  - `structured_style_tokenizer_mask_mean`
  - `structured_style_tokenizer_spatial_map_abs`
  - `structured_style_tokenizer_global_gate_abs`

## Queue Position

- this packet should be the first structure-side successor if `safe_rescan_r2` fails to produce a promotable shelf break
- it supersedes the older `vel_tok32_semantic_topogate_k085` packet as the preferred structure-reentry packet
## Parent Refresh

- Source formal packet: `vel_tok32_safe_rescan_r2`
- Selection policy: `best_clean_allpairs`
- Selected parent epoch: `epoch_0004`
- Selected parent checkpoint: `exp/aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1/epoch_0004.pt`
- Selected parent metrics: transfer `0.672377 / 0.369065`, all-pairs `0.700490 / 0.367229`
