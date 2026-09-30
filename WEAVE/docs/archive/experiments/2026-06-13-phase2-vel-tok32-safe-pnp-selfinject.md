# Phase 2: vel_tok32_safe_pnp_selfinject

Date: 2026-06-13

## Role

- structure-side fallback candidate behind the active `safe_semantic_topogate` run
- same safe tokenizer profile but now refreshed to the promoted `topogate epoch_0003` parent
- changes the structure mechanism from topology-blended semantic cross-attention to explicit `attn_pnp_selfinject`

## Why This Exists

- `612-phase2` already kept `PnP self-inject` as an allowed structure-side tool
- current active packet answers the topology-gate question first
- if it fails to break the board safely, the clean next question is:
  - can direct self-attention structure injection preserve content better than semantic-topology blending
  - without paying the heavier loss-side tax of `topo_anchor`

## Config

- config:
  - [phase2_vel_tok32_safe_pnp_selfinject_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_pnp_selfinject_seed42_b16a1.json)
- safe parent:
  - `vel_tok32_safe_semantic_topogate_k085 epoch_0003`
  - transfer `0.675388 / 0.375598`
  - all-pairs `0.702936 / 0.371762`

## Deltas

- keep:
  - `tokenizer_family = pure_latent_spatial`
  - `tokenizer_query_dim = 96`
  - `tokenizer_query_num_blocks = 5`
  - `tokenizer_pe_temperature = 0.75`
  - `tokenizer_global_gate_scale = 1.15`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
- structure swap:
  - `backbone_attention_family = attn_pnp_selfinject`
- mild structure rollback:
  - `w_kinetic: 0.95 -> 0.90`
- launch shape:
  - `batch_size = 16`
  - `accumulation_steps = 1`

## Intended Read

- success:
  - beat the current safe shelf while staying in-band
  - preserve tokenizer observability
  - avoid the memory overshoot seen in the first `b20a1` topology-gate launch
- failure:
  - style remains pinned near `0.70x`
  - or LPIPS leaves the band
  - or no new Pareto point appears relative to the active topology-gate packet

## Queue Position

- this is the next structure-side fallback after the active `vel_tok32_safe_semantic_topogate_k085` run
- it should be evaluated before the heavier `topo_anchor` loss-side fallback
- if it also closes without a formal shelf break, the next same-lane successor is:
  - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)
## Parent Refresh

- Source packet: `vel_tok32_safe_semantic_topogate_k085`
- Selection policy: `latest`
- Selected parent epoch: `epoch_0003`
- Selected parent checkpoint: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1/epoch_0003.pt`
- Selected parent metrics: transfer `0.675388 / 0.375598`, all-pairs `0.702936 / 0.371762`
