# Phase 2: vel_tok32_safe_semantic_topogate_k085_appalign

Date: 2026-06-13

## Role

- low-intrusion structure-side follow-on behind the active `safe_semantic_topogate` run
- same safe parent, same `velocity + pure_latent_spatial + semantic_topology_gate` family
- adds only a conservative tokenizer-guided output appearance head

## Why This Exists

- the active `vel_tok32_safe_semantic_topogate_k085` packet already recovered the old all-pairs safe shelf on `epoch_0001`
- but transfer style still trails the formal shelf slightly even though LPIPS is much cleaner than the formal lane
- that pattern raises a narrower question than another structure-family swap:
  - is the remaining shortfall partly low-order appearance mismatch
  - such as brightness / contrast / exposure statistics
  - rather than missing structure routing
- this packet tests that question directly without changing:
  - the tokenizer family
  - the solver family
  - the topology-gated structure mechanism

## Config

- config:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json)
- comparison parent:
  - `vel_tok32_safe_rescan_r2 epoch_0004`
  - transfer `0.672377 / 0.369065`
  - all-pairs `0.700490 / 0.367229`
- active sibling reference:
  - [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md)

## Deltas

- keep:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `proximal_mode = crossattn_texture`
  - `batch_size = 16`
  - `accumulation_steps = 1`
- appearance head:
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
  - `output_appearance_hidden_dim = 96`
  - `output_appearance_log_scale_span = ln(1.2)`
  - `output_appearance_shift_span = 0.2`
  - `output_appearance_blend = 0.75`
  - `output_appearance_use_spatial_stats = true`
  - `output_appearance_use_gate_mask_stats = true`

## Intended Read

- success:
  - transfer style rises while LPIPS stays near the active topology-gate band
  - all-pairs shelf recovery is retained
  - new appearance metrics show nontrivial but still small corrections
  - tokenizer observability remains live rather than collapsing
- failure:
  - no measurable style lift relative to the active sibling
  - LPIPS drifts upward enough to erase the safe-band gain
  - the head learns large unstable corrections instead of small low-order alignment

## Queue Position

- this packet is queued directly behind the active `vel_tok32_safe_semantic_topogate_k085` run
- it should be tried before `safe_pnp_selfinject`
- reason:
  - it is a cleaner attribution test
  - and it costs much less architectural disturbance than changing the attention family
- if it also closes without a formal shelf break, the next same-lane successor is:
  - [2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md)
