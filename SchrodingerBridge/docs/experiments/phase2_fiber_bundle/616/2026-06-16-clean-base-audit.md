# 616 Clean-Base Audit

Date: 2026-06-16

## Purpose

This note fixes the experimental interpretation boundary for the current 616 work.

The active `phase616_ot_vertical_scratch_b8a2_e24` lane is the first formal 616 foundation run, but it is not yet a fully purged code-path base. It is a controlled OT and target-geometry intervention on top of the retained `k070 -> k085_appalign` family.

## What The Current 616 Lane Already Turns Off

The resolved config for the live OT lane confirms:

- `style_delta_mode = none`
- `proximal_mode = off`
- `w_content_lowpass_anchor = 0.0`
- `w_content_edge_anchor = 0.0`
- `cycle_consistency_weight = 0.0`

This means the current live lane is already free of the most explicit late-added style-delta and proximal residual branches, and it does not rely on content-anchor or cycle losses.

## What The Current 616 Lane Still Inherits

The same resolved config also confirms:

- `output_appearance_alignment_mode = tokenizer_latent_affine`
- `semantic_supervision_family = legacy_terminal_swd`
- `w_kinetic = 0.85`
- `backbone_attention_family = legacy_semantic_crossattn`
- `tokenizer_family = pure_latent_spatial`

Interpretation:

- positive or negative results from this lane are evidence about `structure-aware OT + sinkhorn_unbalanced + pure_vertical_flow`
- they are not yet evidence about a fully purged 616 architecture
- in particular, appearance alignment and the retained legacy semantic supervision path can still affect the style/structure tradeoff

## Codebase Paths Still Present But Not Part Of The 616 Contract

The codebase still contains a wider historical surface than the current 616 contract wants to expose.

Examples already identified during audit:

- `losses.py`
  - `_cycle_consistency_loss`
  - `_content_topology_anchor_loss`
  - `proximal_trust_penalty`
  - additional legacy regularizers not used by the current lane
- `model.py`
  - `style_delta_mode = basis | predec_section | head_adapter`
  - `proximal_mode = crossattn_texture`
- `lancet_runtime.py` / `lancet_backbone.py`
  - output appearance alignment path
- evaluation/runtime config
  - latent postprocess and appearance-oriented postprocess branches remain in the general surface

These branches should not be confused with the live 616 mechanism conclusion just because they still exist in the repository.

## Clean-Base Target For The Next Refactor Step

The 616 purge target is:

1. Keep the current OT and target-geometry instrumentation intact.
2. Reduce the active training surface to the smallest path needed for:
   - endpoint I2SB
   - retained backbone/tokenizer base
   - structure-aware OT
   - pure vertical target projection
3. De-expose or remove dormant heuristic branches that are outside the current 616 contract.

Practical order:

1. finish the live OT foundation lane and keep its evidence stable
2. run the matched high-VRAM throughput probe on the same mechanism contract
3. only after that, cut a smaller clean base for the next formal 616 mechanism lane

Clean-base artifacts now prepared locally:

- [phase616_cleanbase_i2sb_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_cleanbase_i2sb_k085_b8a2_e24.json)
- [phase616_clean_ot_vertical_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_vertical_k085_b8a2_e24.json)
- [phase616_clean_vertical_target_source_low_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_source_low_k085_b8a2_e24.json)
- [phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json)
- [clean_base_round1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_base_round1.md)
- [run_phase616_clean_ot_vertical_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_vertical_round1.sh)
- [clean_vertical_target_probe_round1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_vertical_target_probe_round1.md)
- [run_phase616_clean_vertical_target_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_vertical_target_probe_round1.sh)

These do not replace the current live OT lane. They define the first reproducible, appalign-free 616 launch surface for the next formal clean-base run.

## Audit Decision

Status labels for the current moment:

- `retained for live evidence`
  - `sinkhorn_unbalanced`
  - `coupling_structure_cost_mode = lowedge`
  - `training_target_projection_mode = pure_vertical_flow`
  - runtime and GPU observability
- `retained legacy base dependency`
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
  - `legacy_terminal_swd`
  - `legacy_semantic_crossattn`
- `disabled in the live lane`
  - `style_delta_mode`
  - `proximal_mode`
  - content anchors
  - cycle consistency
- `present in repo, candidate for later purge or de-exposure`
  - style-delta historical branches
  - proximal historical branches
  - dormant heuristic regularizers
  - latent postprocess branches not needed for the 616 main line
