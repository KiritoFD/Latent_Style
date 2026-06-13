# Phase 2: vel_tok32_safe_semantic_topogate_k070

Date: 2026-06-13

## Role

- guide-aligned next training-side style-lift packet
- keep the current safe tokenizer, velocity transport, and appalign head
- change only one structure-side control:
  - `semantic_self_topology_blend = 0.7`

## Why This Exists

- `appalign` proved that the current family can hold LPIPS near `0.31`
- but by `epoch_0004` the line was still style-limited and had already lost the all-pairs shelf
- the guide read is that the bottleneck is no longer tokenizer capacity or raw structure retention
- the cleanest next training-side hypothesis is:
  - keep the same family
  - keep the recovered parent
  - reduce topology locking slightly so style has more freedom to move

## Config

- config:
  - [phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json)
- selected warm start:
  - `appalign epoch_0001`
  - transfer `0.672604 / 0.336357`
  - all-pairs `0.703506 / 0.332992`

## Deltas

- keep:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `semantic_self_topology_gate = true`
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
  - `output_appearance_blend = 0.75`
- change:
  - `semantic_self_topology_blend: 1.0 -> 0.7`

## Intended Read

- success:
  - transfer style moves back upward without reopening the large LPIPS penalty
  - all-pairs stays near the recovered `0.70x / 0.31x` band
- failure:
  - LPIPS rises quickly with no meaningful style gain
  - or the line simply recreates the same plateau under weaker structure control

## Queue Position

- this is the guide-aligned next training-side packet after:
  - `appalign` closed on in-band style plateau
  - `i2sb_tflooor005` produced an archival-only first settled point
  - `solver_pc` appalign-e3 side probe failed to create a meaningful style lift
- intended role:
  - keep the current true-tokenizer + velocity stack
  - release only part of the topology lock
  - test whether the style bottleneck is caused by over-constrained structure blending rather than missing stochasticity alone

## Launch Read

- current remote state:
  - `training_before_first_settled_eval`
  - remote run name:
    - `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1`
  - warm start:
    - `appalign epoch_0001`
- first expectation:
  - if style can move upward while LPIPS stays close to the `0.31-0.34` band,
    this branch becomes the first post-appalign style-release proof without abandoning the true-tokenizer + velocity family
