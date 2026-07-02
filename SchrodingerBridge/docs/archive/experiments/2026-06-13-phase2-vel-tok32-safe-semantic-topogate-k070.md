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
  - `training_after_settled_eval`
  - remote run name:
    - `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1`
  - warm start:
    - `appalign epoch_0001`
- first expectation:
  - if style can move upward while LPIPS stays close to the `0.31-0.34` band,
    this branch becomes the first post-appalign style-release proof without abandoning the true-tokenizer + velocity family

## Current Read

- latest settled checkpoint:
  - `epoch_0002`
  - transfer `0.671814 / 0.315878`
  - all-pairs `0.703409 / 0.313567`
- read:
  - the branch is still clearly in-band
  - LPIPS continued to improve materially from `epoch_0001`
  - all-pairs remains above the safe recovery shelf
  - transfer style drifted down slightly instead of up
- interpretation:
  - this is stronger evidence that `k070` is buying structure cleanliness rather than a style breakout
  - the line should keep running until the close gate is actually met
  - if later checkpoints keep the same pattern, the next style-facing reads are:
    - `k070_kin070`
    - then the velocity-native stochastic eval-only `k070 e1` SDE probe
    - then `k070_sp256`

## 2026-06-14 All-Ckpt Read

- retained checkpoints with remote full eval:
  - `epoch_0001`: transfer `0.672664 / 0.336344`, all-pairs `0.703589 / 0.333097`, train `25.39 min`
  - `epoch_0002`: transfer `0.671814 / 0.315878`, all-pairs `0.703409 / 0.313567`, train `23.96 min`
  - `epoch_0003`: transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`, train `23.86 min`
  - `epoch_0004`: transfer `0.669782 / 0.323409`, all-pairs `0.701260 / 0.319439`, train `23.82 min`
  - `epoch_0005`: transfer `0.671104 / 0.325637`, all-pairs `0.702300 / 0.322536`, train `23.84 min`
- best transfer style remains `epoch_0001`
- best structure-preserving point is `epoch_0003` by transfer LPIPS and all-pairs LPIPS
- convergence state:
  - best checkpoint is not in the newest two retained checkpoints
  - last Pareto point is `epoch_0003`
  - tail is near-flat, but the formal close gate was not completed because `epoch_0006` was stopped by the runtime VRAM guard
- guard event:
  - `2026-06-14T01:39:29+08:00`
  - observed `11760 MiB` against cap `11000 MiB`
  - process exited with `rc=143`
- plot outputs:
  - raw curve: [k070_epoch1_5_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/k070_epoch1_5_remote_clip_lpips_curve.csv)
  - page-1 plot table: [plot_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/plot_points.csv)
  - AAAI2027 page-1 figure: [fig_distinct5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary.png)
- decision:
  - do not treat `k070` as a style breakthrough
  - use it as an in-band structure/LPIPS parent or matched deterministic control
  - next style-facing mechanism should be eval-only stochasticity before more training-side capacity changes

## 2026-06-14 Queue Closure

- Queue status updated to `closed_inband_no_style`.
- Reason: the run already has all-ckpt `CLIP-S + LPIPS` coverage for epochs `1-5`; `epoch_0006` was stopped by the runtime guard and should not be resumed just to complete a slow long-training tail.
- Operational decision: do not auto-relaunch this lane. Use `epoch_0003` as the deterministic in-band control/parent for eval-only probes, and only revisit training-side style release with a shorter virtual-length schedule.
