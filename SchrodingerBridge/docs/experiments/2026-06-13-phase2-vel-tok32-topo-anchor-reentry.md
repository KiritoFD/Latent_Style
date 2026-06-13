# Phase 2: vel_tok32_topo_anchor_reentry

Date: 2026-06-13

## Role

- queued structure-side candidate after the safe-family tokenizer rescan
- not launched yet
- no longer the immediate next formal packet
- only becomes eligible if the safe-band tokenizer family still cannot break the shelf

## Why This Exists

- the refreshed tokenizer line has now shown that:
  - it can stay inside the formal `LPIPS < 0.40` band
  - it can modestly improve the old safe velocity shelf on LPIPS
  - but it still has not pushed style into the `0.72` zone
- the earlier `velocity + topology anchor` retry was too early in the theory stack:
  - it used the older pure-tokenizer path
  - it crossed into archival-only territory at `epoch_0002`
- this packet asks the cleaner follow-up question:
  - what happens if the refreshed tokenizer is kept, but part of the safety budget is moved from pure kinetic suppression into direct latent topology anchors?

## Config

- config:
  - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
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
  - `w_kinetic: 1.0 -> 0.75`
  - `w_content_lowpass_anchor = 0.25`
  - `w_content_edge_anchor = 0.10`
  - `content_anchor_lowpass_kernel = 9`
- queued warm start:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1/epoch_0004.pt`

## Intended Read

- if the safe-family rescan still fails to break the shelf while remaining safe, this becomes the next structure-side packet to try
- success condition:
  - style rises meaningfully above the current `0.701161 / 0.374695` shelf
  - without crossing into `LPIPS >= 0.40`
- failure condition:
  - repeats the older velocity-topology behavior and leaks into archival-only territory

## Launch Rule

- do not launch while `vel_tok32_pos_refresh` still owns the only formal lane
- before launch:
  - replace `resume_checkpoint` with the best currently closed in-band parent if a better checkpoint than `epoch_0004` emerges
  - rerun local smoke against the exact final JSON

## Smoke

- local synthetic smoke:
  - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1_smoke.json)
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
  - loss read:
    - `loss = 2.315261`
    - `flow = 2.067408`
    - `terminal_swd = 0.009933`
    - `t_mean = 0.456406`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.004504`
