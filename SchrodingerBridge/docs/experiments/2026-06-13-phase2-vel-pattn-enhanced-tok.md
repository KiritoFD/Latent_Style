# Phase 2: vel_pattn_enhanced_tok

Date: 2026-06-13

## Goal

- follow the `612-phase2` structure-first pivot
- keep the pure-latent tokenizer path
- return to `velocity` as the transport target
- combine:
  - enhanced `PureLatentSpatialTokenizer`
  - `manifold_adaptive_split`
  - `crossattn_texture`
- target board:
  - style `>= 0.72`
  - LPIPS `<= 0.30`

## Why This Packet

- `rtfix epoch_0001` proved the corrected true-I2SB runtime can raise style, but it failed the new structure gate:
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- `612-lookback` says the main bottleneck is still:
  - endpoint predicts `x_1` and behaves like repainting
  - velocity predicts delta and is the cleaner edit surface
- this packet therefore keeps:
  - pure latent tokenizer
  - proximal texture refinement
  - manifold-aware motion regularization
- but drops:
  - endpoint mainline training
  - SDE / I2SB as the active Distinct5 training lane

## Config

- config:
  - [phase2_vel_pattn_enhanced_tok_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b8a2.json)
- key settings:
  - `tokenizer_family = pure_latent_spatial`
  - `tokenizer_num_clusters = 32`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
  - `w_kinetic = 1.0`
  - `structure_penalty_mode = off`

## Smoke

- local synthetic smoke:
  - status `ok`
  - tokenizer family `pure_latent_spatial`
  - transport mode `velocity`
  - solver `euler_legacy`
  - no DINO runtime required
- important compatibility check:
  - pure-latent tokenizer now supports `crossattn_texture` proximal refinement without falling back to legacy `style_spatial_id_16`

## Gates

- fail-stop:
  - `content_lpips >= 0.70`
- non-promotable:
  - `0.40 <= content_lpips < 0.70`
- only lines with:
  - `content_lpips < 0.40`
  - remain eligible for the remote main lane

## Run Log

- remote status:
  - initial `b8/a2` calibration launched
  - 20s health check:
    - `5917 MiB`
    - too far under-band
  - after another 150s:
    - still `5917 MiB`
    - no late warmup into the formal band
  - training itself did run and stayed numerically stable, but the packet is a calibration miss
  - relaunch decision:
    - move to `batch=16`
    - move to `accumulation_steps=1`
    - keep effective batch at `16`
    - target the formal `9.x-10.x GiB` band directly
  - corrected `b16/a1` relaunch:
    - 20s health check:
      - `10639 MiB`
    - current state:
      - remote lane accepted as formal in-band training
      - live log is in `Epoch 1/24`
    - watchpoint:
      - current velocity objective still reports `t=1.000`
      - if early eval shows the same old style ceiling or no LPIPS benefit, the next adjustment is the delayed-noise / sampled-time variant from `612-phase2`
