# Phase 2: vel_tok32_safe_semantic_topogate_k070_sp256

Date: 2026-06-13

## Role

- queued tokenizer-capacity follow-on behind `k070`
- keep the same true-tokenizer + velocity + topology-gate family
- adopt the remaining low-risk guide suggestion that has not yet been tried directly:
  - `tokenizer_spatial_dim: body_channels -> 256`

## Why This Exists

- the updated guide's `Query Extractor + Positional Encoding` suggestion is already largely present in the current `pure_latent_spatial` stack:
  - deeper ResBlock query extractor
  - 2D sinusoidal positional encoding
  - pooled global-spatial coupling
- the truly untried part is wider tokenizer spatial value capacity
- this packet therefore keeps the current recovered-family structure controls and only raises tokenizer spatial width

## Config

- config:
  - [phase2_vel_tok32_safe_semantic_topogate_k070_sp256_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k070_sp256_seed42_b12a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json)

## Deltas

- keep:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 0.7`
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
- change:
  - `tokenizer_spatial_dim: default(body_channels=128) -> 256`
  - runtime bridge:
    - project the widened tokenizer spatial map back to `body_channels` through a `1x1` compatibility projection
    - default initialization is channel-preserving on the overlapping prefix so old checkpoints remain safe to resume with `strict=false`

## Intended Read

- success:
  - style rises while LPIPS stays in the recovered `0.31-0.34` band
  - tokenizer observability shows nontrivial routing with the wider spatial state
- failure:
  - no style lift relative to `k070`
  - or the wider tokenizer map simply adds instability without moving the Pareto front

## Remote Smoke Read

- execution surface:
  - remote WSL CPU smoke via [launch_remote_wsl_command.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py)
  - log:
    - `I:\Github\Latent_Style\exp\inmortal-exp\phase2_k070_sp256_cpu_smoke.log`
  - output:
    - `I:\Github\Latent_Style\SchrodingerBridge\_codex_tmp\phase2_k070_sp256_remote_smoke.json`
- result:
  - status `ok`
  - forward / endpoint / integrated shapes all preserved at `[1, 4, 16, 16]`
  - tokenizer debug confirms:
    - `spatial_dim = 256`
    - `spatial_map_channels_raw = 256`
    - `spatial_map_channels_out = 128`
    - `spatial_map_proj_active = 1.0`
  - first gradient landed on:
    - `structured_style_map_proj.weight`
- interpretation:
  - the widened tokenizer path is implemented
  - the compatibility projection is live on the remote machine
  - old-shape downstream consumers do not need to be rewritten before this packet can be trained
