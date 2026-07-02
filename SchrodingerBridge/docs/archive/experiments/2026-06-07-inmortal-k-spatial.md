# `K_spatial` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: `H-family` remote `3060 WSL`
- config:
  - [inmortal_k_spatial_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spatial_seed42_b44.json)
  - retry:
    - [inmortal_k_spatial_seed42_b32.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spatial_seed42_b32.json)

Intent:

- activate time smoothing via `w_curvature`
- replace the dominant global kinetic reading with a spatial low/high-frequency split
- test whether low-frequency motion can be suppressed without killing style growth

Expected upside:

- stronger texture generation than the current trivial low-energy field
- less pressure toward no-op than uniform global `L2`

Expected failure mode:

- style may increase while structure becomes locally mushy because this packet does not yet protect edges with anisotropic or manifold-aware penalties

Reflection template:

- did transfer `CLIP-style` rise above the current `H-family` frontier?
- did `LPIPS` explode or stay near the reviewed operating band?
- did `base_endpoint_abs` decrease while `final_endpoint_abs` remain usable?
- does the run suggest that `K_spatial` should be combined next with anisotropic structure protection or with proximal high-pass refinement?

## Launch anomaly

The first formal `b44` launch is not a valid evidence packet.

Observed failure:

- remote `3060` memory climbed to about `12.0 / 12.3 GiB`
- this violated the current hard formal cap `< 11.0 GiB`
- the lane was stopped immediately instead of letting it continue under a broken machine contract

Immediate corrective action:

- keep the mechanism unchanged
- reduce only the formal batch size
- relaunch as `b32`

Interpretation:

- this is an execution-surface correction, not a mechanism verdict
- no quality conclusion should be drawn from the aborted `b44` attempt

## Root-cause update after `b16` probe

The first stable probe shows the earlier VRAM jump was **not** caused by the spatial low/high-frequency kinetic split itself.

Current reading:

- `kinetic_penalty_mode = spatial_laplacian_split` adds only a light low/high-frequency decomposition around `pred_velocity`
- the dominant new memory cost comes from the newly activated `w_curvature = 1.0`
- in the current `OMF` path, curvature is implemented as extra forward passes that retain additional computation graphs for backpropagation

Operational implication:

- future `time-smoothing / curvature` packets should start from a clearly smaller batch than the legacy `H-family` batch-44 surface
- this is still a mechanism-preserving correction, not a reason to discard the line

## Closed readout

Remote packet:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_k_spatial_seed42_b16`

Training surface:

- batch: `16`
- epochs: `8`
- peak memory recorded by the trainer at `epoch_0008`:
  - allocated about `8.04 GB`
  - reserved about `8.45 GB`

Transfer curve:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6605` | `0.3425` |
| `e2` | `0.6570` | `0.3248` |
| `e3` | `0.6599` | `0.3433` |
| `e4` | `0.6570` | `0.3485` |
| `e5` | `0.6594` | `0.3691` |
| `e6` | `0.6615` | `0.3504` |
| `e7` | `0.6601` | `0.3468` |
| `e8` | `0.6606` | `0.3640` |

Reading:

- the line is numerically stable and clearly trainable once the batch is reduced
- the mechanism does **not** raise the current style ceiling
- the best raw style point is `e6`, but it remains below the reviewed compact frontier
- the best LPIPS point is `e2`, but the style trade-off remains weaker than the current best compact frontier

Mechanism interpretation:

- replacing uniform kinetic with the spatial low/high split is **not enough by itself**
- it does not unlock a stronger style regime on `Distinct5-512`
- this supports the stronger theory claim that the ceiling problem is not only “wrong kinetic geometry”
- the next mechanisms should emphasize either:
  - stronger structure-aware release (`K_manifold`), or
  - transport-target redesign (`XPred_Barycenter`), or
  - explicit proximal refinement (`P_highpass`)

Status:

- `K_spatial` is a closed negative-to-neutral mechanism packet
- keep it as evidence, but do not promote it as the next headline line
