# `XPred + K_manifold + P_attn + Queue` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json)

Intent:

- keep the strongest current family:
  - endpoint prediction
  - manifold-adaptive kinetic
  - cross-attention proximal refinement
- add only the fixed queue-side smoothing bundle on top
- read this as the corrected `queue variance reduction` test on the strongest current transport-plus-proximal family

Why this candidate exists:

- `QueueSmoothing` alone looks too weak as a standalone frontier line
- the most plausible remaining reading of queue smoothing is that it only matters once the model already has a strong style-generating proximal branch
- this packet is therefore the cleanest test of:
  - “does queue-side variance reduction help the current best family lower LPIPS without giving back the style band?”

Success condition:

- transfer style stays in the promoted `Pattn / late-Stokes` band
- LPIPS improves materially relative to the current balanced frontier
- the packet avoids the style collapse seen in the standalone queue control

Failure condition:

- style drops back toward the weak single-mechanism queue regime
- or LPIPS does not improve relative to the promoted `Pattn` family
- or the queue bundle simply acts as a no-op once `P_attn` is already present

Reflection template:

- does queue-side smoothing become useful only on top of the strong proximal family?
- does it improve LPIPS enough to matter without surrendering the current style frontier?
- if negative, does that imply target-variance control must be coupled to a stronger target geometry mechanism rather than a queue bundle alone?
- if positive, is this the cleanest next promoted family before adding further structure penalties?

Status:

- now active on the remote `3060`
- automatic launcher path:
  - `run_inmortal_remaining_queue_v3`

## Live status

Remote run:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_queue_seed42_b16`

Current read:

- the queue advanced into this packet automatically after `QueueSmoothing` closed
- first-health is machine-safe:
  - recent remote `nvidia-smi` reads are around `9.2 / 12.3 GiB`
- early trainer peaks are below the current explosion line:
  - `cuda_peak_allocated_gb = 6.77`
  - `cuda_peak_reserved_gb = 8.04`
- checkpoints currently present through at least `epoch_0002.pt`

## Early training read

Latest training rows currently available:

| epoch | loss | flow | kinetic | terminal SWD |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `7.6470` | `0.4425` | `0.6282` | `6.7188` |
| `e2` | `6.2271` | `0.3629` | `0.4987` | `5.8750` |

Current interim interpretation:

- optimization pressure is substantially cleaner than the standalone queue control
- this is the strongest sign so far that the queue bundle is much more plausible on top of the promoted `P_attn` family than by itself
- no paper-safe metric claim yet:
  - wait for the deferred `CLIP-S / LPIPS` surface

## Early eval readout

Available transfer points so far:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6682` | `0.7599` |
| `e2` | `0.7092` | `0.7231` |
| `e3` | `0.7143` | `0.6730` |
| `e4` | `0.7205` | `0.6871` |
| `e5` | `0.7250` | `0.6582` |

Current partial interpretation:

- the early curve is not yet a frontier signal
- relative to the corresponding earlier `P_attn` family:
  - `P_attn e1 = 0.6617 / 0.7574`
  - `P_attn e2 = 0.7160 / 0.7247`
  - `P_attn + Queue e1 = 0.6682 / 0.7599`
  - `P_attn + Queue e2 = 0.7092 / 0.7231`
- this means:
- `e1` is only a tiny style bump with slightly worse LPIPS
- `e2` improves LPIPS slightly relative to plain `P_attn e2`, but gives back noticeable style
- later retained points do raise style again, but LPIPS remains well above the promoted frontier band
- by `e5` the packet is still behind the current mainline comparisons:
  - `P_attn e5 = 0.7271 / 0.6559`
  - `P_attn + Queue e5 = 0.7250 / 0.6582`
  - `late Stokes balanced best = 0.7274 / 0.6033`
- so the current reading is:
  - not negative enough to kill
  - not strong enough to promote
  - still worth waiting for later retained points, because this family historically improves late

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6682` | `0.7599` |
| `e2` | `0.7092` | `0.7231` |
| `e3` | `0.7143` | `0.6730` |
| `e4` | `0.7205` | `0.6871` |
| `e5` | `0.7250` | `0.6582` |
| `e6` | `0.7278` | `0.6359` |
| `e7` | `0.7241` | `0.6317` |
| `e8` | `0.7240` | `0.6138` |

Best retained point:

- `e6`
  - transfer `clip_style = 0.7278`
  - transfer `content_lpips = 0.6359`
  - full `clip_style = 0.7327`
  - full `content_lpips = 0.6270`

Final epoch:

- `e8`
  - transfer `clip_style = 0.7240`
  - transfer `content_lpips = 0.6138`
  - full `clip_style = 0.7303`
  - full `content_lpips = 0.6042`

Closure interpretation:

- this packet is not a failure, but it is not a frontier improvement either
- relative to plain `P_attn`, it stays in the same broad style band and only gets modest late LPIPS recovery
- relative to the promoted balanced frontier:
  - `late Stokes balanced best = 0.7274 / 0.6033`
  - `P_attn + Queue best = 0.7278 / 0.6359`
- so queue-side smoothing on top of `Kmanifold + P_attn` does not convert into a better tradeoff frontier
- the strongest paper-safe reading is:
  - queue smoothing alone was a positive secondary mechanism
  - queue smoothing on top of the strong `P_attn` family is mostly neutral-to-weak
  - the remaining upside likely sits in the current `AnisoStokesQueue` continuation rather than another plain queue rerun
