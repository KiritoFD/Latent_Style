# `XPred_Barycenter` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: `H-family` remote `3060 WSL`
- config:
  - [inmortal_xpred_bary_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_bary_seed42_b16.json)

Intent:

- test the user-proposed `x-prediction / endpoint prediction` direction
- stop asking the model to regress raw residuals as the primary target
- replace sampled OT target pressure with:
  - structure-aware OT cost
  - barycentric target projection
  - weak low-frequency EMA teacher

Expected upside:

- higher style ceiling than plain residual/velocity prediction
- lower target variance than single sampled OT endpoints
- less drift toward trivial mean residuals

Expected failure mode:

- endpoint prediction may become too coarse and oversmooth style structure
- barycentric target may wash out high-frequency modes if the target teacher or top-k projection dominates too hard

Reflection template:

- does endpoint prediction raise transfer `CLIP-style` faster than the velocity baseline?
- is `base_transfer_clip_style` already useful, or is all quality deferred to later terminal correction?
- does barycentric target smoothing reduce instability without flattening style?
- does the EMA teacher help or over-average the target manifold?

## Batch policy

Observed probe behavior:

- `b16` training peak stayed far below the `3060` ceiling
- therefore `b16` is treated as a mechanism probe, not the final throughput setting

Promoted rerun target:

- [inmortal_xpred_bary_seed42_b40.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_bary_seed42_b40.json)

Intended use:

- if the `b16` result is promising, rerun the line closer to the `~10 GB` target band instead of keeping an under-filled GPU lane

## Early readout

First available retained point:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6914` | `0.7484` |

Immediate interpretation:

- this is the strongest raw style jump seen so far in the `inmortal` program
- it already clears the compact `LANCET` style band on transfer style
- but it does so by collapsing content preservation

Mechanism reading:

- `endpoint prediction + barycentric target` is a real ceiling-raising direction
- but in its current transport-only form it is too coarse and too destructive
- the most likely next useful combination is:
  - `XPred_Barycenter`
  - plus a stronger structure-preserving kinetic family
  - or a constrained proximal high-pass refinement branch

Status:

- keep running the full `e1-e8` eval surface
- treat `e1` as an early positive style-gain checkpoint, not yet a promotable final frontier point
