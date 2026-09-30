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

## Mid-run promoted readout (`b40`)

Higher-batch rerun:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_bary_seed42_b40`

Observed transfer curve through `e6`:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6319` | `0.7813` |
| `e2` | `0.6762` | `0.7843` |
| `e3` | `0.7016` | `0.7656` |
| `e4` | `0.7061` | `0.7507` |
| `e5` | `0.7102` | `0.7264` |
| `e6` | `0.7155` | `0.7241` |

Current reading:

- this line is the first `inmortal` packet that clearly pushes the style ceiling far above the compact baseline band
- the price is still severe content damage
- the mechanism is therefore **positive for ceiling**, but **not yet a usable frontier point**

Next interpretation target:

- finish the `e7/e8` readout
- then decide whether this family should next be:
  - combined with a stronger structure-preserving kinetic packet, or
  - combined with a constrained proximal high-pass branch

## Full `b40` readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6319` | `0.7813` |
| `e2` | `0.6762` | `0.7843` |
| `e3` | `0.7016` | `0.7656` |
| `e4` | `0.7061` | `0.7507` |
| `e5` | `0.7102` | `0.7264` |
| `e6` | `0.7155` | `0.7241` |
| `e7` | `0.7161` | `0.7176` |
| `e8` | `0.7129` | `0.7120` |

Best retained style point:

- `e7`
  - transfer `clip_style = 0.7161`
  - transfer `content_lpips = 0.7176`
  - full `clip_style = 0.7187`
  - full `content_lpips = 0.7104`

Interpretation against the current target:

- this family is now very close to the style half of the `0.72 / 0.30` target
- it is still extremely far from the LPIPS half
- therefore it is a **real style-ceiling success** and a **content-preservation failure**

Mechanism conclusion:

- `x-pred + barycentric target` is the strongest style-raising mechanism found so far
- it should not be dropped
- the next best use of GPU time is to combine this family with a stronger content-preserving mechanism, most likely:
  - `K_manifold`, or
  - `P_highpass`

## Eval speed note

The slow part is not CLIP/LPIPS themselves.

Observed cost profile on this family:

- `eval_metrics_loop`: about `21-23s`
- `summary_grid`: about `23-25s`
- `vae_decode`: about `54s` on the early epochs

This means:

- `summary_grid` is pure overhead for mechanism sweeps
- future sweeps should default to `--no-save_summary_grid`
- `profile_timing` should stay off unless timing is the target of the experiment
