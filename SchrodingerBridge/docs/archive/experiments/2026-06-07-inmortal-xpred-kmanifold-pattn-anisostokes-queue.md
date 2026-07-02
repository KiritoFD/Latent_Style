# `XPred + K_manifold + P_attn + AnisoStokes + Queue` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json)

Intent:

- resume from the stronger `P_attn` continuation family
- then add:
  - anisotropic structure regularization
  - weak Stokes smoothing
  - the fixed queue-side smoothing bundle
- this is the strongest currently queued combination arm and is the closest operational realization of the corrected `C6` spirit

Why this candidate exists:

- `Aniso` alone helped LPIPS but strangled style too hard
- weak `Stokes` helped the `P_attn` family hold a better balance than plain `Aniso`
- queue-side smoothing remains theoretically plausible as a variance-control ingredient
- combining these on top of the already-promoted family is the highest-ceiling queued attempt that still stays inside the current theory family

Success condition:

- keep style in or near the current promoted band
- improve LPIPS beyond the current balanced frontier
- avoid reverting into the weak standalone queue regime

Failure condition:

- style is over-constrained again, as in the earlier `Aniso` packet
- LPIPS gain is too small to justify the added complexity
- or the combined regularizers simply smooth away the style benefit

Reflection template:

- does `Aniso + Stokes + Queue` finally convert the extra structure pressure into a real frontier improvement?
- is the queue bundle helping the strong family, or merely adding more conservative drift?
- if this still fails, does that mean the current remaining ceiling debt is no longer target variance, but instead proximal texture expressivity?

Status:

- remote packet completed and closed
- launch advanced automatically after:
  1. standalone `QueueSmoothing`
  2. `Kmanifold + Pattn + Queue`
- queue runner:
  - `run_inmortal_remaining_queue_v3`

## Live status

Remote run:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2`

Current read:

- this packet resumed from the promoted `Pattn` continuation checkpoint as intended
- training batch was `8`
- the earlier over-cap incident was not caused by this line itself
  - it came from stale baseline processes overlapping the formal lane
  - those stale baseline processes were killed
  - the lane then completed cleanly under single-run operation
- final checkpoints and eval summaries exist through `epoch_0016`

## Early training read

Latest training rows currently available:

| epoch | loss | flow | kinetic | terminal SWD |
| --- | ---: | ---: | ---: | ---: |
| `e12` | `9.6552` | `0.7694` | `0.1341` | `5.4688` |
| `e13` | `9.2731` | `0.8198` | `0.0969` | `5.6875` |

Current interim interpretation:

- the resumed strong-family combo remained numerically healthy end-to-end
- transport energy stayed very low relative to the earlier queue-only lines

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e12` | `0.6856` | `0.4213` | `0.7128` | `0.4150` |
| `e13` | `0.7102` | `0.4603` | `0.7303` | `0.4559` |
| `e14` | `0.6945` | `0.5774` | `0.6999` | `0.5732` |
| `e15` | `0.6934` | `0.5803` | `0.6984` | `0.5757` |
| `e16` | `0.6934` | `0.5954` | `0.6985` | `0.5908` |

Best retained point:

- `e13`
  - transfer `clip_style = 0.7102`
  - transfer `content_lpips = 0.4603`
  - all-pairs `clip_style = 0.7303`
  - all-pairs `content_lpips = 0.4559`
  - identity `clip_style = 0.8108`
  - identity `content_lpips = 0.4384`

Final epoch:

- `e16`
  - transfer `clip_style = 0.6934`
  - transfer `content_lpips = 0.5954`

Closure interpretation:

- this packet does not improve the main style frontier
- but it becomes the strongest low-LPIPS successor point in the current `inmortal` surface
- relative to the earlier secondary lines:
  - `EndpointTeacher e5 = 0.7159 / 0.5555`
  - `StructOT e5 = 0.7190 / 0.5589`
  - `QueueSmoothing e5 = 0.7198 / 0.5595`
  - `AnisoStokesQueue e13 = 0.7102 / 0.4603`
- so this line gives back style, but buys a very large LPIPS improvement
- against the user-provided ideal transfer reference:
  - ideal transfer `0.6920 / 0.4923`
  - this packet beats that LPIPS target clearly, and slightly exceeds that transfer style reference as well
- the line is therefore paper-safe as:
  - the new low-LPIPS successor point
  - not the new headline frontier
- an independent confirmatory rerun on the same remote eval surface reproduced the selected point almost exactly:
  - rerun `e13 transfer = 0.7101 / 0.4604`
  - rerun `e13 all-pairs = 0.7302 / 0.4560`
- relative to the repaired `Seedream 4.5` Distinct5 package on the same metric surface:
  - `Seedream transfer = 0.6920 / 0.4923`
  - `Seedream all-pairs = 0.7198 / 0.4767`
  - `Seedream identity = 0.8312 / 0.4145`
- so the current reading is:
  - stronger than Seedream on transfer style and transfer LPIPS
  - stronger than Seedream on all-pairs style and all-pairs LPIPS
  - weaker than Seedream on identity preservation

Mechanism conclusion:

- adding `Aniso + Stokes + Queue` on top of the strong family does not raise the style ceiling
- but it does produce the first genuinely strong content-preserving tradeoff point in the successor family
- the resulting interpretation is now cleaner:
  - `late Stokes` remains the better balanced frontier
  - `AnisoStokesQueue` becomes the stronger low-LPIPS anchor

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e12` | `0.6856` | `0.4213` | `0.7128` | `0.4150` |
| `e13` | `0.7102` | `0.4603` | `0.7303` | `0.4559` |
| `e14` | `0.6945` | `0.5774` | `0.6999` | `0.5732` |
| `e15` | `0.6934` | `0.5803` | `0.6984` | `0.5757` |
| `e16` | `0.6934` | `0.5954` | `0.6985` | `0.5908` |

Best retained point:

- `e13`
  - transfer `clip_style = 0.7102`
  - transfer `content_lpips = 0.4603`
  - all-pairs `clip_style = 0.7303`
  - all-pairs `content_lpips = 0.4559`
  - identity `clip_style = 0.8108`
  - identity `content_lpips = 0.4384`

Final epoch:

- `e16`
  - transfer `clip_style = 0.6934`
  - transfer `content_lpips = 0.5954`

Closure interpretation:

- this packet does not improve the main style frontier
- but it is the strongest low-LPIPS tradeoff result in the current `inmortal` surface
- relative to the earlier paper-facing secondary lines:
  - `EndpointTeacher e5 = 0.7159 / 0.5555`
  - `StructOT e5 = 0.7190 / 0.5589`
  - `QueueSmoothing e5 = 0.7198 / 0.5595`
  - `AnisoStokesQueue e13 = 0.7102 / 0.4603`
- so this line gives back style, but buys a very large LPIPS improvement
- against the user-provided ideal transfer reference:
  - ideal transfer `0.6920 / 0.4923`
  - this packet beats that LPIPS target clearly, and slightly exceeds that transfer style reference as well
- the line is therefore paper-safe as:
  - the new low-LPIPS successor point
  - not the new headline frontier

Mechanism conclusion:

- adding `Aniso + Stokes + Queue` on top of the strong family does not raise the style ceiling
- but it does produce the first genuinely strong content-preserving tradeoff point in the successor family
- the resulting interpretation is now cleaner:
  - `late Stokes` remains the better balanced frontier
  - `AnisoStokesQueue` becomes the stronger low-LPIPS anchor
