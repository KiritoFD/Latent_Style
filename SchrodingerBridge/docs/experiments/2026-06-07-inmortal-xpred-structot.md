# `XPred + StructOT` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_structot_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_structot_seed42_b16.json)

Intent:

- isolate the contribution of structure-aware OT
- keep endpoint prediction active
- disable:
  - barycentric target smoothing
  - EMA teacher support
- preserve the lowfreq-edge coupling cost so the packet reads as:
  - `XPred + StructOT`, not `XPred + Barycentric`, and not `XPred + Teacher`

Expected upside:

- lower target-assignment variance than naive raw matching
- better content retention than `XPred_Barycenter` at the same style band
- potential improvement over the current `XPred` family if structure-aware matching is the missing stabilizer

Expected failure mode:

- the line may remain too close to the current `sample` target regime
- if so, style might stay muted relative to the promoted `Pattn/Stokes` family
- or the packet may keep LPIPS reasonable without actually raising the frontier

Reflection template:

- does `StructOT` improve LPIPS relative to `XPred_Barycenter` without collapsing style?
- does it outperform the pure kinetic controls in a way that supports the transport-target story?
- if the packet is negative, is that because:
  - structure-aware cost is too weak alone, or
  - barycentric smoothing is still the dominant driver?
- if the packet is positive, should the next combination be:
  - `StructOT + Pattn`, or
  - `StructOT + Queue`, or
  - `StructOT + Teacher`

## Live status

Remote run:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_structot_seed42_b16`

Execution chain:

- queue runner:
  - `run_inmortal_remaining_queue_v3`
- summary refresh helper:
  - `structot-refresh-after-exit`

Current read:

- the queue advanced into `StructOT` automatically after `K_spectral b12` fast-eval closure
- training completed its `8` epochs and produced checkpoints through `epoch_0008.pt`
- the training-owned deferred `CLIP-S / LPIPS` eval has now produced the full `e1-e8` readout under `full_eval/`
- first-health on the remote `3060` stayed safely below the current machine cap
- runtime memory stayed comfortably below the ceiling during training
  - trainer peaks stayed around `2.84 / 3.16 GB`

## Partial readout

Full transfer curve:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6446` | `0.7587` |
| `e2` | `0.6926` | `0.6480` |
| `e3` | `0.7057` | `0.6300` |
| `e4` | `0.7090` | `0.6055` |
| `e5` | `0.7190` | `0.5589` |
| `e6` | `0.7131` | `0.5637` |
| `e7` | `0.7132` | `0.5598` |
| `e8` | `0.7103` | `0.5577` |

Best retained point:

- `e5`
  - transfer `clip_style = 0.7190`
  - transfer `content_lpips = 0.5589`
  - full `clip_style = 0.7302`
  - full `content_lpips = 0.5510`

Interpretation:

- the line is clearly stronger than the trivial `e1` start and keeps improving through the first half of training
- `StructOT` does produce a meaningful low-LPIPS tradeoff point by `e5`
- later epochs do not improve the frontier:
  - style falls back after `e5`
  - LPIPS only improves marginally by `e8`
  - that marginal LPIPS improvement is not enough to offset the style loss under the current promotion rule
- compared against the current headline families:
  - it is much better than the weak kinetic-only controls
  - it is clearly better than early failed `XPred` variants
  - but it still trails the promoted `Pattn/Stokes` family on the actual frontier

Mechanism conclusion:

- `StructOT` is a positive mechanism signal, but a secondary one
- it supports the theory that better target assignment matters
- however, structure-aware OT alone is not the dominant source of the final ceiling jump
- the strongest reading is:
  - `StructOT` is useful as a stabilizing tradeoff ingredient
  - but not sufficient, by itself, to replace the current promoted `Pattn/Stokes` line
- the most credible next combination is now:
  - `StructOT + Pattn`, or
  - `StructOT + Queue`, or
  - `StructOT + Teacher`

Pending closure:

- normalized retained checkpoint curve on the shared snapshot surface if we still want strict surface uniformity
- follow-on combination packets to test whether `StructOT` becomes decisive only when paired with the stronger proximal/queue families
