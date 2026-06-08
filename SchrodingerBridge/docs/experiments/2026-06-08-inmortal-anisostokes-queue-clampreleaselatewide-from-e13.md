# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Late Wider ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleaselatewide_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleaselatewide_reseed_from_e13_seed42_b8a2.json)

Intent:

- keep the successful `1.10` early clamp from the first positive release packet
- avoid the negative `1.25 -> 1.60 / 4 epochs` wide-release regime
- test whether style recovery needs a later and slower release rather than a looser early basin

Why this candidate exists:

- the first release packet showed that the tighter `1.10` early squeeze materially improves LPIPS
- the wider-release packet showed that removing that squeeze is harmful
- the remaining open question is:
  - can we keep the good early basin
  - but still recover more style later by releasing farther and more slowly

Mechanism:

- start clamp ratio at `1.10`
- linearly relax to `1.60`
- release over the first `10` epochs

Success condition:

- match or beat the first release packet's low-LPIPS point:
  - `e3 = 0.7007 / 0.4754`
- while recovering style later in training without reopening the `e14`-style proximal takeover failure mode

Failure condition:

- the later wider release still fails to recover style
- or it reintroduces the same late proximal domination that destroyed the parent continuation after `e13`

Outcome:

- training completed through `epoch_0016`
- full eval completed through `epoch_0016`

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7064` | `0.5175` | `0.7206` | `0.5111` |
| `e2` | `0.7005` | `0.5399` | `0.7131` | `0.5327` |
| `e3` | `0.7014` | `0.4768` | `0.7188` | `0.4716` |
| `e4` | `0.6963` | `0.5212` | `0.7103` | `0.5151` |
| `e5` | `0.6946` | `0.5128` | `0.7096` | `0.5069` |
| `e6` | `0.6893` | `0.5165` | `0.7056` | `0.5083` |
| `e7` | `0.6929` | `0.4885` | `0.7109` | `0.4803` |
| `e8` | `0.6881` | `0.5113` | `0.7037` | `0.5033` |
| `e9` | `0.6899` | `0.4871` | `0.7078` | `0.4803` |
| `e10` | `0.6871` | `0.5048` | `0.7036` | `0.4965` |
| `e11` | `0.6853` | `0.4952` | `0.7018` | `0.4882` |
| `e12` | `0.6857` | `0.4905` | `0.7033` | `0.4831` |
| `e13` | `0.6898` | `0.4754` | `0.7083` | `0.4686` |
| `e14` | `0.6845` | `0.4949` | `0.7021` | `0.4874` |
| `e15` | `0.6839` | `0.4908` | `0.7017` | `0.4833` |
| `e16` | `0.6838` | `0.4951` | `0.7013` | `0.4874` |

Best retained points:

- best transfer-style:
  - `e1 = 0.7064 / 0.5175`
  - all-pairs `= 0.7206 / 0.5111`
- best LPIPS under `transfer >= 0.70`:
  - `e3 = 0.7014 / 0.4768`
  - all-pairs `= 0.7188 / 0.4716`

Training-side read:

- this packet keeps the good early basin much better than the failed `wide release`
- but because the schedule is still purely linear, it eventually walks all the way to the wide cap:
  - `proximal_to_transport_ratio` reaches about `1.60` by `e11`
  - then remains pinned there through convergence
- the family therefore recovers almost all of the first release packet's quality, but does not actually beat it on the transfer frontier

Mechanism conclusion:

- slowing the release is directionally correct
- preserving the early `1.10` squeeze matters much more than widening the late ceiling
- a slow linear release to `1.60` can nearly match the positive `clamp-release` packet:
  - first release best `= 0.7007 / 0.4754`
  - late-wide best `= 0.7014 / 0.4768`
- but it still gives back a small amount of LPIPS
- this strongly supports the next step:
  - explicit hold-then-release
  - instead of another purely linear schedule

Decision:

- near-tie negative closure
- not a promoted frontier under the current rule
- retain as a useful mechanism packet because it narrows the gap substantially and shows that the main missing ingredient is the explicit early hold phase
