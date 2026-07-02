# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Wider ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleasewide_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleasewide_reseed_from_e13_seed42_b8a2.json)

Intent:

- keep the successful `e13 -> optimizer reset -> hard proximal control` family
- remove the extra early squeeze introduced by the `1.10` start clamp
- release the proximal branch earlier and farther than the first release packet

Why this candidate exists:

- fixed clamp at `1.25` was already safe through the strongest early retained point
- the first release packet improved LPIPS, but its tighter `1.10` start likely suppressed some of the early style ceiling
- this packet tests whether the better tradeoff comes from release itself rather than from the tighter initial cap

Mechanism:

- start clamp ratio at `1.25`
- linearly relax to `1.60`
- release over the first `4` epochs

Success condition:

- recover some of the fixed-clamp style that was given back by the `1.10 -> 1.45 / 6 epoch` release
- keep LPIPS materially below the fixed-clamp `e3 = 0.7022 / 0.4867` operating point

Failure condition:

- wider release immediately reopens proximal takeover
- or it simply falls back to the fixed-clamp regime without recovering style

Planned readout:

- retain every epoch fast eval on `CLIP-style + LPIPS`
- compare against:
  - parent `AnisoStokesQueue e13 = 0.7102 / 0.4603`
  - fixed clamp `e3 = 0.7022 / 0.4867`
  - first release `e3 = 0.7007 / 0.4754`

Outcome:

- training completed through `epoch_0016`
- full eval completed through `epoch_0016`

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7075` | `0.5298` | `0.7198` | `0.5237` |
| `e2` | `0.6983` | `0.5553` | `0.7089` | `0.5483` |
| `e3` | `0.6989` | `0.4931` | `0.7143` | `0.4880` |
| `e4` | `0.6928` | `0.5230` | `0.7065` | `0.5170` |
| `e5` | `0.6912` | `0.5295` | `0.7044` | `0.5233` |
| `e6` | `0.6868` | `0.5265` | `0.7025` | `0.5183` |
| `e7` | `0.6908` | `0.4868` | `0.7086` | `0.4790` |
| `e8` | `0.6867` | `0.5193` | `0.7017` | `0.5108` |
| `e9` | `0.6889` | `0.5017` | `0.7055` | `0.4944` |
| `e10` | `0.6861` | `0.5158` | `0.7019` | `0.5072` |
| `e11` | `0.6845` | `0.5064` | `0.7007` | `0.4989` |
| `e12` | `0.6847` | `0.5031` | `0.7014` | `0.4954` |
| `e13` | `0.6898` | `0.4910` | `0.7066` | `0.4837` |
| `e14` | `0.6836` | `0.5098` | `0.6994` | `0.5017` |
| `e15` | `0.6829` | `0.5060` | `0.6995` | `0.4977` |
| `e16` | `0.6833` | `0.5115` | `0.6989` | `0.5031` |

Best retained points:

- best transfer-style:
  - `e1 = 0.7075 / 0.5298`
  - all-pairs `= 0.7198 / 0.5237`
- best LPIPS overall:
  - `e7 = 0.6908 / 0.4868`
  - all-pairs `= 0.7086 / 0.4790`
- best LPIPS under `transfer >= 0.70`:
  - only `e1` qualifies

Training-side read:

- the wider release does exactly what it was supposed to do mechanically:
  - `proximal_to_transport_ratio` grows from about `1.12` at `e1`
  - reaches the released ceiling `1.60` by `e5`
  - then stays pinned there for the rest of training
- unlike the parent `e13 -> e14` collapse, this packet does not reopen uncontrolled proximal takeover
- but it also never converts the extra proximal freedom into a better transfer frontier

Mechanism conclusion:

- removing the tighter `1.10` early squeeze was a mistake for this family
- the first release packet's LPIPS gain was not just coming from "having a release"
- it depended on that tighter early proximal suppression
- once the run starts from the looser `1.25` cap and releases farther to `1.60`, style does not recover enough to compensate, and LPIPS regresses badly

Decision:

- negative closure
- strictly worse than:
  - fixed clamp `e3 = 0.7022 / 0.4867`
  - first release `e3 = 0.7007 / 0.4754`
  - parent anchor `e13 = 0.7102 / 0.4603`
- keep this packet only as evidence that:
  - early tight proximal control matters
  - later wider release alone does not rescue style
