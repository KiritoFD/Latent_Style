# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-Mid ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4mid_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4mid_reseed_from_e13_seed42_b8a2.json)

Intent:

- preserve the newly validated explicit `4`-epoch hold
- remove the part that still looks harmful:
  - widening the clamp all the way to `1.60`
- test whether the hold benefit survives when the release endpoint is pulled back to the earlier successful `1.45` family

Why this candidate exists:

- the original release packet at `1.45` gave the best low-LPIPS recovery signal so far
- the new `hold4wide` packet shows the explicit early hold is directionally useful:
  - its selected `e3` slightly improves over the old release family
- but the later `1.60` release still pushes the packet into the same late degradation pattern

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- then linearly relax only to `1.45`
- release over the next `8` epochs

Success condition:

- keep the small `e3` gain from the explicit hold
- avoid the later LPIPS regression seen in the wide-release packet
- produce a cleaner retained curve than both:
  - the original `1.45` release packet
  - the `hold4wide` packet

Failure condition:

- the packet simply collapses back onto the old release family with no real gain
- or the later epochs still drift even when the endpoint is no longer widened to `1.60`

Early health:

- remote launch passed first health under the paper-facing machine contract
- observed runtime memory stayed around `4.55 / 12.29 GiB` during the first several epochs
- the explicit `4`-epoch hold segment completed cleanly:
  - `e1 = loss 9.3115, terminal_swd 5.5938, samples_per_sec 28.75`
  - `e2 = loss 9.2747, terminal_swd 5.6875, samples_per_sec 29.26`
  - `e3 = loss 9.0994, terminal_swd 5.4375, samples_per_sec 29.33`
  - `e4 = loss 8.9711, terminal_swd 5.0625, samples_per_sec 29.53`
- early read:
  - the packet is numerically stable
  - the first hold window does not reintroduce the wide-release instability
  - the main question is now whether the narrower `1.45` release preserves the `hold4wide` `e3` gain without drifting later

Mid-train read:

- the first release steps show a visible transition shock, but not a complete collapse:
  - `e5 = loss 9.1012, terminal_swd 5.3125`
  - `e6 = loss 9.1083, terminal_swd 5.5000`
  - `e7 = loss 9.0739, terminal_swd 5.4062`
- later retained epochs partially recover and then continue improving:
  - `e8 = loss 8.8816, terminal_swd 5.2188`
  - `e9 = loss 8.8749, terminal_swd 5.1250`
  - `e10 = loss 8.9563, terminal_swd 5.5312`
  - `e11 = loss 8.7527, terminal_swd 5.0938`
  - `e12 = loss 8.6133, terminal_swd 5.0312`
- interpretation:
  - the explicit hold still appears useful
  - switching the endpoint from `1.60` back to `1.45` removes the worst wide-release drift
  - the release transition is still rough, but the packet can recover after that shock
  - by `e11/e12`, the packet is no longer merely surviving; it is actively re-entering a cleaner basin
  - this means `hold4mid` remains a live line rather than a premature negative closure
  - `hold4slowmid` stays justified as the next queue candidate, but it is now a hedge against transition roughness, not an immediate rescue from failure

Partial eval read:

- once training finished, the run immediately started writing `full_eval/epoch_*/summary.json`
- the first `8` retained eval points are extremely consistent:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.6675` | `0.2950` | `0.7005` | `0.2951` |
| `e2` | `0.6680` | `0.2889` | `0.7014` | `0.2890` |
| `e3` | `0.6673` | `0.2943` | `0.7004` | `0.2943` |
| `e4` | `0.6671` | `0.2913` | `0.7005` | `0.2913` |
| `e5` | `0.6675` | `0.2924` | `0.7007` | `0.2925` |
| `e6` | `0.6669` | `0.2921` | `0.7003` | `0.2921` |
| `e7` | `0.6671` | `0.2955` | `0.7002` | `0.2956` |
| `e8` | `0.6679` | `0.2877` | `0.7014` | `0.2878` |

Interim interpretation:

- this is not a style-ceiling packet
- it is a very strong content-preserving packet with unusually low LPIPS
- the line is almost flat across the first half of retained epochs, which suggests:
  - the hold-plus-mid-release schedule is stabilizing geometry very strongly
  - but it may also be suppressing later style escalation too hard
- `e8` is the current early best point in this partial read:
  - `0.6679 / 0.2877` transfer
  - `0.7014 / 0.2878` all-pairs

Outcome:

- training completed through `epoch_0016`
- full eval completed through `epoch_0016`
- the remote run itself had already produced all `summary.json` files, but `clip_lpips_curve.csv` was missing
- closure was finalized by rerunning:
  - `rerun_full_eval_for_run.py --skip-existing --output-subdir full_eval`
  - this wrote the missing `clip_lpips_curve.csv` without recomputing completed eval points

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.6675` | `0.2950` | `0.7005` | `0.2951` |
| `e2` | `0.6680` | `0.2889` | `0.7014` | `0.2890` |
| `e3` | `0.6673` | `0.2943` | `0.7004` | `0.2943` |
| `e4` | `0.6671` | `0.2913` | `0.7005` | `0.2913` |
| `e5` | `0.6675` | `0.2924` | `0.7007` | `0.2925` |
| `e6` | `0.6669` | `0.2921` | `0.7003` | `0.2921` |
| `e7` | `0.6671` | `0.2955` | `0.7002` | `0.2956` |
| `e8` | `0.6679` | `0.2877` | `0.7014` | `0.2878` |
| `e9` | `0.6669` | `0.2966` | `0.7000` | `0.2966` |
| `e10` | `0.6673` | `0.2930` | `0.7005` | `0.2929` |
| `e11` | `0.6671` | `0.2938` | `0.7004` | `0.2937` |
| `e12` | `0.6674` | `0.2906` | `0.7008` | `0.2907` |
| `e13` | `0.6675` | `0.2928` | `0.7008` | `0.2928` |
| `e14` | `0.6674` | `0.2926` | `0.7007` | `0.2926` |
| `e15` | `0.6668` | `0.2929` | `0.7002` | `0.2929` |
| `e16` | `0.6674` | `0.2936` | `0.7007` | `0.2936` |

Best retained points:

- best transfer-style:
  - `e2 = 0.6680 / 0.2889`
  - all-pairs `= 0.7014 / 0.2890`
- best LPIPS within the same style band:
  - `e8 = 0.6679 / 0.2877`
  - all-pairs `= 0.7014 / 0.2878`

Why this packet matters:

- it is by far the strongest low-LPIPS line in the current `inmortal` recovery family
- relative to the current promoted low-LPIPS frontier:
  - promoted `AnisoStokesQueue e13 = 0.7102 / 0.4603`
  - `hold4mid e8 = 0.6679 / 0.2877`
- that is not a better headline transfer point
- but it proves the family can lock into an extremely content-preserving basin with stable geometry on `Distinct5-512`

Mechanism conclusion:

- `hold + mid release` is not a style-ceiling mechanism
- it is a geometry-anchor mechanism:
  - strong content preservation
  - very flat LPIPS-stable retained curve
  - weak style escalation throughout the run
- the family is therefore useful as evidence that:
  - the proximal clamp schedule can decisively control geometric drift
  - but this particular schedule over-constrains style lift

Decision:

- positive closure as a geometry/content anchor
- not promotable over the current paper-facing headline frontier
- retain `e8` as the strongest low-LPIPS operating point from this family
- next step should treat this family as a source of geometry control and reopen style with a different late mechanism, rather than expecting more epochs alone to lift style
