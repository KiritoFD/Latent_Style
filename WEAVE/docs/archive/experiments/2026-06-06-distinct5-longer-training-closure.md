# Distinct5 Longer-Training Closure

Date: 2026-06-06

Scope:

- `LBM-F longer` on `Distinct5-512`
- `LBM-K longer` on `Distinct5-512`
- same-family longer-training check only

## Purpose

Close the reviewer-sensitive question:

- are the current minute-scale Distinct5 operating points merely under-trained?

The protocol here is intentionally narrow:

- keep architecture, tokenizer, queue, loss, and data fixed
- only extend training duration
- judge retention by transfer-only `CLIP-S + LPIPS`
- use targetwise `ArtFID` only after a point survives that first screen

## Baseline anchors

Paper-facing transfer anchors:

| point | transfer CLIP-S | transfer LPIPS | transfer targetwise ArtFID | train |
| --- | ---: | ---: | ---: | ---: |
| `F e1` | `0.664360` | `0.324528` | `126.826` | `1.2m` |
| `K e1` | `0.671167` | `0.372281` | `406.151` | `1.2m` |

The retained IDT transfer floor implied by the current packet is:

- `0.639922`

## Landed `F-longer` readout

Remote output root:

- `I:\GitHub\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_f_seed42_b44_e8`

Closed transfer curve:

| epoch | transfer CLIP-S | transfer LPIPS | eval wall |
| --- | ---: | ---: | ---: |
| `e1` | `0.665761` | `0.339680` | `112.00s` |
| `e2` | `0.663725` | `0.365532` | `121.94s` |
| `e3` | `0.661765` | `0.360604` | `126.47s` |
| `e4` | `0.660711` | `0.348542` | `130.40s` |
| `e5` | `0.661519` | `0.350166` | `127.12s` |
| `e6` | `0.662654` | `0.361785` | `130.28s` |
| `e7` | `0.665558` | `0.381684` | `138.85s` |
| `e8` | `0.666298` | `0.384665` | `136.38s` |

Interpretation:

- best balance remains the earliest retained point, effectively `e1`
- later epochs slightly recover style, but every later point is worse on LPIPS
- the required gate was not met:
  - no `+0.006` transfer-CLIP gain over `F e1`
  - no `0.025` LPIPS reduction with near-flat style
- standalone targetwise `ArtFID` was not promoted by rule

Closure:

- `F-longer` is a **negative closure**
- the `F` branch behaves like early convergence, not delayed frontier recovery

## Landed `K-longer` readout

Remote output root:

- `I:\GitHub\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8`

Training runtime from the retained log:

- `e5` cumulative train wall: `332.70s` (`5.5450m`)
- `e6` cumulative train wall: `397.60s` (`6.6267m`)
- `e7` cumulative train wall: `462.29s` (`7.7048m`)
- `e8` cumulative train wall: `526.96s` (`8.7826m`)
- training VRAM stayed in the formal band:
  - peak allocated about `8.87 GiB`
  - peak reserved about `9.15 GiB`

Closed transfer curve with standalone targetwise `ArtFID`:

| epoch | transfer CLIP-S | transfer LPIPS | transfer targetwise ArtFID | eval wall |
| --- | ---: | ---: | ---: | ---: |
| `e5` | `0.667010` | `0.358785` | `408.309` | `101.55s` |
| `e6` | `0.669324` | `0.385004` | `410.737` | `104.54s` |
| `e7` | `0.670530` | `0.401353` | `410.323` | `99.20s` |
| `e8` | `0.670490` | `0.407218` | `410.930` | `105.18s` |

Interpretation:

- `e5` is the best retained balance point after the full closure
- later epochs recover style toward `K e1`, but only by paying progressively
  worse LPIPS
- standalone targetwise `ArtFID` also drifts slightly worse than the current
  `K e1` closure
- the gate was therefore not met:
  - no `+0.006` transfer-CLIP gain over `K e1`
  - no LPIPS reduction relative to `K e1`
  - no compensating ArtFID improvement

Closure:

- `K-longer` is a **negative closure**
- longer same-family training does not create a new Distinct5 frontier within
  the current compact `K` branch

## Joint read

What this packet now safely supports:

- the current Distinct5 minute-scale points are not invalidated by the simple
  "just train longer" objection
- within the current `F/K` same-family branches, longer training mainly shifts
  the trade-off toward heavier edits rather than a better retained frontier

What this packet does **not** support:

- a universal early-convergence theorem
- a claim that all longer compact runs are futile
- any use of `F-longer` or `K-longer` as a new headline paper point

## Next decision

Do not spend the next GPU slot on more same-family longer training.

Use the next slot only for:

- a new reviewer-closing mechanism packet; or
- fixed-rule follow-up split evidence

not for `F-longer` / `K-longer` continuation.
