# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-SlowMid ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4slowmid_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4slowmid_reseed_from_e13_seed42_b8a2.json)

Intent:

- preserve the two ingredients that currently look least wrong:
  - explicit `4`-epoch early hold
  - narrower `1.45` release endpoint
- isolate a new hypothesis:
  - the current degradation may come from release onset being too abrupt, not just from the release endpoint

Why this candidate exists:

- `hold4wide` suggests the explicit hold is useful, but its later `1.60` release is too permissive
- `hold4mid` removes the `1.60` endpoint, but its early training read still shows rebound once the release starts:
  - `e1-e4` steadily improve
  - `e5-e6` lose that monotonic trend after release begins
- that makes the next most coherent probe:
  - keep the same hold
  - keep the same endpoint
  - only slow the release

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- then linearly relax only to `1.45`
- release over `12` epochs instead of `8`

Success condition:

- preserve the stronger early basin from the hold family
- avoid the immediate post-hold rebound seen in `hold4mid`
- keep later epochs closer to the selected `e3/e4` tradeoff instead of reopening drift

Failure condition:

- the packet still degrades as soon as release begins
- or the longer release simply reproduces the old `hold4mid` curve more slowly without a better retained point
