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
