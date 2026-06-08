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
