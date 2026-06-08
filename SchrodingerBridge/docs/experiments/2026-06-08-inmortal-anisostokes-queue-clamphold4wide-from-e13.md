# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-Wide ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4wide_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4wide_reseed_from_e13_seed42_b8a2.json)

Intent:

- preserve the known-good early `1.10` clamp exactly
- stop approximating the early hold with a slow linear schedule
- test whether the missing ingredient is an explicit hold window before wider release

Why this candidate exists:

- the first release packet suggests the early low-LPIPS basin depends on a genuinely tight proximal cap
- the negative `wide release` packet shows that loosening the early basin is harmful
- the current `late-wide linear` packet is still only an approximation of "hold, then release"
- this packet turns that hypothesis into an explicit mechanism

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- then linearly relax to `1.60`
- release over the next `8` epochs

Success condition:

- keep the early packet in the same low-LPIPS basin as:
  - first release `e3 = 0.7007 / 0.4754`
- while giving the model a later style-recovery window that can exceed that style level without reopening proximal takeover

Failure condition:

- even with an explicit hold window, later wider release still cannot improve the frontier
- or the release phase simply reintroduces late proximal domination
