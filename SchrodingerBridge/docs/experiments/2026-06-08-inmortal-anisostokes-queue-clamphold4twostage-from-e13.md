# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-TwoStage ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json)

Intent:

- stop spending more GPU on single-stage release smoothing
- preserve what the hold family clearly does well:
  - geometry stabilization
  - low-LPIPS basin formation
- reopen style only after a mid-band geometry basin has already been established

Why this candidate exists:

- `hold4mid` proved the family can lock geometry into an extreme low-LPIPS basin
- `hold4slowmid` showed that simply slowing the same one-stage release does not improve on that anchor
- the next coherent mechanism change is therefore structural:
  - first release into a controlled middle band
  - pause there
  - then reopen later toward a wider style regime

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- release to `1.30` over the next `4` epochs
- hold the `1.30` band for another `4` epochs
- then reopen late toward `1.60` over the final `8` epochs
- total budget is extended to `20` epochs

Success condition:

- preserve the `hold4mid` geometry anchor through the middle of training
- then recover a meaningful amount of style in the late second release
- beat either:
  - `hold4mid` on style at comparable LPIPS
  - or the current recovery-family `e3` points on LPIPS at comparable style

Failure condition:

- the packet just reproduces the geometry anchor with no late style recovery
- or the late second release reintroduces the old wide-release instability
