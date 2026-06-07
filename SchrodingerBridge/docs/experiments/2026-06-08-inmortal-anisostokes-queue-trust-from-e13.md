# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + ProximalTrust` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2.json)

Intent:

- preserve the parent `e13` low-LPIPS anchor:
  - transfer `0.7102 / 0.4603`
- do not change the parent family:
  - `Aniso + weak Stokes + Queue`
- only add a proximal trust-region penalty

Why this candidate exists:

- the parent line does not look like a fake eval spike
- but after `e13`, the run drifts into a proximal-dominant regime
- the clearest symptom is:
  - `proximal_residual_abs` stays small through `e13`
  - then jumps hard at `e14+`
- we therefore want a mechanism that:
  - leaves transport alone
  - leaves the good `e13` basin reachable
  - blocks the later proximal takeover

Mechanism:

- measure proximal RMS from `last_proximal_residual`
- measure transport RMS from detached `last_base_endpoint - content`
- only penalize the excess if:
  - `proximal_rms > trust_ratio * detached_transport_rms`
- penalty acts on proximal only
- transport reference is detached so the model is not encouraged to collapse transport just to satisfy the trust gate

Success condition:

- keep the low-LPIPS behavior of the parent line
- while preventing the catastrophic `e14+` degradation
- ideally recover a stable retained region near or above the parent `e13` point

Failure condition:

- the trust gate over-constrains style immediately
- or it fails to prevent the same post-`e13` collapse

Primary readout:

- transfer `CLIP-style / LPIPS`
- all-pairs `CLIP-style / LPIPS`
- training-side:
  - `proximal_residual_abs`
  - `base_transport_abs`
  - `proximal_to_transport_ratio`
  - `proximal_trust_penalty`

Reflection prompt:

- if this works, the remaining ceiling debt is not transport quality but proximal stability
- if this fails, the family probably needs a stronger architectural split than a soft trust-region can provide
