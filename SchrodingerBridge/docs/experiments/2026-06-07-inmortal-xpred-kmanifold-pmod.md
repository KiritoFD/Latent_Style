# `XPred + K_manifold + P_mod` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest current transport family:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic repair
- replace the weak high-pass residual proximal branch with a stronger residual modulation branch

Why this candidate exists:

- `XPred + K_manifold` is currently the best `inmortal` packet
- `XPred + P_highpass` and `XPred + K_manifold + P_highpass` both look likely to under-use the proximal branch while still perturbing transport
- the next principled escalation is a stronger proximal family, not more tuning on the failed high-pass residual family

Success condition:

- style stays near the `XPred + K_manifold` band
- LPIPS improves relative to `XPred + K_manifold`
- base/final endpoint metrics show that the proximal branch adds useful residual refinement instead of weakening transport

Failure condition:

- if style collapses back toward the weak proximal packets
- or LPIPS worsens relative to `XPred + K_manifold`
- then the next surviving proximal family should be `crossattn_texture`
