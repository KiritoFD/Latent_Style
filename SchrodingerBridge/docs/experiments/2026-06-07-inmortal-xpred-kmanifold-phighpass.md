# `XPred + K_manifold + P_highpass` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest currently surviving family:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic repair
- then re-test the lightweight proximal high-pass branch only after transport has already been improved

Why this candidate exists:

- `XPred + K_manifold` is the current best `inmortal` packet
- `XPred + P_highpass` failed badly as a standalone repair
- the most likely explanation is that the proximal branch only becomes useful after the transport field is already disciplined

Success condition:

- style stays near the `XPred + K_manifold` band
- LPIPS improves relative to `XPred + K_manifold`
- base/final endpoint metrics show additive refinement rather than a degraded transport field

Failure condition:

- if style falls back toward the plain `P_highpass` band
- or LPIPS worsens relative to `XPred + K_manifold`
- then the current high-pass proximal family should be treated as a negative branch and replaced by a stronger proximal family
