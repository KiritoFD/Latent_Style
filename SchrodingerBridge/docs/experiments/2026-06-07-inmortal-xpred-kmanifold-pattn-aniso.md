# `XPred + K_manifold + P_attn + S_aniso` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest current family:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic
  - cross-attention texture proximal refinement
- add the anisotropic structure penalty to suppress boundary-normal drift without killing tangential brushstroke motion

Why this candidate exists:

- `P_attn` is the first proximal family that improves the current frontier
- longer training gives only a moderate LPIPS gain
- the next most justified step is a transport-side structure repair on top of the best current family, not another weaker proximal branch

Success condition:

- style stays near the promoted `P_attn` band
- LPIPS improves relative to the current promoted point
- the packet avoids the content-damage plateau seen in the plain `P_attn` family

Failure condition:

- style collapses materially below the current `P_attn` band
- or LPIPS fails to improve beyond the continuation frontier
