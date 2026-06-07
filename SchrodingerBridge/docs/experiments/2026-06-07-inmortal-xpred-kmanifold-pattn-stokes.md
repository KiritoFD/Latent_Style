# `XPred + K_manifold + P_attn + S_stokes` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest current family:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic
  - cross-attention texture proximal refinement
- add the weaker `Stokes` structural smoother instead of the harsher anisotropic normal penalty

Why this candidate exists:

- `P_attn` is the best current frontier
- `Anisotropic` buys LPIPS but strangles style too hard
- the next cleaner structure-side repair is weak Laplacian / Stokes smoothing, which should regularize transport without the same boundary-normal overconstraint

Success condition:

- style stays near the promoted `P_attn` continuation band
- LPIPS improves relative to the current promoted point

Failure condition:

- style falls materially below the current `P_attn` band
- or LPIPS fails to improve beyond the continuation frontier
