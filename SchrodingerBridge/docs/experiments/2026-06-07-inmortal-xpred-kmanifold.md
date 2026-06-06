# `XPred + K_manifold` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest known style-ceiling family:
  - endpoint prediction
  - barycentric target smoothing
- add the strongest currently hypothesized content-preserving kinetic family:
  - manifold-adaptive split

Why this candidate exists:

- `XPred_Barycenter b40` already reaches the `0.71+` transfer style band
- its failure mode is catastrophic LPIPS
- `K_manifold` is the most natural attempt to repair that failure without abandoning the endpoint-target geometry

Success condition:

- style remains near the `XPred_Barycenter` band
- LPIPS improves materially relative to the plain `XPred_Barycenter` line
