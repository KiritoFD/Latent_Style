# `XPred + P_highpass` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest known style-ceiling family:
  - endpoint prediction
  - barycentric target smoothing
- add the lightest proximal high-frequency residual branch

Why this candidate exists:

- the plain `XPred_Barycenter` line proves the endpoint-target family can push style far above the compact baseline band
- but its content damage suggests the transport target is too coarse
- a constrained high-pass proximal branch may let transport stay coarse while reintroducing localized texture without further destroying structure

Success condition:

- style stays in the `0.71` neighborhood
- LPIPS does not worsen further
- base/final endpoint readouts show that the proximal branch helps rather than completely bypassing transport
