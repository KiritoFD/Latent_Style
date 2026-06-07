# `XPred + K_manifold + P_attn` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest current transport family:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic repair
- escalate to the strongest remaining proximal family:
  - cross-attention texture residual

Why this candidate exists:

- `highpass_residual` fails badly even after transport repair
- `normfree_modulation` is directionally better but still not enough
- the last remaining strong proximal family in the corrected `inmortal` ladder is the explicit cross-attention texture residual branch

Success condition:

- style stays near the `XPred + K_manifold` band
- LPIPS improves relative to `XPred + Kmanifold`
- base/final endpoint metrics show a real residual refinement instead of another transport degradation

Failure condition:

- if style still drops away from the `XPred + Kmanifold` band
- or LPIPS stays worse than `XPred + Kmanifold`
- then the current proximal direction should be treated as exhausted under the present endpoint-target regime
