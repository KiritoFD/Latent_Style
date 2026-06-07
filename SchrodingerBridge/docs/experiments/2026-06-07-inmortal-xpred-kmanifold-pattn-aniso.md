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

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.7136` | `0.5931` |
| `e2` | `0.6863` | `0.6568` |
| `e3` | `0.6888` | `0.6887` |
| `e4` | `0.6840` | `0.6911` |
| `e5` | `0.6853` | `0.7170` |
| `e6` | `0.6769` | `0.6917` |
| `e7` | `0.6823` | `0.7191` |
| `e8` | `0.6877` | `0.7186` |

Best retained point under the current promotion rule:

- `e1`
  - transfer `clip_style = 0.7136`
  - transfer `content_lpips = 0.5931`
  - full `clip_style = 0.7184`
  - full `content_lpips = 0.5878`

## Mechanism reading

This is a negative packet overall.

Relative to the current promoted `P_attn` continuation point (`0.7289 / 0.6211` transfer):

- style drops by about `-0.0154`
- LPIPS improves by about `-0.0280`

Interpretation:

- anisotropic structure pressure does buy a cleaner early-LPIPS point
- but it does so by over-constraining transport too hard
- the family cannot hold the style band while keeping that LPIPS gain
- later epochs do not recover into a better frontier; they simply degrade

So this is not a frontier improvement, only a lower-style tradeoff point.

## Conclusion

Do not promote `P_attn + Aniso` over the current `P_attn` continuation frontier.

What it does tell us:

- the next transport-side repair should be gentler than anisotropic normal suppression
- if we want to keep the `P_attn` family and still improve LPIPS, the next cleaner candidate is:
  - `Stokes` smoothing

because it is a weaker structural smoother than anisotropic gating and is less likely to strangle the style field this early.
