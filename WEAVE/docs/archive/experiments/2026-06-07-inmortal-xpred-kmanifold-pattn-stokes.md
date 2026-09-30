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

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.7016` | `0.6836` |
| `e2` | `0.7127` | `0.6418` |
| `e3` | `0.7193` | `0.6222` |
| `e4` | `0.7117` | `0.5943` |
| `e5` | `0.7161` | `0.5884` |
| `e6` | `0.7159` | `0.5770` |
| `e7` | `0.7057` | `0.5747` |
| `e8` | `0.7105` | `0.5607` |

Best retained point under the current promotion rule:

- `e3`
  - transfer `clip_style = 0.7193`
  - transfer `content_lpips = 0.6222`
  - full `clip_style = 0.7257`
  - full `content_lpips = 0.6138`

Lower-style LPIPS tradeoff point:

- `e8`
  - transfer `clip_style = 0.7105`
  - transfer `content_lpips = 0.5607`

## Mechanism reading

This packet is better than `P_attn + Aniso`, but still not a new frontier.

Relative to the current promoted `P_attn` continuation point (`0.7289 / 0.6211` transfer):

- selected point `e3` loses about `-0.0096` style
- and is still slightly worse on LPIPS by about `+0.0011`

The late LPIPS-heavy tradeoff point `e8` is interesting:

- LPIPS improves a lot
- but style falls far enough that it does not qualify as a promoted replacement

Interpretation:

- weak `Stokes` smoothing is much gentler than `Aniso`
- it does not strangle the style field as aggressively
- but it still shifts the family into a style-vs-LPIPS tradeoff rather than a strict frontier improvement

## Conclusion

`P_attn + Stokes` is a useful negative-to-neutral packet:

- clearly better than the harsher `Aniso` repair
- not better than the current `P_attn` continuation frontier under the promoted selection rule

This means:

- the best current family is still `P_attn` without extra structure penalty
- but if we want to chase lower LPIPS, the `Stokes` direction is the cleaner structure-side tradeoff than `Aniso`

## Continuation result

The `12-epoch` continuation is now landed separately at:

- [2026-06-07-inmortal-xpred-kmanifold-pattn-stokes-longer.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-inmortal-xpred-kmanifold-pattn-stokes-longer.md)

Key continuation point:

- `e9 = 0.7112 / 0.5653` transfer

So `Stokes` does continue to improve LPIPS with more budget, but it still remains a style-for-content tradeoff rather than becoming the promoted frontier.
