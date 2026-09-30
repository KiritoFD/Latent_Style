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

## Evaluation correction

Trusted numbers here come only from the snapshot-matched fast rerun:

- run-local `src`
- current fast `run_evaluation.py` overlaid onto that run-local source
- output root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pmod_seed42_b32/full_eval_fast_snapshot`

The automatic post-train eval from `src/run.py` was not used as evidence because it still materialized the checkpoint through the drifting repo-root source path.

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6878` | `0.7873` |
| `e2` | `0.7113` | `0.7684` |
| `e3` | `0.6876` | `0.7413` |
| `e4` | `0.7015` | `0.7341` |
| `e5` | `0.7041` | `0.7465` |
| `e6` | `0.7116` | `0.7391` |
| `e7` | `0.7153` | `0.7305` |
| `e8` | `0.7096` | `0.7238` |

Best retained point under the current promotion rule:

- `e7`
  - transfer `clip_style = 0.7153`
  - transfer `content_lpips = 0.7305`
  - full `clip_style = 0.7161`
  - full `content_lpips = 0.7244`

## Mechanism reading

This packet is better than both `highpass_residual` packets, but still not good enough.

Relative to `XPred + K_manifold` best (`0.7259 / 0.6863` transfer):

- style drops by about `-0.0106`
- LPIPS worsens by about `+0.0442`

Relative to the plain `XPred_Barycenter` best (`0.7161 / 0.7176` transfer):

- style is still slightly lower by about `-0.0008`
- LPIPS is still worse by about `+0.0129`

Interpretation:

- `normfree_modulation` is a **real improvement over the failed high-pass residual family**
- it preserves much more of the endpoint-target style ceiling
- but it still fails to add a net content-repair benefit on top of `K_manifold`

## Conclusion

Do not promote `P_mod` over `XPred + K_manifold`.

What survives:

- the transport side remains correct:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic
- the weak proximal family is dead
- the stronger modulation family is not enough

Most justified next step:

- `crossattn_texture`

because it is the last strong proximal family in the corrected `inmortal` ladder that can still plausibly add structured style refinement without collapsing transport.
