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

## Evaluation correction

Trusted numbers here come only from the snapshot-matched fast rerun:

- run-local `src`
- current fast `run_evaluation.py` overlaid onto that run-local source
- output root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_phighpass_seed42_b32/full_eval_fast_snapshot`

The automatic post-train eval launched by `src/run.py` was not used as evidence because it loaded from the drifting repo-root source and non-strictly materialized the checkpoint against a changed tokenizer/proximal schema.

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6485` | `0.7830` |
| `e2` | `0.6613` | `0.8013` |
| `e3` | `0.6351` | `0.7992` |
| `e4` | `0.6576` | `0.7929` |
| `e5` | `0.6563` | `0.7892` |
| `e6` | `0.6657` | `0.7878` |
| `e7` | `0.6630` | `0.7831` |
| `e8` | `0.6644` | `0.7879` |

Best retained point:

- `e6`
  - transfer `clip_style = 0.6657`
  - transfer `content_lpips = 0.7878`
  - full `clip_style = 0.6657`
  - full `content_lpips = 0.7829`

## Mechanism reading

This packet is a stronger negative result than the standalone `P_highpass` packet.

Relative to `XPred + K_manifold` best (`0.7259 / 0.6863` transfer):

- style drops by about `-0.0602`
- LPIPS worsens by about `+0.1014`

Relative to the plain `XPred_Barycenter` best (`0.7161 / 0.7176` transfer):

- style drops by about `-0.0504`
- LPIPS worsens by about `+0.0702`

Interpretation:

- adding the lightweight high-pass residual branch on top of an already repaired transport field still hurts
- so the failure is not just "proximal is too weak unless transport is fixed"
- the current `highpass_residual` family itself appears misaligned with this endpoint-target regime

## Conclusion

Do **not** continue the current high-pass residual proximal family.

The surviving transport family remains:

- `XPred + K_manifold`

The next justified proximal escalation is:

- `normfree_modulation`

and, if that also fails:

- `crossattn_texture`
