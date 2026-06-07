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

## Evaluation correction

As with `XPred + K_manifold`, trusted numbers here come only from the snapshot-matched fast rerun:

- run-local `src`
- current fast `run_evaluation.py` overlaid onto that run-local source
- output root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_phighpass_seed42_b28/full_eval_fast_snapshot`

The automatic post-train eval launched by `src/run.py` was not used as paper-facing evidence because it loaded from the drifting repo-root source instead of the run-local snapshot.

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6522` | `0.7864` |
| `e2` | `0.6679` | `0.7763` |
| `e3` | `0.6631` | `0.7906` |
| `e4` | `0.6605` | `0.7820` |
| `e5` | `0.6655` | `0.7751` |
| `e6` | `0.6803` | `0.7748` |
| `e7` | `0.6787` | `0.7707` |
| `e8` | `0.6775` | `0.7724` |

Best retained point:

- `e6`
  - transfer `clip_style = 0.6803`
  - transfer `content_lpips = 0.7748`
  - full `clip_style = 0.6805`
  - full `content_lpips = 0.7703`

## Mechanism reading

This packet is a clean negative result.

Relative to `XPred_Barycenter` best (`0.7161 / 0.7176` transfer):

- style drops by about `-0.0359`
- LPIPS gets worse by about `+0.0572`

Relative to `XPred + K_manifold` best (`0.7259 / 0.6863` transfer):

- style drops by about `-0.0456`
- LPIPS gets worse by about `+0.0884`

Interpretation:

- the lightweight proximal high-pass branch did **not** rescue content
- it also did **not** preserve the style ceiling of the endpoint-target family
- in this form it appears to blunt transport pressure without adding a useful texture correction

## What likely went wrong

Most plausible reading:

- the residual proximal branch is too weak to add meaningful structure
- but it still perturbs the optimization enough to reduce style-driving transport quality
- this means the branch is neither a clean bypass nor a useful repair; it is currently just extra optimization friction

This is different from the hoped-for `Chord-style` transport/refine decomposition.

## Implication for the next round

Do **not** continue `XPred + P_highpass` as a standalone family.

The strongest surviving direction remains:

- `XPred + K_manifold`

Most justified next escalation:

- add proximal refinement only **after** the positive transport-side repair
- or move to a stronger proximal family:
  - `normfree_modulation`
  - or `crossattn_texture`

## Eval timing note

This packet also confirms the current eval bottleneck diagnosis:

- `vae_decode` remains about `52.6s`
- `eval_total` ranges about `23.7s` to `34.2s`
- `wall_total` ranges about `89.6s` to `106.6s`

So the packet is a model-side negative closure, not an evaluator artifact.
