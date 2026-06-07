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

## Evaluation correction

Trusted numbers here come only from the snapshot-matched fast rerun:

- run-local `src`
- current fast `run_evaluation.py` overlaid onto that run-local source
- output root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16/full_eval_fast_snapshot`

The automatic post-train eval from `src/run.py` was not used as evidence because it still materialized the checkpoint through the drifting repo-root path.

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6617` | `0.7574` |
| `e2` | `0.7160` | `0.7247` |
| `e3` | `0.7141` | `0.6779` |
| `e4` | `0.7224` | `0.6774` |
| `e5` | `0.7271` | `0.6559` |
| `e6` | `0.7289` | `0.6370` |
| `e7` | `0.7257` | `0.6354` |
| `e8` | `0.7257` | `0.6121` |

Best retained point under the current promotion rule:

- `e6`
  - transfer `clip_style = 0.7289`
  - transfer `content_lpips = 0.6370`
  - full `clip_style = 0.7338`
  - full `content_lpips = 0.6278`

## Mechanism reading

This is the first proximal family that produces a real frontier improvement over `XPred + K_manifold`.

Relative to `XPred + Kmanifold` best (`0.7259 / 0.6863` transfer):

- style improves by about `+0.0029`
- LPIPS improves by about `-0.0493`

Relative to `P_mod` best (`0.7153 / 0.7305` transfer):

- style improves by about `+0.0135`
- LPIPS improves by about `-0.0935`

Interpretation:

- the proximal direction was not wrong in principle
- the weak residual family was wrong
- the stronger modulation family helped somewhat
- explicit cross-attention texture refinement is the first proximal branch that actually adds useful residual structure on top of the repaired transport field

## What is now true

Current strongest `inmortal` family:

- `endpoint prediction`
- `barycentric target smoothing`
- `manifold-adaptive kinetic`
- `cross-attention texture proximal refinement`

This family now dominates the earlier `XPred + Kmanifold` packet on both primary transfer metrics.

## What is still not solved

- LPIPS is still far from the long-range `0.30` target
- the packet is clearly better, but not yet enough to close the ceiling-push objective
- style remains high while LPIPS continues to improve through later epochs, so this family may still have useful headroom

## Next round implication

Most justified next action:

- stay inside the `P_attn` family
- extend training budget beyond 8 epochs

Reason:

- unlike the earlier proximal families, this one does not collapse immediately
- its curve keeps improving on LPIPS through at least `e8`
- the next question is no longer "does proximal help at all?"
- it is now "how much of the remaining LPIPS gap can this stronger proximal family recover with more training?"

## Continuation result

The follow-up `12-epoch` continuation is now landed separately at:

- [2026-06-07-inmortal-xpred-kmanifold-pattn-longer.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-inmortal-xpred-kmanifold-pattn-longer.md)

Key takeaway:

- longer training does improve LPIPS further while holding style essentially flat
- selected continuation point:
  - `e11 = 0.7289 / 0.6211` transfer

So this family remains the best current `inmortal` frontier, but the continuation also shows diminishing returns from budget alone.
