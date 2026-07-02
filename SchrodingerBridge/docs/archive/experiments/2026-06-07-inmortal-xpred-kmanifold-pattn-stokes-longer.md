# `XPred + K_manifold + P_attn + S_stokes` Longer-Training Continuation

Date: 2026-06-07

Purpose:

- revisit the `P_attn + Stokes` tradeoff family under the corrected rule that `8 epochs` is not proof of convergence

Why this continuation exists:

- the `8-epoch` Stokes packet keeps improving LPIPS deep into the later epochs
- it underperforms the current promoted `P_attn` frontier on style, but it is the strongest structure-side LPIPS tradeoff family so far
- that makes it a valid continuation candidate rather than a closed dead packet

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_stokes_seed42_b16/epoch_0008.pt`
- extend the horizon from `8` to `12` epochs
- keep the same family and batch
- evaluate with the same snapshot-matched fast `clip+lpips` contract

Success condition:

- style holds near the late `Stokes` band
- LPIPS continues improving enough to justify the style tradeoff

Failure condition:

- the later epochs flatten without material LPIPS gain
- or style collapses faster than LPIPS improves

## Full continuation readout (`e9-e12`)

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e9` | `0.7112` | `0.5653` |
| `e10` | `0.7117` | `0.5676` |
| `e11` | `0.7105` | `0.5708` |
| `e12` | `0.7140` | `0.5696` |

Selected continuation point:

- `e9`
  - transfer `clip_style = 0.7112`
  - transfer `content_lpips = 0.5653`
  - full `clip_style = 0.7219`
  - full `content_lpips = 0.5554`

## What changed relative to the 8-epoch Stokes packet

Previous selected point from the `8-epoch` Stokes packet:

- `e3`
  - transfer `0.7193 / 0.6222`

Continuation selected point:

- `e9`
  - transfer `0.7112 / 0.5653`

So the continuation buys:

- about `-0.0569` LPIPS
- but also about `-0.0081` style

## Interpretation

This is a real continuation gain, but it remains a tradeoff family rather than a promoted replacement.

Relative to the current promoted `P_attn` continuation point (`0.7289 / 0.6211` transfer):

- style drops by about `-0.0178`
- LPIPS improves by about `-0.0558`

So the family is useful if the next round wants to explore a style-for-content tradeoff, but it still does not dominate the current frontier.

## Conclusion

Longer training confirms the same structural reading:

- `Stokes` is much better than `Aniso`
- it can keep driving LPIPS downward
- but it does so by paying too much style

Most justified next step:

- do **not** keep scaling this from-scratch `Stokes` family indefinitely
- instead try a fine-tuning packet that starts from the promoted `P_attn` frontier and then enables weak `Stokes` smoothing

Reason:

- the current evidence suggests the family is directionally useful
- but the from-scratch optimization path loses too much style before settling
