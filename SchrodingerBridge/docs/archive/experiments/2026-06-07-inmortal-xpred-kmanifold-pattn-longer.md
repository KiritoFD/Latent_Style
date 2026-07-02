# `XPred + K_manifold + P_attn` Longer-Training Continuation

Date: 2026-06-07

Purpose:

- take the first promoted proximal frontier
  - `XPred + K_manifold + P_attn`
- and test whether the remaining LPIPS gap is still shrinking with more training budget

Why this continuation is justified:

- the `8-epoch` packet already beats `XPred + Kmanifold` on both primary transfer metrics
- LPIPS continues improving into the late epochs
- this is now the strongest evidence-backed family, so increasing its budget is more justified than opening another new side branch first

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16/epoch_0008.pt`
- extend the training horizon from `8` to `12` epochs
- keep the same batch and mechanism family
- evaluate with the same snapshot-matched fast `clip+lpips` contract

Success condition:

- transfer `clip_style` stays in the promoted band
- transfer `content_lpips` keeps improving beyond the `epoch_0006` frontier

Failure condition:

- later epochs recover only negligible LPIPS gains while style degrades
- or the curve plateaus so clearly that additional budget no longer buys frontier movement

## Full continuation readout (`e9-e12`)

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e9` | `0.7271` | `0.6141` |
| `e10` | `0.7262` | `0.6137` |
| `e11` | `0.7289` | `0.6211` |
| `e12` | `0.7295` | `0.6299` |

Selection note:

- highest raw style is `e12`
- but under the current promotion rule, any point within `0.002` of the style leader breaks ties on lower LPIPS
- that makes `e11` the selected retained point, not `e12`

Selected continuation point:

- `e11`
  - transfer `clip_style = 0.7289`
  - transfer `content_lpips = 0.6211`
  - full `clip_style = 0.7352`
  - full `content_lpips = 0.6113`

## What changed relative to the 8-epoch frontier

Previous promoted point from the base `8-epoch` packet:

- `e6`
  - transfer `0.7289 / 0.6370`

Continuation result:

- `e11`
  - transfer `0.7289 / 0.6211`

So the continuation buys:

- essentially flat style
- but another about `-0.0160` LPIPS improvement

## Interpretation

This is a **real** same-family gain, not noise.

However it is also a **diminishing-return** gain:

- the family clearly still has some LPIPS headroom
- but the additional budget is no longer moving the frontier dramatically
- the curve is already showing tension between the best-style point (`e12`) and the best-selected point (`e11`)

## Conclusion

Longer training on the `P_attn` family is worth reporting and keeping.

But the result also suggests:

- more training alone is unlikely to close the remaining LPIPS gap to the long-range target
- the next round should keep this family as the best current backbone
- and add a new transport-side repair on top of it rather than only stretching budget again
