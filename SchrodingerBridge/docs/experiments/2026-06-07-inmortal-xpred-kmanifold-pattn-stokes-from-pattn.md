# `XPred + K_manifold + P_attn` Frontier -> `Stokes` Fine-Tune

Date: 2026-06-07

Intent:

- start from the current promoted `P_attn` continuation frontier
- then enable weak `Stokes` smoothing as a fine-tuning repair, rather than paying the full style cost of training the `Stokes` family from scratch

Why this candidate exists:

- `P_attn + Stokes` from scratch keeps improving LPIPS but gives up too much style
- `P_attn` continuation is still the strongest promoted frontier
- the obvious next test is whether `Stokes` helps more when applied late, after the high-style transport/proximal geometry is already formed

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- enable:
  - `structure_penalty_mode = stokes`
  - weak `w_stokes_viscous`

Success condition:

- transfer style stays near the promoted `P_attn` band
- LPIPS improves beyond the current promoted point

Failure condition:

- the fine-tune still pays the same large style penalty seen in the from-scratch `Stokes` family

## Full fine-tune readout (`e13-e16`)

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e13` | `0.7274` | `0.6033` |
| `e14` | `0.7245` | `0.6280` |
| `e15` | `0.7215` | `0.6262` |
| `e16` | `0.7169` | `0.6196` |

Selected fine-tune point:

- `e13`
  - transfer `clip_style = 0.7274`
  - transfer `content_lpips = 0.6033`
  - full `clip_style = 0.7356`
  - full `content_lpips = 0.5915`

## What changed relative to the previous promoted frontier

Previous promoted point:

- `XPred + K_manifold + P_attn` continuation `e11`
  - transfer `0.7289 / 0.6211`

Late `Stokes` fine-tune result:

- `e13`
  - transfer `0.7274 / 0.6033`

So the late fine-tune buys:

- about `-0.0015` style
- but about `-0.0178` LPIPS

Under the current promotion rule, that is enough to promote:

- style stays within the `0.002` tie band of the previous leader
- LPIPS improves materially beyond the `0.01` threshold

## Interpretation

This is a **positive closure** for the mechanism.

The important distinction is not just that `Stokes` can lower LPIPS.
We already knew the from-scratch `Stokes` family could do that, but it paid too much style.

What this packet shows is stronger:

- late, weak `Stokes` smoothing can recover a meaningful LPIPS gain
- while preserving almost all of the high-style geometry already formed by the promoted `P_attn` family

So the main problem with `Stokes` was not that the mechanism itself was unusable.
The problem was **when** and **how strongly** it was applied.

## Conclusion

This packet becomes the new promoted frontier.

It does not solve the long-range LPIPS gap to the `0.30` target band, but it changes the next-round logic:

- keep the `P_attn` family as the backbone
- keep `late / weak structure repair` as an active direction
- stop treating `Stokes` as only a tradeoff family from the from-scratch evidence
