# `XPred + K_manifold + P_attn` Frontier -> Weaker `Stokes` Fine-Tune

Date: 2026-06-07

Intent:

- keep the same late-fine-tune mechanism that just produced the promoted frontier
- but reduce `w_stokes_viscous` from `0.05` to `0.02`

Why this follow-up is justified:

- the `0.05` late fine-tune already proved that late `Stokes` can improve LPIPS without collapsing style immediately
- but the selected point happened at `e13`, and later epochs kept giving style away
- that is direct evidence of over-smoothing, not evidence against the mechanism family

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- enable:
  - `structure_penalty_mode = stokes`
  - weaker `w_stokes_viscous = 0.02`

Success condition:

- transfer style stays at or above the new promoted `0.7274` band
- LPIPS stays below the promoted `0.6033` point, or improves further

Failure condition:

- style gain from weakening `Stokes` is too small to matter
- or LPIPS rebounds enough that the `0.05` late fine-tune remains the best tradeoff

## Full fine-tune readout (`e12-e16`)

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e12` | `0.7284` | `0.6186` |
| `e13` | `0.7307` | `0.6183` |
| `e14` | `0.7275` | `0.6337` |
| `e15` | `0.7265` | `0.6449` |
| `e16` | `0.7201` | `0.6385` |

Selected point under the current promotion rule:

- `e13`
  - transfer `clip_style = 0.7307`
  - transfer `content_lpips = 0.6183`
  - full `clip_style = 0.7372`
  - full `content_lpips = 0.6069`

## What changed relative to the `0.05` late-Stokes frontier

Previous promoted point:

- `late weak Stokes 0.05`
  - transfer `0.7274 / 0.6033`

Weaker-Stokes result:

- `late weaker Stokes 0.02`
  - transfer `0.7307 / 0.6183`

So this follow-up buys:

- about `+0.0032` style
- but pays about `+0.0150` LPIPS

## Interpretation

This packet is a **positive style-ceiling result** and a **negative balance result**.

What it proves:

- the `0.05` packet was not the maximum style ceiling available inside the late-Stokes family
- weakening `Stokes` really does recover more of the style band

What it also proves:

- the extra style comes from giving back a meaningful chunk of the LPIPS gain
- so the `0.05` packet remains the cleaner balance point even though `0.02` is the stronger raw-style point

That is a useful mechanism read:

- `late Stokes` is not a binary on/off win
- it behaves like a continuous tradeoff knob
- and the family now has at least two paper-useful operating points:
  - `0.02` for the stronger style frontier
  - `0.05` for the better LPIPS-balanced frontier

## Conclusion

Under the current promotion rule, `0.02` becomes the new promoted raw-style frontier because the style gain exceeds the `0.002` threshold.

But the practical reading is more nuanced:

- if the next round optimizes for pure transfer style, start from `0.02`
- if it optimizes for balance, keep `0.05` as the better near-frontier anchor
