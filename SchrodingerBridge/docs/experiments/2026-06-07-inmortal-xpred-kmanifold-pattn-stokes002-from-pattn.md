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
