# `XPred + K_manifold + P_attn` Frontier -> Intermediate `Stokes` Fine-Tune

Date: 2026-06-07

Intent:

- keep the same late-fine-tune mechanism family
- test the midpoint `w_stokes_viscous = 0.03`

Why this follow-up is justified:

- `0.05` is now the cleaner LPIPS-balanced near-frontier point
- `0.02` is now the stronger raw-style point
- that exposes a clear one-dimensional tradeoff curve instead of a binary win/loss
- the obvious next probe is the midpoint, to test whether this family can produce a better compromise than either endpoint

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- enable:
  - `structure_penalty_mode = stokes`
  - intermediate `w_stokes_viscous = 0.03`

Success condition:

- transfer style stays above the `0.05` late-Stokes point
- while LPIPS stays materially below the `0.02` late-Stokes point

Failure condition:

- the packet lands strictly on the same monotone tradeoff curve without improving the current useful anchors
