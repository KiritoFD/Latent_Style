# `XPred + K_manifold + P_attn` Frontier -> Intermediate `Stokes` Fine-Tune (`b12`)

Date: 2026-06-07

Intent:

- keep the same midpoint `w_stokes_viscous = 0.03` probe
- but lower training batch from `16` to `12`

Why this revision exists:

- the user tightened the practical explosion boundary to `> 11.5 GiB`
- earlier late-Stokes packets were already far below that band, but the next launch should still keep extra headroom
- the midpoint `0.03` probe remains the highest-value next experiment on mechanism grounds, so the right response is to lower batch rather than drop the line

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- set:
  - `training.batch_size = 12`
  - `structure_penalty_mode = stokes`
  - `w_stokes_viscous = 0.03`

Success condition:

- transfer style stays above the `0.05` late-Stokes point
- while LPIPS stays materially below the `0.02` late-Stokes point

Failure condition:

- the midpoint remains strictly on the same tradeoff curve and does not dominate either useful anchor
