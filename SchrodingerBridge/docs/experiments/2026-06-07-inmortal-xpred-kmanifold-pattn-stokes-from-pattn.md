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
