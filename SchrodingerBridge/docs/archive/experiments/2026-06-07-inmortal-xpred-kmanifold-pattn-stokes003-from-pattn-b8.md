# `XPred + K_manifold + P_attn` Frontier -> Intermediate `Stokes` Fine-Tune (`b8`)

Date: 2026-06-07

Intent:

- keep the same midpoint `w_stokes_viscous = 0.03` probe
- reduce training batch further from `12` to `8`

Why this revision exists:

- the user tightened the explosion line to `> 11.5 GiB`
- the remote owner surface is currently showing a persistent host-GUI background load around `3.7-4.2 GiB`
- under that background band, `b12` still looks too close to the edge
- the right response is to lower batch again, not to abandon the midpoint probe

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- set:
  - `training.batch_size = 8`
  - `structure_penalty_mode = stokes`
  - `w_stokes_viscous = 0.03`

Success condition:

- the packet launches and stays below the new `11.5 GiB` explosion line
- transfer style stays above the `0.05` late-Stokes point
- while LPIPS stays materially below the `0.02` late-Stokes point

Failure condition:

- even `b8` remains structurally too close to the host-side VRAM floor
- or the midpoint still sits strictly on the same tradeoff curve without improving either anchor
