# `XPred + K_manifold + P_attn` Frontier -> Intermediate `Stokes` Fine-Tune (`b4a2`)

Date: 2026-06-07

Intent:

- keep the same midpoint `w_stokes_viscous = 0.03` probe
- reduce micro-batch to `4`
- add `accumulation_steps = 2`

Why this revision exists:

- the remote owner surface is still showing large, time-varying host-GUI background VRAM usage
- even after moving from `b16` to `b12` and then `b8`, launchability still depends too much on the host dropping to an unusually low idle band
- the training code already supports gradient accumulation, so the right next adjustment is:
  - lower micro-batch for VRAM
  - keep part of the effective batch via accumulation

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- extend the horizon to `16` epochs
- set:
  - `training.batch_size = 4`
  - `training.accumulation_steps = 2`
  - `structure_penalty_mode = stokes`
  - `w_stokes_viscous = 0.03`

Success condition:

- the packet launches and survives first-health below the `11.5 GiB` explosion line
- transfer style stays above the `0.05` late-Stokes point
- while LPIPS stays materially below the `0.02` late-Stokes point

Failure condition:

- the host-side VRAM floor is still too high even for this reduced micro-batch
- or the midpoint still fails to dominate the two established late-Stokes anchors
