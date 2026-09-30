# solver_unsb_cycle Plan

- Wave: `wave2_solver`
- Axis: `solver`
- Notes: UNSB-inspired stochastic bridge solver with cycle-consistency support.

## Launch Intent

- Parent:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json)
- Canonical family config:
  - [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Manifest status:
  - `planned`
- Queue position:
  - next solver-family candidate after `solver_pc` truly closes

## Formal Target

- First launch target:
  - `batch=8`
  - `accumulation_steps=2`
  - `num_epochs=48`
  - `virtual_length_multiplier=0.5`
- First health gate:
  - check within `30s`
  - preferred band `9216-11059 MiB`
  - stop or recalibrate above `11571 MiB`
- Eval contract:
  - remote all-ckpt `CLIP-S + LPIPS` remains the convergence authority
  - local heavy review stays deferred until family closure

## Expected Read

- This family should be judged as a solver-style structural alternative, not as a tokenizer experiment.
- The practical goal is to see whether the stochastic bridge plus cycle term creates a different style/structure frontier than `solver_pc` and `solver_tangent_rk`.
- No promotion is allowed on cheap internal lift alone:
  - it still needs the same deep review package before any keep/reject decision
