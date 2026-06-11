# solver_unsb_cycle Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`

## Launch Readiness

- Status:
  - first formal handoff attempt made
  - current state is `recalibration_needed`
- Canonical config:
  - [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Initial formal target:
  - `batch=8`
  - `accumulation_steps=2`
  - `num_epochs=48`
  - `virtual_length_multiplier=0.5`

## First Calibration Read

- First direct launch sample:
  - `5223 MiB / 12288 MiB`
  - `epoch 1/48`
  - read: well below the formal floor
- Decision:
  - do not count this as a formal paper-facing lane
  - keep the family in `recalibration_needed`
  - raise the effective batch on the next retry

## Next Action

- Recalibrate `solver_unsb_cycle` upward from `batch=8`.
- Keep the same remote fast-eval authority contract once the family re-enters the formal VRAM band.

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_unsb_cycle`
- Run name: `aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Config: [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Manifest status: `recalibration_needed`
- Local fast root: [round1_solver_unsb_cycle_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_fast_local)
- Local review root: [round1_solver_unsb_cycle_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_solver_unsb_cycle_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Remote GPU live sample:
  - `5223 MiB / 12288 MiB`, `util=0%`
  - `band_status=under_band`
  - `formal_status=nonformal_under_band`
- Remote train progress:
  - `epoch 1/48`
  - `step 23/1180`
<!-- ROUND1_AUTO_STATUS:END -->
