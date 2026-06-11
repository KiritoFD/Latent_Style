# solver_unsb_cycle Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`

## Launch Readiness

- Status:
  - waiting for `solver_pc` closure
- Canonical config:
  - [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Initial formal target:
  - `batch=8`
  - `accumulation_steps=2`
  - `num_epochs=48`
  - `virtual_length_multiplier=0.5`
- Queue rule:
  - do not launch until the manifest has zero `running` families
  - once launched, use the same remote fast-eval authority path as `solver_pc`

## First Read Checklist

- Verify switch smoke still reads `ok`.
- Check the first `30s` health sample.
- Keep memory within the formal `9.0-10.8 GiB` band.
- If the family opens under-band, treat it as calibration only and relaunch with a corrected batch.
- If it opens above `11.3 GiB`, stop immediately and record recalibration.

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_unsb_cycle`
- Run name: `aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Config: [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Manifest status: `planned`
- Local fast root: [round1_solver_unsb_cycle_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_fast_local)
- Local review root: [round1_solver_unsb_cycle_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_solver_unsb_cycle_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_switch_smoke_latest.json)
- Switch smoke row count: `1`
<!-- ROUND1_AUTO_STATUS:END -->





