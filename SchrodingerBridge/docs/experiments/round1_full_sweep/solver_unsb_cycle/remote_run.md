# solver_unsb_cycle Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`

## Formal Opening

- Canonical config:
  - [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Calibration history:
  - `batch=8`
    - `5223 MiB / 12288 MiB`
    - under-band, rejected as calibration only
  - `batch=15`
    - `9677 MiB / 12288 MiB`
    - first authoritative in-band opening

## Current Read

- Current live sample:
  - `9677 MiB / 12288 MiB`
  - `util=83%`
  - `epoch 1/48`
  - `step 187/629`
  - `loss=8.4092`
  - `tswd=5.7812`
- Read:
  - the lane is formally alive
  - this family has now moved beyond calibration

## Next Action

- Let the lane continue.
- Keep remote fast-eval as the convergence authority.
- Do not spend local heavy review budget until the family has enough settled curve points to justify a shortlist.

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_unsb_cycle`
- Run name: `aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
- Config: [aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_unsb_cycle_seed42_b8a2.json)
- Manifest status: `running`
- Local fast root: [round1_solver_unsb_cycle_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_fast_local)
- Local review root: [round1_solver_unsb_cycle_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_solver_unsb_cycle_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Remote GPU live sample:
  - `9677 MiB / 12288 MiB`, `util=83%`
  - `band_status=in_band`
  - `formal_status=formal_in_band`
- Remote train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2_train.log`
- Remote train progress:
  - `epoch 1/48`
  - `step 187/629`
  - `loss=8.4092`
  - `tswd=5.7812`
<!-- ROUND1_AUTO_STATUS:END -->
