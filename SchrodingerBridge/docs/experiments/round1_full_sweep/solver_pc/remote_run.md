# solver_pc Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_pc_seed42_b8a2`
- First formal launch read on `2026-06-11`:
  - opening batch:
    - `8`
  - 30-second health sample:
    - `5216 MiB / 12288 MiB`
  - interpretation:
    - this is far below the requested `9.0-10.8 GiB` formal band
    - so `batch=8` is only a calibration starting point
  - next recalibration target:
    - `batch=14`
- Second calibration read on `2026-06-11`:
  - opening batch:
    - `14`
  - 30-second health sample:
    - `8226 MiB / 12288 MiB`
  - interpretation:
    - still below the effective formal floor
    - so `batch=14` remains a calibration-only opening
  - next recalibration target:
    - `batch=16`

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_pc`
- Run name: `aaai2027_round1_solver_pc_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_pc_seed42_b8a2`
- Config: [aaai2027_round1_solver_pc_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_pc_seed42_b8a2.json)
- Manifest status: `planned`
- Local fast root: [round1_solver_pc_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_fast_local)
- Local review root: [round1_solver_pc_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_solver_pc_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_switch_smoke_latest.json)
- Switch smoke row count: `1`
<!-- ROUND1_AUTO_STATUS:END -->
