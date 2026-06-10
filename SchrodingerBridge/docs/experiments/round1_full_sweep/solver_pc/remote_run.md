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
- First formal in-band relaunch on `2026-06-11`:
  - opening batch:
    - `16`
  - 30-second health sample:
    - `9334 MiB / 12288 MiB`
  - interpretation:
    - this is inside the requested `9.0-10.8 GiB` formal band
    - so `batch=16` is the first authoritative `solver_pc` launch setting
- First settled fast-eval point:
  - `epoch_0001`
  - transfer `CLIP-S / LPIPS = 0.7074 / 0.5621`
  - all-pairs `CLIP-S / LPIPS = 0.7170 / 0.5552`
  - wall `= 177.75s`
  - immediate read:
    - style score opens slightly above the current internal tangent late tail
    - but LPIPS is clearly worse than the stronger tangent structure points
    - therefore the family stays open, but the first point is not a promote signal
- Second settled fast-eval point:
  - `epoch_0002`
  - transfer `CLIP-S / LPIPS = 0.6974 / 0.5426`
  - all-pairs `CLIP-S / LPIPS = 0.7109 / 0.5368`
  - wall `= 178.42s`
  - immediate read:
    - LPIPS improved materially from the opening point
    - but style scores backed off on both transfer and all-pairs
    - so the line has active movement, but it is still too early to rank this family beyond “alive”

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_pc`
- Run name: `aaai2027_round1_solver_pc_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_pc_seed42_b8a2`
- Config: [aaai2027_round1_solver_pc_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_pc_seed42_b8a2.json)
- Manifest status: `running`
- Local fast root: [round1_solver_pc_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_fast_local)
- Local review root: [round1_solver_pc_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_solver_pc_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_switch_smoke_latest.json)
- Switch smoke row count: `1`
<!-- ROUND1_AUTO_STATUS:END -->
