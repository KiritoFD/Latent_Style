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

- Current interrupted state:
  - retained checkpoints stop at `epoch_0014`
  - the train log ends at:
    - `2026-06-11 13:33:06 +08:00`
    - `rc=143`
  - no remote train pid is currently alive for this family
- Read:
  - the lane was formally valid
  - but it is not currently active
  - the family therefore needs resume logic, not passive waiting
- First settled authority curve:
  - `epoch_0001`
    - transfer `0.7057 / 0.5669`
    - full `0.7150 / 0.5608`
  - `epoch_0002`
    - transfer `0.6975 / 0.5372`
    - full `0.7101 / 0.5312`
  - `epoch_0003`
    - transfer `0.7027 / 0.5117`
    - full `0.7195 / 0.5024`
  - `epoch_0004`
    - transfer `0.7001 / 0.5181`
    - full `0.7164 / 0.5097`
  - `epoch_0005`
    - transfer `0.6951 / 0.5144`
    - full `0.7119 / 0.5054`
  - interpretation:
    - `epoch_0001 -> 0002` was structure-favoring
    - `epoch_0003` then became the first clear best point on both structure and all-pairs style
    - `epoch_0004-0005` are two mild rollbacks after that improvement

## Next Action

- Keep the retained fast-eval packet authoritative through `epoch_0014`.
- Do not treat the lane as currently alive.
- Resume via bounded segmented continuation when the remote lane is actually free:
  - see [relaunch_prep.md](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/solver_unsb_cycle/relaunch_prep.md)
- Do not spend local heavy review budget until either:
  - a resumed run produces new retained checkpoints, or
  - the family is explicitly closed without resume

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
- Remote train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_solver_unsb_cycle_seed42_b8a2_train.log`
- Remote train pid: not alive
- Remote fast-eval pid count: `2`
<!-- ROUND1_AUTO_STATUS:END -->
