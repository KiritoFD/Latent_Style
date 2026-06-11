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

- Current resumed state:
  - bounded continuation is now validated at `batch=17`
  - retained checkpoints extend through `epoch_0018`
  - all retained checkpoints through `epoch_0018` have remote `CLIP-S + LPIPS`
  - the lane is not left resident between segments:
    - no train pid is currently alive
    - no active fast-eval pid is required once the packet settles
- Read:
  - the lane has moved past pure interruption recovery
  - the remaining question is convergence shape, not resume viability
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

- Keep the retained fast-eval packet authoritative through `epoch_0022`.
- Treat `batch=17` as the current canonical UNSB setting.
- Continue via further bounded segmented continuation rather than handing off to a different family yet:
  - `epoch_0018` created a new Pareto point and reset patience
- Do not spend local heavy review budget yet:
  - this family is still in convergence-shaping mode rather than closure mode
- latest bounded followup:
  - `epoch_0019-0024` are now settled
  - neither displaced `epoch_0018`
  - so the family remains open, with `since_last_pareto = 6`

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
  - `10624 MiB / 12288 MiB`, `util=96%`
  - `band_status=in_band`
  - `formal_status=formal_in_band`
- Remote train progress:
  - `epoch 22/22`
  - `step 326/555`
  - `loss=7.9858`
  - `tswd=6.4375`
<!-- ROUND1_AUTO_STATUS:END -->
