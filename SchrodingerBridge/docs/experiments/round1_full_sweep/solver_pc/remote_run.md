# solver_pc Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_pc_seed42_b8a2`
- Authority packet root:
  - [round1_solver_pc_remote_full_eval_pull](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_remote_full_eval_pull)
- Fast curve note:
  - [fast_curve_read.md](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/solver_pc/fast_curve_read.md)

## Launch and Calibration

- `2026-06-11` calibration sequence:
  - `batch=8`
    - `5216 MiB / 12288 MiB`
    - below formal band
  - `batch=14`
    - `8226 MiB / 12288 MiB`
    - still below formal band
  - `batch=16`
    - `9334 MiB / 12288 MiB`
    - first authoritative formal opening
- Current live health sample:
  - `9343 MiB / 12288 MiB`
  - `util=89%`
  - read: lane is healthy and in-band

## Settled Curve Milestones

- Opening style peak:
  - `epoch_0001`
  - transfer `0.7074 / 0.5621`
  - full `0.7170 / 0.5552`
- Early structure knee:
  - `epoch_0009`
  - transfer `0.6911 / 0.4548`
  - full `0.7155 / 0.4429`
- Post-knee style recovery frontier:
  - `epoch_0013`
  - transfer `0.6968 / 0.5101`
  - full `0.7142 / 0.4996`
- Strongest late balanced tradeoff so far:
  - `epoch_0015`
  - transfer `0.6962 / 0.4854`
  - full `0.7165 / 0.4746`
- Latest locally pulled point:
  - `epoch_0022`
  - transfer `0.6888 / 0.4866`
  - full `0.7087 / 0.4774`
  - wall `177.18s`

## Operational Read

- `solver_pc` is not monotone; it is cycling between structure repair and style recovery.
- The family is still alive because real Pareto updates kept reappearing after apparent rollbacks.
- `epoch_0021` was a sharp rollback on both style and LPIPS versus `epoch_0020`.
- `epoch_0022` repaired most of that LPIPS damage, but still stayed below the frontier anchored by `epoch_0017`.
- `epoch_0018-0022` are now five consecutive non-frontier points, so the lane is firmly in the late patience band.
- The remote fast-eval contract is working, but local docs must distinguish:
  - locally pulled curve points
  - remote scan points still mid-write
- That distinction is now part of the sync tooling to avoid false closure reads.
- Pending-only remote epochs no longer dirty tracked docs during active runs.
- Remote fast-eval watchers and local packet-sync watchers now also self-exit once a family leaves `running` and no backlog remains.

## Next Action

- Continue the same lane with no batch change.
- Keep syncing each retained checkpoint.
- Do not hand the queue to the next family until `solver_pc` satisfies the solver-family closure rule.

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
