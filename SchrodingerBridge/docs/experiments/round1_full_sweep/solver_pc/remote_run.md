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
  - `10344 MiB / 12288 MiB`
  - `util=94%`
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
  - `epoch_0033`
  - transfer `0.6904 / 0.5026`
  - full `0.7092 / 0.4913`
  - wall `255.84s`

## Operational Read

- `solver_pc` is not monotone; it is cycling between structure repair and style recovery.
- The family is still alive because real Pareto updates kept reappearing after apparent rollbacks.
- `epoch_0021` was a sharp rollback on both style and LPIPS versus `epoch_0020`.
- `epoch_0022` repaired most of that LPIPS damage, but still stayed below the frontier anchored by `epoch_0017`.
- `epoch_0023` improved LPIPS slightly again, but style softened and the point still remained below the frontier.
- `epoch_0024` then rolled back clearly on both style and LPIPS.
- `epoch_0025` and `epoch_0026` then repaired part of that rollback, but still stayed below the frontier anchored by `epoch_0017`.
- `epoch_0027` then rolled back again on LPIPS while style recovered only slightly.
- `epoch_0028` remained another non-frontier tail point.
- `epoch_0029` repaired a little, but `epoch_0030` still remained non-frontier.
- `epoch_0031` still remained non-frontier after that small repair.
- `epoch_0032` then regressed sharply on LPIPS and also took a large eval wall-time spike.
- `epoch_0033` repaired part of that LPIPS collapse, but still remained non-frontier.
- `epoch_0018-0033` are now sixteen consecutive non-frontier points, so the lane is extremely deep into the post-patience tail.
- The remote fast-eval contract is working, but local docs must distinguish:
  - locally pulled curve points
  - remote scan points still mid-write
- That distinction is now part of the sync tooling to avoid false closure reads.
- Pending-only remote epochs no longer dirty tracked docs during active runs.
- Remote fast-eval watchers and local packet-sync watchers now also self-exit once a family leaves `running` and no backlog remains.

## Next Action

- Continue the same lane with no batch change.
- Keep syncing each retained checkpoint.
- The next settled point should be judged almost entirely through the flat-tail condition:
  - if it still fails to create a new Pareto point and the tail finally flattens, close the family
  - if the tail is still not flat, keep the lane open despite the long non-frontier streak

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
- Remote GPU live sample:
  - `10344 MiB / 12288 MiB`, `util=92%`
  - `band_status=in_band`
  - `formal_status=formal_in_band`
- Remote train progress:
  - `epoch 33/48`
  - `step 149/590`
  - `loss=7.8960`
  - `tswd=5.6562`
<!-- ROUND1_AUTO_STATUS:END -->
