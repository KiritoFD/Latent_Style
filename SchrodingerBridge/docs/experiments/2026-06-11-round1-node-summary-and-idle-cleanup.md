# Round 1 Node Summary And Idle Cleanup

Date: 2026-06-11

Purpose:

- record the current open-lane read for round 1
- document the local cleanup decisions made while the remote lane stays healthy
- update the near-term plan without changing the formal experiment contract

## Current Experiment Snapshot

- active formal lane:
  - `solver_unsb_cycle`
- authority root:
  - [round1_solver_unsb_cycle_remote_full_eval_pull](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_remote_full_eval_pull)
- latest settled fast-eval point:
  - `epoch_0007`
  - transfer `0.6904 / 0.5549`
  - all-pairs `0.7037 / 0.5453`
- current best reads inside the lane:
  - best transfer `CLIP-S`:
    - `epoch_0001`
    - `0.7057 / 0.5669`
  - best transfer `LPIPS`:
    - `epoch_0003`
    - `0.7027 / 0.5117`
  - best all-pairs `CLIP-S`:
    - `epoch_0003`
    - `0.7195 / 0.5024`
- convergence read:
  - `row_count = 7`
  - `since_last_pareto = 4`
  - `tail_flat = false`
  - `converged = false`
- remote live read after doc refresh:
  - `9514 MiB / 12288 MiB`
  - `epoch 8/48`
  - `step 468/629`
  - `loss=8.0018`
  - `tswd=4.7500`

## Cleanup Decisions

- keep the authoritative family evidence surface narrow:
  - per-epoch `metrics.csv` + `summary.json` under `round1_*_remote_full_eval_pull/`
  - family notes under `docs/experiments/round1_full_sweep/<family>/`
- do not let remote packet wrappers accumulate in the active root:
  - future `*_full_eval_fast_snapshot_*.tar` files are now ignored at `aaai2027/`
- do not let loose checkpoint drops accumulate in the active root:
  - future `epoch_*.pt` and `*_epoch_*.pt` drops are now ignored at `aaai2027/`
- keep packet telemetry distinct from evidence:
  - runtime watcher `json/jsonl/log` files remain operational artifacts
  - summary/decision docs remain the git-facing read surface

## Plan Update

1. Keep `solver_unsb_cycle` on the remote lane until the solver-family `6`-checkpoint patience rule is actually satisfied or a new Pareto point appears and resets the tail.
2. Do not spend local heavy-review budget on this family yet; the current curve is still too early for a meaningful shortlist.
3. Use local idle time for low-risk maintenance only:
   - doc refresh
   - directory/index simplification
   - ignore-rule tightening
   - infra/code cleanup that does not perturb the live training/eval contract
4. Keep tokenizer DINO families at the tail of the queue:
   - if they are opened next, prefer warm-start or reconstruction-pretrain entry before a full formal lane
   - do not preempt the current solver closure work with a DINO-heavy launch

## Theory Read

- the early `solver_unsb_cycle` curve looks like a trajectory-shaping solver rather than an immediate board winner:
  - `epoch_0001 -> epoch_0003` improves LPIPS materially
  - style peaks do not keep rising after the opening
  - `epoch_0004 -> epoch_0007` currently reads as rollback rather than a second frontier expansion
- if that pattern persists, the solver should be treated as:
  - a structure-preserving component that may still help in a later composite
  - not a standalone promotion candidate on the external board
- the practical consequence for round 1:
  - do not promote UNSB on internal curve motion alone
  - only keep it for integration if later closure shows a real structure benefit that survives composition with a stronger tokenizer or backbone family
