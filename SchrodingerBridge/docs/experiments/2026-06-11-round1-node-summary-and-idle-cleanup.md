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
  - `epoch_0008`
  - transfer `0.6955 / 0.5184`
  - all-pairs `0.7121 / 0.5088`
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
  - `row_count = 8`
  - `since_last_pareto = 5`
  - `tail_flat = false`
  - `converged = false`
- remote live read after doc refresh:
  - `10267 MiB / 12288 MiB`
  - `epoch 9/48`
  - `step 286/629`
  - `loss=7.8823`
  - `tswd=6.2812`

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
  - `epoch_0004 -> epoch_0007` read as rollback
  - `epoch_0008` then rebounds materially on both style and LPIPS relative to `epoch_0007`
- the current sharper read:
  - `epoch_0008` is still not a new Pareto point over `epoch_0003`
  - but it is strong enough to invalidate any premature "the tail is just monotonic decay" story
  - this family still needs at least one more settled point before the solver patience rule can close cleanly
- if that pattern persists, the solver should be treated as:
  - a structure-preserving component that may still help in a later composite
  - not a standalone promotion candidate on the external board
- the practical consequence for round 1:
  - do not promote UNSB on internal curve motion alone
  - only keep it for integration if later closure shows a real structure benefit that survives composition with a stronger tokenizer or backbone family
