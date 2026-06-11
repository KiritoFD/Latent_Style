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
  - `epoch_0014`
  - transfer `0.6929 / 0.5097`
  - all-pairs `0.7097 / 0.5009`
- current best reads inside the lane:
  - best transfer `CLIP-S`:
    - `epoch_0001`
    - `0.7057 / 0.5669`
  - best transfer `LPIPS`:
    - `epoch_0009`
    - `0.6996 / 0.4421`
  - best all-pairs `CLIP-S`:
    - `epoch_0009`
    - `0.7245 / 0.4311`
- convergence read:
  - `row_count = 14`
  - `since_last_pareto = 5`
  - `tail_flat = false`
  - `converged = false`
- remote live read after doc refresh:
  - `9515 MiB / 12288 MiB`
  - `epoch 14/48`
  - `step 25/629`
  - `loss=7.9068`
  - `tswd=5.5625`

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
  - `epoch_0009` is a real new Pareto point
  - it becomes the best all-pairs style point and the best LPIPS point inside this family
  - it resets the solver patience clock and invalidates the earlier near-closure interpretation
  - `epoch_0010` then softens from `epoch_0009`, so the frontier reactivation is real but not yet stable
  - `epoch_0011` continues that softening, which makes the near-term read more like "frontier spike followed by weaker confirmations" than "stable new plateau"
  - `epoch_0011` also shows a large eval wall-time jump to about `325s`, which is an efficiency anomaly worth watching if it repeats
  - `epoch_0012` recovers materially over `epoch_0011`, but it still remains clearly below the `epoch_0009` frontier on both style and LPIPS
  - `epoch_0013-0014` stay in the same regime as `epoch_0012`: better than the `epoch_0011` trough, but still below `epoch_0009` on both style and LPIPS
- if that pattern persists, the solver should be treated as:
  - a structure-preserving component that may still help in a later composite
  - but it now also deserves renewed attention as a possible standalone keep candidate
- the practical consequence for round 1:
  - do not promote UNSB on internal curve motion alone
  - but do reopen the possibility that this solver family could survive round-1 closure on its own curve, not only as a later composite ingredient
  - the next key test is whether `epoch_0015+` keep climbing back toward `epoch_0009` or whether `epoch_0012-0014` were only a weak recovery shelf inside a broader post-frontier fade
