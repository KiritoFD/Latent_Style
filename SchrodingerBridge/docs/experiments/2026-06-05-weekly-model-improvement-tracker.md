# Weekly Model Improvement Tracker

Date: 2026-06-05

Scope:

- one-week autonomous execution plan for AAAI 2027 paper hardening
- remote `RTX 3060` WSL only
- priority order:
  - improve our model
  - repeat and validate our model
  - add cheap reviewer-directed controls
  - keep latent/baseline work bounded

Primary plan:

- `docs/plans/2026-06-05-one-week-model-improvement-plan.md`

## Current keep / stop policy

Keep spending formal remote budget on:

1. executor-side model improvement
2. narrow softening / stability sweeps on the current paper-facing family
3. repeatability packets for any promoted improved point
4. cheap reviewer controls tied to concrete questions

Stop spending formal remote budget on:

- more endpoint-only reruns
- more semantic-vs-random reruns
- broad speed rhetoric experiments
- turning `latent SaMam` into the main lane

## Day 1 packet

### A1. Executor-side promotion

Intent:

- port the landed `executor-only` localization signal onto the paper-facing
  `H` family instead of opening a new tokenizer family

Config:

- `configs/aaai2027/executor_promotion_h_e1_seed42_b44.json`

Checkpoint source:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/epoch_0001.pt`

Training rule:

- load only:
  - `style_tokenizer.*`
  - `style_spatial_id_16`
- train only the executor side

Status:

- `prepared`

### A2. Narrow mainline softening sweep

Reasoning:

- the current same-family path packet says weakening `w_kinetic` hurts quality,
  so this sweep does **not** reduce kinetic
- the softening lane therefore changes only:
  - terminal endpoint pressure
  - semantic routing temperature

Configs:

- `configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json`

Shared policy:

- keep `w_kinetic = 1.0`
- use the reviewed Distinct5 full-eval contract
- compare against current `H e1/e2` and `F e1`

Status:

- `prepared`

## Near-term comparison anchors

Primary anchors already in the ledger:

- `LBM-F e1`
- `LBM-H e1`
- `LBM-H e2`
- `LBM-K e1`

Use these as the first comparison surface before touching broader baseline work.

## Side quest budget

`latent SaMam`:

- allowed only as a short smoke lane later in the week
- not allowed to block the main improvement queue

## Next log entries to add

When the next packet launches, append:

1. exact remote launcher / command
2. save dir
3. GPU memory band
4. first-health heartbeat
5. completion or failure note

## 2026-06-05 preflight snapshot

What is already confirmed:

- remote host `100.115.18.62:2222` is reachable
- remote WSL project root resolves cleanly to:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge`
- Windows-side GPU snapshot before launch attempt:
  - `memory.used = 991 MiB / 12288 MiB`
  - `utilization.gpu = 18%`
  - `power.draw = 15.16 W`

What is not yet trusted:

- a shell-safe one-line launch contract for this host
- path-truth for the exact reusable `H e1` checkpoint under the current remote
  workspace surface

Immediate next ops task:

- add a small remote launch / preflight wrapper that avoids the current
  PowerShell + SSH + WSL quoting ambiguity before the first formal `A1` launch

## 2026-06-06 live update

Remote latent side quest currently occupying the only allowed GPU lane:

- active run:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4`
- current GPU usage snapshot:
  - `7459 MiB / 12288 MiB`
  - safely below the hard stop:
    - `< 11.0 GiB`
- latest observed training progress:
  - around `Epoch 0 step 4251`
  - observed train rate:
    - about `0.76 step/s`
- rough remaining wall to first retained checkpoint:
  - about `16.4 min` from the latest heartbeat if throughput stays flat
- retained checkpoint status:
  - none yet
  - first save still waits for `step 5000`
- state persistence status:
  - `step_checkpoints/last.ckpt` now exists
  - retained numbered checkpoints still have not reached the first `5000-step`
    boundary
- concurrency enforcement:
  - keep this as the only GPU training lane
  - do not launch `A1`, `A2`, or `C1` until this lane is paused or closed
  - do not accept any launch plan whose measured peak could reach `11.0 GiB`

Interpretation:

- `latent SaMam` is healthy enough to keep running as a bounded side quest
- but it is currently blocking the main `A1/A2` queue because the remote `3060`
  must stay below the hard `< 11.0 GiB` single-run budget
- no `A1` launcher output has been created on the remote yet
  - the handoff order remains:
    - first retained `latent SaMam` checkpoint
    - then `A1`
    - then `C1`
    - then `C2` if reviewer sensitivity evidence is still needed

Ops progress landed during this update:

- added reviewed remote launcher helper:
  - [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)
- launcher purpose:
  - sync the current `src` plus `configs/aaai2027`
  - write a remote `_codex_tmp/*.sh` launch script
  - run targeted remote `py_compile`
  - start the packet via remote `schtasks`

What this unblocks:

- once the current `SaMam` lane yields a checkpoint or is explicitly paused,
  `A1` can be launched without reopening the earlier SSH + WSL quoting failure
  class

Additional review-risk reduction landed during this update:

- implementation clarity packet:
  - [2026-06-06-distinct5-implementation-clarity.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-distinct5-implementation-clarity.md)
- this note records the current Distinct5-512 train root, eval root, VAE
  preset, latent-scale contract, style list, and prototype-pairing cache
  contract for the active paper-facing `H` family

Next cheap reviewer control prepared:

- pairing-cache sensitivity packet:
  - [2026-06-06-pairing-cache-sensitivity-packet.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-pairing-cache-sensitivity-packet.md)
- launch config:
  - [pairing_cache_h_randompair_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/pairing_cache_h_randompair_seed42_b44.json)
- scope:
  - same paper-facing `H` surface
  - clear only the offline prototype pairing cache
  - short two-epoch random-pairing control

Additional cheap reviewer control now prepared:

- projection-count sensitivity packet:
  - [2026-06-06-projection-count-sensitivity-packet.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-projection-count-sensitivity-packet.md)
- launch config:
  - [projection_count_h_sem32_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/projection_count_h_sem32_seed42_b44.json)
- scope:
  - same paper-facing `H` surface
  - keep semantic axis selection fixed
  - reduce only `semantic_swd_num_projections`
  - short two-epoch sensitivity packet

A2 queue is now explicitly prepared as a packet, not just a plan mention:

- queue note:
  - [2026-06-06-a2-softening-queue.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-a2-softening-queue.md)
- queued arms:
  - [mainline_h_softterm18_sem010_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json)
  - [mainline_h_softterm18_sem012_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json)
  - [mainline_h_softterm16_sem012_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json)
- preflight status:
  - all three launcher dry-runs already resolve stable remote task names and
    remote log paths under the current `1500 MiB` prelaunch gate

Remote handoff helper now prepared:

- helper note:
  - [2026-06-06-samam-a1-handoff-helper.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-samam-a1-handoff-helper.md)
- helper script:
  - [handoff_remote_latent_samam_to_a1.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py)
- current dry-run state:
  - retained checkpoint list is still empty
  - latent `SaMam` pid is still alive as `pid 414`
  - `A1` remote log does not yet exist
  - helper therefore correctly refuses handoff until the first retained checkpoint appears

Auto-handoff watcher now prepared:

- watcher script:
  - [watch_remote_latent_samam_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_latent_samam_handoff.py)
- watcher role:
  - poll the handoff helper every `60s`
  - once the first retained checkpoint appears:
    - stop latent `SaMam`
    - wait until remote total `memory.used <= 1500 MiB`
    - launch `A1`
    - wait `30s` and record the first `A1` health heartbeat
- reason:
  - this removes manual polling while preserving the hard `< 11.0 GiB`
    single-lane rule on the remote `3060`
- active local watcher:
  - pid file:
    - [watch_remote_latent_samam_handoff.pid](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.pid)
  - stdout log:
    - [watch_remote_latent_samam_handoff.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.out.log)
  - stderr log:
    - [watch_remote_latent_samam_handoff.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.err.log)

Post-A1 queue watcher now prepared:

- queue watcher note:
  - [2026-06-06-aaai2027-queue-watcher.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-aaai2027-queue-watcher.md)
- queue watcher script:
  - [watch_remote_aaai2027_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_aaai2027_queue.py)
- default queue:
  - wait for `A1` to start and pass first health
  - wait for `A1` to finish
  - then continue `A2a -> A2b -> A2c` under the same `<= 1500 MiB`
    prelaunch idle gate
- active local watcher:
  - pid file:
    - [watch_remote_aaai2027_queue.pid](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.pid)
  - stdout log:
    - [watch_remote_aaai2027_queue.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.out.log)
  - stderr log:
    - [watch_remote_aaai2027_queue.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.err.log)
- current live state:
  - it is waiting for:
    - [executor_promotion_h_e1_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json)
  - current log shows:
    - `process_alive=False`
    - `log_exists=False`
  - this is expected until the latent handoff watcher actually starts `A1`

Runtime guard refinement now landed:

- `watch_remote_latent_samam_handoff.py` now rejects the first `A1` health
  heartbeat if observed GPU usage reaches `>= 11000 MiB`
- `watch_remote_aaai2027_queue.py` applies the same first-health guard to `A1`
  and every later queued `A2` arm
- `report_remote_aaai2027_status.py` now reports:
  - `hard_runtime_cap_mib`
  - `cap_status.max_observed_memory_mib`
  - `cap_status.within_hard_runtime_cap`
- the active local watcher instances were restarted after this patch so the live
  queue is already running on the stricter gate

Single-note autonomy snapshot now available:

- status note:
  - [2026-06-06-remote-autonomy-status.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-remote-autonomy-status.md)
- purpose:
  - one place for current remote step, watcher PIDs, watcher logs, and the
    next automatic queue transitions

Single-command live reporter now available:

- reporter:
  - [report_remote_aaai2027_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_remote_aaai2027_status.py)
- note:
  - [2026-06-06-remote-status-reporter.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-remote-status-reporter.md)
