# Remote Autonomy Status

Date: 2026-06-06

Purpose:

- record the current autonomous execution surface in one place
- avoid reconstructing live state from multiple watcher logs
- keep the remote `3060` queue auditable while the bounded latent side quest is
  still occupying the only allowed GPU lane

## Current remote lane

Active training lane:

- remote run:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4`
- latest observed progress:
  - around `Epoch 0 step 3508`
- current retained checkpoint state:
  - only `step_checkpoints/last.ckpt`
  - first numbered retained checkpoint still waits for `step_5000`
- current remote GPU snapshot:
  - `7459 MiB / 12288 MiB`
  - still below the formal hard stop:
    - `< 11.0 GiB`

Interpretation:

- the remote lane is healthy but still blocked on the first retained checkpoint
- no paper-facing `A1` packet has started yet

## Active local watchers

Latent handoff watcher:

- script:
  - [watch_remote_latent_samam_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_latent_samam_handoff.py)
- pid:
  - `36376`
- start time:
  - `2026-06-06 00:41:06`
- stdout:
  - [watch_remote_latent_samam_handoff.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.out.log)
- stderr:
  - [watch_remote_latent_samam_handoff.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.err.log)
- current read:
  - still polling every `60s`
  - latest dry-run remains:
    - `retained_checkpoints=[]`
    - `latent_pid=414`
    - `a1_remote_log_exists=False`

Post-A1 queue watcher:

- script:
  - [watch_remote_aaai2027_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_aaai2027_queue.py)
- pid:
  - `35444`
- start time:
  - `2026-06-06 00:49:09`
- stdout:
  - [watch_remote_aaai2027_queue.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.out.log)
- stderr:
  - [watch_remote_aaai2027_queue.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.err.log)
- current read:
  - waiting for:
    - [executor_promotion_h_e1_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json)
  - latest loop still reports:
    - `process_alive=False`
    - `log_exists=False`

## Next automatic transitions

Expected order:

1. latent watcher sees the first retained `SaMam` checkpoint
2. latent watcher stops the side quest
3. latent watcher waits for remote total `memory.used <= 1500 MiB`
4. latent watcher launches `A1`
5. queue watcher sees `A1` process plus remote log
6. queue watcher performs first-health validation on `A1`
7. queue watcher waits for `A1` to finish
8. queue watcher continues:
   - `A2_softterm18_sem010`
   - `A2_softterm18_sem012`
   - `A2_softterm16_sem012`

## Current blocker

Nothing is failing at the orchestration layer right now.

The only active blocker is:

- latent `SaMam` still has not produced the first numbered retained checkpoint

That means the queue is delayed by remote runtime, not by missing scripts,
missing configs, or local GPU dependence.
