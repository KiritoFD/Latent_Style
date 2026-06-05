# Remote Autonomy Status

Date: 2026-06-06

Purpose:

- keep the remote `3060` execution state auditable from the repo
- record the current launcher contract after the WSL-detach failure audit
- show which queue stage is live without reconstructing it from chat

Quick status command:

- reporter:
  - [report_remote_aaai2027_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_remote_aaai2027_status.py)

## Current state

Latest verified state after the launcher repair:

- remote formal cap:
  - `< 11.0 GiB`
- latent `SaMam` side quest:
  - retained checkpoint reached at:
    - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4/step_checkpoints/step-step=005000.ckpt`
  - lane already stopped cleanly
- `A1` mainline packet:
  - `executor_promotion_h_e1_seed42_b44`
  - successfully launched on the remote host
  - first-health check passed under the repaired queue watcher
  - later exited, allowing the queue to continue
- current live queue stage:
  - `A2_softterm18_sem010`
- latest verified remote GPU sample while the repaired launcher was healthy:
  - about `9006 MiB / 12288 MiB`
  - still below the hard cap

## Root cause audit

The previous detached launch path was not stable enough.

Observed failure mode:

- `nohup` / in-WSL `tmux` launches could start training
- but the remote WSL instance was then shut down by the host
- `dmesg` showed repeated:
  - `systemd-shutdow`
  - filesystem unmount / remount

Interpretation:

- the failure was not a CUDA OOM
- it was a host-session-lifetime problem for WSL background launches

## Repaired launch contract

Current launcher behavior:

1. sync reviewed `src/` and `configs/aaai2027/`
2. write the remote WSL shell launcher
3. write a remote Windows-side launcher `.ps1`
4. have the remote Windows host register and start a one-shot scheduled task
5. let that scheduled task run `wsl.exe ... bash <launcher.sh>` in the foreground

Why this is the durable path:

- the training process no longer depends on the SSH session surviving
- the WSL lifetime is now owned by the remote host scheduler
- the launcher remains bounded by the same single-run cap checks

## Local orchestrators

Queue watcher:

- script:
  - [watch_remote_aaai2027_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_aaai2027_queue.py)
- current pid file:
  - [watch_remote_aaai2027_queue.pid](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.pid)
- current stdout:
  - [watch_remote_aaai2027_queue.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.out.log)
- current role:
  - attached to `A1`
  - observed `A1` finish
  - already launched `A2_softterm18_sem010`

Latent handoff watcher:

- previous pid file remains:
  - [watch_remote_latent_samam_handoff.pid](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.pid)
- current state:
  - stale / not running
- note:
  - this watcher is no longer authoritative for the live queue because the
    latent handoff has already been completed

## Remaining follow-through

- let the repaired queue watcher carry `A2_softterm18_sem010`
- if that arm exits cleanly, the same watcher should continue:
  - `A2_softterm18_sem012`
  - `A2_softterm16_sem012`
- after the queue lands, update the paper-facing experiment logs and promote or
  drop the softening arms based on full eval rather than launch success
