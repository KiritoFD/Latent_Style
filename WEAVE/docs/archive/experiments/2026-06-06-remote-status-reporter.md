# Remote Status Reporter

Date: 2026-06-06

Purpose:

- provide one command for the current remote AAAI2027 autonomy state
- avoid repeated manual inspection across:
  - remote `train.log`
  - retained checkpoints
  - GPU snapshot
  - `A1` process state
  - local watcher logs

## Helper

- [report_remote_aaai2027_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_remote_aaai2027_status.py)

## What it reports

The JSON output includes:

1. remote GPU rows from `nvidia-smi`
2. hard runtime cap in MiB plus a current within-cap boolean
3. latent `SaMam` latest parsed step
4. latent `SaMam` latest parsed `it/s` and ETA to `step_5000`
5. retained checkpoint list excluding `last.ckpt`
6. `A1` process and remote-log existence state
7. local watcher PIDs, process info, and recent log tails

## Example

```bash
python SchrodingerBridge/tools/experiments/report_remote_aaai2027_status.py
```

## Intended use

Use this before manual intervention or when refreshing docs, instead of
repeating several separate shell checks.

Current role in the queue:

- this is the quickest way to verify whether the bounded latent side quest is
  still the only active lane
- once `A1` starts, the same output becomes the single check for:
  - `process_alive`
  - `log_exists`
  - current GPU memory remaining strictly below the hard `< 11.0 GiB` cap
