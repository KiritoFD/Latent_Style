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
2. latent `SaMam` latest parsed step
3. retained checkpoint list excluding `last.ckpt`
4. `A1` process and remote-log existence state
5. local watcher PIDs, process info, and recent log tails

## Example

```bash
python SchrodingerBridge/tools/experiments/report_remote_aaai2027_status.py
```

## Intended use

Use this before manual intervention or when refreshing docs, instead of
repeating several separate shell checks.
