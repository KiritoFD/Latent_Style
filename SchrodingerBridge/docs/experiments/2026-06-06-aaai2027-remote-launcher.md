# AAAI2027 Remote Launcher

Date: 2026-06-06

Purpose:

- promote `A1/A2` from "config prepared" to "remote launch surface ready"
- avoid repeating PowerShell + SSH + WSL quoting failures
- keep the launch contract explicit in the repo instead of hidden in chat

## Added helper

- [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)

## What it does

1. syncs the reviewed `SchrodingerBridge/src` tree and `configs/aaai2027`
2. resolves the target config and its merged `checkpoint.save_dir`
3. writes a remote launcher shell script under:
   - `SchrodingerBridge/_codex_tmp/`
4. verifies the key training/eval Python files with remote `py_compile`
5. launches the packet through remote `schtasks`

## Current remote contract

- host:
  - `100.115.18.62:2222`
- WSL distro:
  - `Ubuntu-26.04`
- remote workspace root:
  - `/mnt/i/Github/Latent_Style`
- default Python:
  - `/home/xy/venvs/samam312/bin/python`

## Example dry run

```bash
python SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py \
  --config SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json \
  --dry-run
```

## Intended first formal launches

`A1`

- `SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json`

`A2`

- `SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json`
- `SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json`
- `SchrodingerBridge/configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json`

## Boundary

- this helper does not justify overlapping runs
- the remote `3060` still follows the hard single-run VRAM ceiling
- use it only when the GPU lane is free enough for a paper-facing packet
