# AAAI2027 Remote Launcher

Date: 2026-06-06

Purpose:

- make the remote `A1/A2` packet launch reproducible from the repo
- avoid fragile SSH -> PowerShell -> WSL detach chains
- preserve the remote `3060` single-run cap under `< 11.0 GiB`

## Helper

- [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)

## Current launch contract

The launcher now writes two remote artifacts under `SchrodingerBridge/_codex_tmp/`:

1. a WSL shell launcher:
   - runs `src/run.py --config ...`
   - appends to the packet `remote_train.log`
2. a Windows-side PowerShell launcher:
   - registers a one-shot scheduled task on the remote host
   - starts that task immediately
   - lets the task own the foreground `wsl.exe ... bash <launcher.sh>` process

This replaced the older in-session detach path because that path allowed the
remote WSL instance itself to be shut down shortly after launch.

## Why the scheduled-task path is required

Observed failure of the old path:

- `nohup` or in-WSL `tmux` could start training
- but WSL was then shut down by the host after the SSH-linked launch returned
- `dmesg` showed repeated `systemd-shutdow` and filesystem unmounts

Operational conclusion:

- the correct detach boundary is the remote Windows host
- not a nested background process inside WSL

## What the helper still checks before launch

1. sync reviewed `src/` and `configs/aaai2027/`
2. resolve merged config and `checkpoint.save_dir`
3. verify key Python entrypoints with remote `py_compile`
4. refuse launch unless remote total GPU memory is inside the idle gate:
   - `<= 1500 MiB`

## Remote contract

- host:
  - `100.115.18.62:2222`
- WSL distro:
  - `Ubuntu-26.04`
- remote workspace root:
  - `/mnt/i/Github/Latent_Style`
- default Python:
  - `/home/xy/venvs/samam312/bin/python`

## Example

```bash
python SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py \
  --config SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json
```

Expected immediate output:

- `prelaunch_gpu_memory_used_mib=...`
- `STARTED TASK=...`

## Current validated result

The repaired launcher has already been validated on:

- `executor_promotion_h_e1_seed42_b44`

Verified behavior:

- the packet stayed alive long enough to pass first-health
- observed runtime memory was around `9006 MiB`
- this remained below the hard `< 11.0 GiB` cap
- the queue watcher was then able to continue into the first `A2` arm
