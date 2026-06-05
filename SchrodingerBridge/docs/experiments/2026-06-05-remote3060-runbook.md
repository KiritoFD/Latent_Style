# Remote 3060 Runbook

Date: 2026-06-05

Purpose:

- reduce repeated launch failures on the remote `RTX 3060`
- make remote runs reproducible without rereading chat history
- standardize sync, preflight, launch, monitor, and closure

## Machine contract

- host:
  - `100.115.18.62:2222`
- user:
  - `administrator`
- WSL distro:
  - `Ubuntu-26.04`
- authoritative repo root inside WSL:
  - `/mnt/i/Github/Latent_Style`

Do not assume:

- local branch state already exists on remote
- remote git branch is current
- remote working tree has the latest helper scripts

Treat local reviewed files as the source of truth, then push an explicit packet.

## The three failure classes

Most remote failures fall into one of these buckets:

1. launch-transport failure
2. path / workspace mismatch
3. runtime health failure

Diagnose in that order.

### 1. Launch-transport failure

Common symptoms:

- command works locally but not through `ssh`
- `wsl -d ... -- bash -lc '...'` breaks on quotes
- remote shell reports syntax errors before Python starts

Rule:

- do not build long one-liner launch commands through nested quoting unless there is no alternative
- prefer one of:
  - packet sync + remote `py_compile`
  - uploaded shell script under `_codex_tmp`
  - reviewed launcher script committed in repo

If the run must survive disconnects:

- Windows-side detached launch should use `schtasks`
- do not rely on `Start-Process` via SSH

Reference:

- [remote_server.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/remote_server.md)

### 2. Path / workspace mismatch

Common symptoms:

- checkpoint path exists locally but not remotely
- remote branch points at an older worktree
- dataset path in docs uses `/mnt/f/...` while active machine uses `/mnt/i/...`

Rules:

- verify the exact remote path before launch
- prefer current `/mnt/i/...` paths over historical `/mnt/f/...` notes
- verify the actual remote repo root:
  - `/mnt/i/Github/Latent_Style`
- verify the actual data root before formal launch

Do not trust:

- old path mentions in older notes
- implicit branch assumptions
- unverified checkpoint reuse

### 3. Runtime health failure

Common symptoms:

- import failure after sync
- OOM
- train loop starts but throughput collapses
- evaluator starts but stalls on the wrong test root

Rules:

- run remote `python3 -m py_compile` immediately after packet sync
- before formal training, verify:
  - dataset root exists
  - checkpoint root exists
  - output root is writable
  - GPU is actually idle enough
- first heartbeat must be checked within `30s`
- do not leave a fresh run uninspected for `120s`

## Stable remote workflow

### Step 1. Preflight

Minimum checks:

```powershell
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader"
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "wsl -d Ubuntu-26.04 --cd /mnt/i/Github/Latent_Style --exec pwd"
```

For formal training, record:

- GPU memory used / total
- expected output root
- exact dataset root
- exact launcher command or wrapper path

### Step 2. Packet sync

Do not assume the remote checkout is current.

Use a reviewed push script, for example:

- [push_remote_samst_step_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_samst_step_packet.py)
- [push_remote_latent_baseline_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_latent_baseline_packet.py)

Important:

- these sync scripts are run from the local machine
- they should not be invoked from inside the remote WSL shell

After sync:

- immediately run remote `py_compile`
- if compile fails, stop and repair locally first

### Step 3. Launch

Use the smallest stable launch surface:

- short repo script if possible
- wrapper `.sh` in `_codex_tmp` if quoting is fragile
- `schtasks` only when the job must detach and survive SSH disconnect

Avoid:

- very long `ssh "... wsl ... bash -lc 'python ... --many flags ...'"` strings

### Step 4. Monitor

First monitoring window:

- inspect within `30s`
- confirm:
  - Python process exists
  - log file is growing
  - GPU memory moved into the expected band

Formal run policy:

- remote `3060` paper-facing runs should usually sit around the expected formal memory band for that method
- under-cap runs are smoke or calibration, not formal evidence

### Step 5. Closure

A run is not paper-safe just because training finished.

Required closure is:

1. retained checkpoint path
2. primary train log
3. evaluator output directory
4. `summary.json`
5. `metrics.csv`
6. `aggregate_targetwise_artfid.json` when required

## Current path truths worth reusing

Distinct5-512 active roots:

- latent train:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- latent test:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/test`
- classview test:
  - `/mnt/i/wikiart_distinct5_samam_512_classview/test`

Legacy256 active roots:

- latent:
  - `G:\GitHub\Latent_Style\latent-256`
- eval images:
  - `G:\GitHub\Latent_Style\style_data\overfit50`

## Git boundary that matters

Not every code path is tracked by the same git root.

Current important case:

- `Related_Works/repos/SaMam` is a nested git repo
- outer `Latent_Style` commits do not automatically capture its changes

Implication:

- if remote execution depends on local `SaMam` edits, packet sync must explicitly include those files
- do not assume outer-repo push is sufficient

## Minimum logging contract for every remote run

Always record:

1. exact launcher command
2. remote workspace root
3. dataset root
4. output root
5. log path
6. first healthy heartbeat time
7. finish status or failure reason

If this metadata is missing, the run is not auditable enough for paper use.

## Practical default

For new paper-facing remote work, prefer this order:

1. local code change
2. local `py_compile`
3. packet sync
4. remote `py_compile`
5. remote preflight
6. launch
7. `30s` health check
8. periodic monitoring
9. evaluator closure

This order is slower than ad hoc launching, but it fails less often.
