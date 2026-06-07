# Remote 3060 Runbook

Date: 2026-06-05

Purpose:

- reduce repeated launch failures on the remote `RTX 3060`
- make remote runs reproducible without rereading chat history
- standardize sync, preflight, launch, monitor, and closure
- enforce a hard single-run explosion boundary of `11.5 GiB`
- treat the host-owned remote launcher as the default entrypoint for formal paper runs

## Machine contract

- host:
  - `100.115.18.62:2222`
- user:
  - `administrator`
- WSL distro:
  - `Ubuntu-26.04`
- authoritative repo root inside WSL:
  - `/mnt/i/Github/Latent_Style`
- hard VRAM cap for paper-facing runs:
  - treat any reading above `11.5 GiB` as exploded

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
  - `SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py`
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

- [launch_remote_wsl_command.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py)
- [launch_remote_distinct5_latent_baseline.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_distinct5_latent_baseline.py)
- [push_remote_samst_step_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_samst_step_packet.py)
- [push_remote_latent_baseline_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_latent_baseline_packet.py)
- [handoff_remote_latent_samam_to_a1.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py)
- [watch_remote_latent_samam_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_latent_samam_handoff.py)

Important:

- these sync scripts are run from the local machine
- they should not be invoked from inside the remote WSL shell

After sync:

- immediately run remote `py_compile`
- if compile fails, stop and repair locally first

### Step 3. Launch

Use the smallest stable launch surface:

- host-owned remote launcher for any formal paper-facing task:
  - `SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py`
- short repo script if possible
- wrapper `.sh` in `_codex_tmp` if quoting is fragile
- `schtasks` only when the job must detach and survive SSH disconnect

Avoid:

- very long `ssh "... wsl ... bash -lc 'python ... --many flags ...'"` strings
- launching a new packet while remote total `memory.used` is still above:
  - `1500 MiB`

### Step 4. Monitor

First monitoring window:

- inspect within `30s`
- confirm:
  - Python process exists
  - log file is growing
  - GPU memory moved into the expected band
- the current `watch_remote_latent_samam_handoff.py` helper can perform this
  first-health check automatically after it launches `A1`

Formal run policy:

- remote `3060` paper-facing runs should usually sit around the expected formal memory band for that method
- treat `11.5 GiB` as the hard explosion stop, not a soft target
- under-cap runs are smoke or calibration, not formal evidence

Concurrency rule:

- do not overlap latent baseline runs on the `3060`
- only one training lane may hold GPU at a time unless a measured combined peak is still strictly below `11.5 GiB`
- if a second lane pushes total usage near or above `11.5 GiB`, stop it immediately and relaunch later as a single-run lane
- when handing off between lanes, wait for remote total `memory.used <= 1500 MiB`
  before launching the next packet

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
- latent held-out:
  - `/mnt/i/wikiart_distinct5_latents_512_ema_test`
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

## 2026-06-05 latent baseline incident log

This is the concrete failure chain that motivated this runbook.

Target:

- task:
  - latent `SaMam` baseline on `legacy256_overfit50`
- remote session:
  - `samam_latent_legacy256`
- remote output root:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_remote`
- log:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_remote/train.log`

Observed sequence:

1. `/usr/bin/python3` launched, but failed immediately:
   - `ModuleNotFoundError: No module named 'pytorch_lightning'`
2. switched to `/home/xy/venvs/samam312/bin/python`
3. launch then failed on import path setup:
   - `ModuleNotFoundError: No module named 'TRAIN'`
4. local fix was required in:
   - [train_SaMam_latent.py](/G:/GitHub/Latent_Style/Related_Works/repos/SaMam/TRAIN/train_SaMam_latent.py)
   - the script now derives repo root from `Path(__file__).resolve().parents[1]` instead of the caller cwd
5. after resync, training reached evaluator-side VAE import and failed again:
   - `ImportError: cannot import name 'Dinov2WithRegistersConfig' from 'transformers'`
6. the trigger was not the model itself, but `diffusers` importing extra autoencoder modules through a broad package import path
7. upgrading `transformers` blindly made the environment worse:
   - newer `transformers` then broke `mamba-ssm` imports in `samam312`

Operational lesson:

- do not patch remote environments first when the failure originates from a broad local import path
- prefer narrowing the local import site, resyncing, and retrying with the original env contract
- for this family, `diffusers` and `transformers` must be treated as a coupled constraint because `mamba-ssm` also depends on that env

Follow-up fix that actually moved the run forward:

- restore `/home/xy/venvs/samam312` `transformers` to `4.41.2`
- downgrade `diffusers` in that env to `0.29.2`
- install `modelscope` so `download_vae_with_fallback()` can fetch `stabilityai/sd-vae-ft-ema` when `huggingface.co` is unreachable from WSL
- run latent `SaMam` with `32-true` precision on this machine; `16-mixed` hit a `mamba_ssm` selective-scan dtype failure during sanity-check evaluation
- for latent token grids, `Related_Works/repos/SaMam/ARCHI/StyleEmbedder.py` must avoid blindly downscaling tiny `4x4` maps

Observed network fact on this machine:

- `huggingface.co` may be unreachable from remote WSL even when `pip` to the Tsinghua mirror works
- if VAE cache is missing, `modelscope` fallback is the fastest repair path currently validated on this machine

Current latent-method lessons:

- `SaMam` latent on `legacy256_overfit50` is stable only after:
  - narrow VAE import path
  - `transformers 4.41.2`
  - `diffusers 0.29.2`
  - `modelscope`
  - `32-true` precision
- `SaMST` latent on `legacy256_overfit50` needed three separate wrapper fixes before it could stay alive past startup:
  - correct workspace-root derivation for `SchrodingerBridge/src`
  - style enumeration from latent style subdirectories instead of flat files
  - float32 decode output before VGG feature extraction

Current resource rule:

- remote `3060` paper runs must stay at or below `11.5 GiB`
- do not overlap latent baseline probes if combined usage pushes the card above that cap
- on 2026-06-05, concurrent `SaMam` + `SaMST` pushed usage to about `12.0 / 12.3 GiB`, so the `SaMST` probe was stopped and `SaMam` was kept as the active formal run
- current verified single-run reference:
  - `SaMam legacy256` at `batch=2`, `precision=32-true` used about `7.46 GiB`
- until `SaMam` yields a retained checkpoint, defer `SaMST` to a non-overlapping retry and start that retry from `batch=1`

## Remote launch pattern that proved structurally stable

For tmux-backed WSL launches, this pattern was reliable:

- call `wsl --exec tmux ...` directly
- avoid wrapping tmux startup inside nested `bash -lc` quoting

Example shape:

```powershell
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 ^
  "wsl -d Ubuntu-26.04 --exec tmux new-session -d -s SESSION -c /mnt/i/Github/Latent_Style PYTHON SCRIPT ..."
```

The exact command still depends on the run, but the structural rule is durable:

- direct `wsl --exec`
- direct `tmux`
- short argument surface
