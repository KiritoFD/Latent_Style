# img2img-turbo Remote Smoke Status

Date: 2026-06-06

Scope:

- baseline family: `img2img-turbo` / `CycleGAN-Turbo`
- target surface: `Distinct5-512`
- machine: remote `RTX 3060 WSL`
- purpose: first remote smoke for the `SD1.5 / large-prior adaptation` baseline arm

## Summary

This lane is no longer blocked on launcher fragility or missing wrapper glue.

The current blocker has been narrowed to one specific infrastructure issue:

- the remote machine cannot reach `huggingface.co`
- the required upstream base model `stabilityai/sd-turbo` is not cached there yet

Everything before that point is now exercised successfully.

## What is already closed

### 1. Remote launcher integration

Closed:

- `img2img-turbo` now has a reviewed host-owned remote launcher entry:
  - [launch_remote_img2img_turbo_smoke.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_img2img_turbo_smoke.py)
- it reuses:
  - [launch_remote_wsl_command.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py)

### 2. Remote smoke dataset materialization

Closed:

- the remote machine can build a smoke-only `Distinct5` dataset packet from the
  available `classview/test` surface
- current smoke builder settings:
  - `train-root = /mnt/i/wikiart_distinct5_samam_512_classview/test`
  - `test-root = /mnt/i/wikiart_distinct5_samam_512_classview/test`
  - `30` train images per style
  - `30` test images per style
- this is explicitly **smoke-only**, not paper-safe train/test protocol

### 3. Remote repo sync

Closed:

- the `img2img-turbo` repo code is now being explicitly synced to the remote
  owner workspace by the smoke launcher
- the remote owner surface did not previously contain this repo

### 4. Python and dependency chain

Closed progression:

1. initial remote smoke used the shared `samam312` env and failed on missing
   `wandb`
2. after installing `wandb`, the same env failed on missing `peft`
3. after adding the missing packages, the shared env still had incompatible
   versions for:
   - `transformers`
   - `diffusers`
   - `peft`
4. a dedicated remote env was then created:
   - `/home/xy/venvs/img2img_turbo312`
5. that env was moved to a more compatible stack:
   - `transformers==4.35.2`
   - `diffusers==0.25.1`
   - `peft==0.7.1`
   - `accelerate==0.24.1`
   - `huggingface_hub==0.20.2`

Result:

- the smoke launcher now correctly binds to the dedicated env
- the training entry progresses beyond import-time dependency failure

## Remote snapshot recovery

Local `sd-turbo` evidence was later promoted into a remote owner-side snapshot:

- local cache source:
  - `C:\Users\xy\.cache\huggingface\hub\models--stabilityai--sd-turbo`
- remote tar:
  - `/mnt/i/Github/Latent_Style/Related_Works/runs/hf_snapshots/sd_turbo_snapshot_20260606.tar`
- remote extracted snapshot:
  - `/mnt/i/Github/Latent_Style/Related_Works/runs/hf_snapshots/sd_turbo_snapshot_20260606`

In parallel, the code path was repaired so the training stack can actually use a
local pretrained root instead of hard-coding `stabilityai/sd-turbo` in:

- `run_img2img_turbo_distinct5_smoke.py`
- `src/train_cyclegan_turbo.py`
- `src/cyclegan_turbo.py`
- `src/model.py`

## Current active blocker

Latest smoke run:

- run root:
  - `/mnt/i/Github/Latent_Style/Related_Works/runs/img2img_turbo_distinct5_remote_smoke_20260606_061608`

Latest hard failure:

- the lane now reaches real training
- low-VRAM smoke settings used:
  - `mixed_precision = fp16`
  - `gradient_checkpointing = on`
  - `allow_tf32 = on`
  - `train_batch_size = 1`
- failure point:
  - first training step
  - `torch.OutOfMemoryError`

Interpretation:

- the lane is no longer blocked on missing model files
- the lane is no longer blocked on Python or dependency incompatibility
- the current blocker is now a **true runtime memory overflow** on the reviewed
  `3060`

Observed evidence:

- by the time of failure, the smoke had already completed:
  - reference image preparation
  - FID reference feature preparation
  - LPIPS setup
- the OOM happens inside the actual generator forward/backward path

## Immediate next action

Do next:

1. treat the current `img2img-turbo` line as **runnable but not yet
   machine-safe**
2. only continue this lane with a concrete memory-reduction move, for example:
   - `xformers`
   - smaller validation/reference packet
   - reduced LoRA rank
   - smaller train prep if the protocol is explicitly re-scoped
3. do not promote the current smoke into any paper-facing same-cost row

Do not do next:

- do not continue mutating the shared `samam312` env
- do not claim this baseline is ready for same-cost or convergence yet
- do not interpret the current smoke as a model-quality result
