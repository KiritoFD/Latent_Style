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

## Current active blocker

Latest smoke run:

- run root:
  - `/mnt/i/Github/Latent_Style/Related_Works/runs/img2img_turbo_distinct5_remote_smoke_20260606_052546`

Latest hard failure:

- `AutoTokenizer.from_pretrained("stabilityai/sd-turbo", ...)`
- remote failure:
  - cannot reach `https://huggingface.co`
  - cannot find the model in local cache

Interpretation:

- this is now an **offline model-cache blocker**, not a code or wrapper blocker

## Local evidence that can unblock it

Local machine already has a complete `sd-turbo` Hugging Face cache:

- `C:\Users\xy\.cache\huggingface\hub\models--stabilityai--sd-turbo`

Observed size:

- about `5.16 GB`

Observed snapshot:

- `snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2`

## Immediate next action

Do next:

1. copy the local `sd-turbo` Hugging Face cache to the remote owner surface
2. point the remote dedicated env at that cache
3. rerun the same `img2img-turbo` smoke launcher unchanged

Do not do next:

- do not continue mutating the shared `samam312` env
- do not claim this baseline is ready for same-cost or convergence yet
- do not interpret the current smoke as a model-quality result
