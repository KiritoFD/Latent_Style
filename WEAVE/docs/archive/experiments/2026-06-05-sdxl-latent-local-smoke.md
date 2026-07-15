# SDXL Latent Local Smoke on RTX 4070 Laptop

Date: 2026-06-05

Scope: local `Distinct5-512` smoke for the current LBM-K bridge using SDXL VAE
latents. This is a feasibility and stability note, not a paper-facing result.

## Goal

Answer one narrow question first:

- can the current bridge train on `SDXL`-space latents after image encoding;
- if yes, what is the smallest stable local path on the `RTX 4070 Laptop GPU`.

## Setup

Source image root:

- `F:/wikiart_distinct5_512_images/train`

Styles:

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

Smoke subset:

- `2` train images per style
- total `10` latent samples
- `1` epoch
- batch size `1`

Temporary smoke configs:

- `_codex_tmp/ema_latent_smoke_config.json`
- `_codex_tmp/sdxl_latent_smoke_config.json`
- `_codex_tmp/sdxl_fix_latent_smoke_config.json`

## Findings

### 1. Official `stabilityai/sdxl-vae` fp16 encode path is unsafe here

Command family:

- `python tools/encode_image_folder_latents.py ... --vae-model sdxl`

Observed outcome:

- all `10` encoded latent files contained only `NaN` values
- every file had `16384` `NaN` entries (`4 x 64 x 64`)
- training then failed at the first backward step with:
  - `FloatingPointError: Non-finite gradient detected at epoch=1 step=1 param=style_spatial_atoms_16`

Interpretation:

- the bridge failure is downstream of bad encoded latents
- this is not a generic training-memory issue

### 2. EMA control smoke is stable

Control root:

- `G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/ema_latent_smoke/train`

Outcome:

- training completed for `1` epoch
- checkpoint saved to:
  - `_codex_tmp/ema_latent_smoke_run/epoch_0001.pt`
- final log line reported:
  - compute time `6.1s`
  - peak CUDA allocated/reserved about `0.29 / 0.30 GB`

Purpose:

- confirms the tiny smoke protocol itself is valid
- isolates the failure to the SDXL latent path, not the batch-1 smoke setup

### 3. `madebyollin/sdxl-vae-fp16-fix` is a stable SDXL latent path

Command family:

- `python tools/encode_image_folder_latents.py ... --vae-model madebyollin/sdxl-vae-fp16-fix`

Observed latent statistics on the same 10-sample subset:

- bad file count: `0`
- mean: `0.1117`
- std: `0.8445`
- min/max: `-2.6543 / 3.2129`

Training outcome:

- training completed for `1` epoch
- checkpoint saved to:
  - `_codex_tmp/sdxl_fix_latent_smoke_run/epoch_0001.pt`
- final log line reported:
  - compute time `6.2s`
  - peak CUDA allocated/reserved about `0.29 / 0.30 GB`

Interpretation:

- the current bridge can train on SDXL-space latents locally
- the immediate blocker is VAE selection / loading stability, not model size

## Code Changes

Minimal loader support was added so the main codebase can use stable SDXL VAE
aliases directly:

- `src/utils/inference.py`
  - added `sdxl-fp32`
  - added `sdxl-fp16-fix` / `sdxl-fix`
- `tools/encode_image_folder_latents.py`
  - updated CLI help text
- `src/utils/run_evaluation.py`
  - updated CLI help text

## Practical Recommendation

For the next real SDXL-latent experiment, do not use plain `sdxl` on the current
fp16 encode path.

Use one of:

- `sdxl-fp16-fix` for the practical default
- `sdxl-fp32` if a slower official-VAE encode path is specifically needed for audit

The next useful step is no longer "can it run", but:

- scale the smoke from `10` samples to a real `Distinct5-512` packet
- keep `model.latent_scale_factor = 0.13025`
- set eval/decode `vae_model` to the same SDXL alias used for encoding
