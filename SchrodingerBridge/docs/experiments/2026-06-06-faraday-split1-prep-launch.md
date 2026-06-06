# Faraday Split1 Prep Launch

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress1`
- objective:
  - sync the selected classview packet to the remote owner surface
  - build `512` EMA latents for training
  - build packed latent cache
  - build prototype-aware pairing cache

## Why this lane exists

The follow-up stress-split line was already selected and materialized locally,
but it was not yet runnable on the remote `3060` because the owner surface
still lacked:

- the split packet under `/mnt/i/wikiart_faraday_splits/...`
- `latents_ema/train`
- `.latent_cache/manifest.json`
- `.latent_cache/prototype_pairing_top8.pt`

This prep lane closes exactly that gap for the first follow-up split.

## Local packet

Selected split slug:

- `wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism`

Local packet root:

- `F:\wikiart_faraday_splits\wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism`

Measured packet size before sync:

- about `1.225 GiB`
- `5151` files

Styles:

- `Color_Field_Painting`
- `High_Renaissance`
- `Mannerism_Late_Renaissance`
- `Pop_Art`
- `Realism`

## New helper surface

Remote prep runner:

- [run_faraday_split_prep.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_faraday_split_prep.py)

Local launcher:

- [launch_remote_faraday_split_prep.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_faraday_split_prep.py)

Design:

- local helper syncs the external split packet from `F:\wikiart_faraday_splits`
  to `/mnt/i/wikiart_faraday_splits`
- then it reuses the reviewed host-owned remote launcher:
  - [launch_remote_wsl_command.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py)
- remote helper runs:
  1. `encode_image_folder_latents.py`
  2. `build_latent_packed_cache.py`
  3. `build_latent_prototype_pairing_cache.py`

## Launch command

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_faraday_split_prep.py `
  --split-slug wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism
```

## Remote launch contract

Task name:

- `faraday-prep-wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism`

Remote split root:

- `/mnt/i/wikiart_faraday_splits/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism`

Remote log:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_prep.log`

Remote process at first-health:

- launcher pid file resolved to `446`
- active Python pid observed:
  - `452`

## First-health snapshot

Observed immediately after launch:

- remote GPU prelaunch:
  - `282 MiB`
- first-health GPU:
  - about `2178 MiB / 12288 MiB`
- runtime band:
  - comfortably below the hard `< 11.0 GiB` policy

First active step in the log:

- `encode_image_folder_latents.py`
- class:
  - `Color_Field_Painting`
- observed throughput after warmup:
  - about `7 images/s`
  - with occasional slower outliers caused by very large source images

Operational read:

- the lane is healthy
- this is a data-prep lane, not yet a paper-facing train/eval packet
- once the prep finishes, the first follow-up split becomes launchable under the
  normal reviewed `F`-family remote config

## Next action after prep closes

Only after the remote prep lands with:

- `latents_ema/train/<style>/*.pt`
- `latents_ema/train/.latent_cache/manifest.json`
- `latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

then the next formal GPU action should be:

- one remote `F`-family train/eval lane on this split
- not a second split prep in parallel
