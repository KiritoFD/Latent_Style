# Distinct5 Path-Stability Launch Status

Date: 2026-06-03

Status:

- `in_progress`

Scope:

- packet:
  - Distinct5 `H`-family path-stability / weakened-kinetic packet
- intended arms:
  - `base`
  - `k025`
  - `k000`
- target machine:
  - remote `RTX 3060`

## What was verified before launch

- remote GPU was idle before launch:
  - about `114 MiB`, `~11.5 W`, `0%` GPU util
- remote project root in use:
  - `I:\Github\Latent_Style\SchrodingerBridge`
- remote packet assets were missing initially:
  - `configs\aaai2027\path_kinetic_h_base_seed42_b44_{base,k025,k000}.json`
  - `tools\probe_path_stability.py`
- those assets were pushed from the current local branch to the remote root
- remote compile check passed for:
  - `src\config_schema.py`
  - `src\run.py`
  - `src\model.py`
  - `src\losses.py`
  - `src\ot_cost.py`
  - `tools\probe_path_stability.py`

## Launch issues found and resolved

### 1. Remote sync artifact collision

- a broken earlier transfer had created:
  - `I:\Github\Latent_Style\SchrodingerBridge\configs\aaai2027`
- but it was a **file**, not a directory
- this blocked later config writes
- the file was removed and replaced with a real directory

### 2. UTF-8 BOM in remote JSON configs

- first remote config writes used Windows PowerShell UTF-8 with BOM
- `json.load` in remote `src/run.py` failed with:
  - `Unexpected UTF-8 BOM`
- configs were rewritten as UTF-8 **without BOM**

### 3. Remote data-root mismatch

- initial packet config pointed to:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- on this Windows remote, that root exposed only a packed-manifest surface and
  did not present per-style latent directories required by the current
  dataset-loader path
- verified usable Windows latent root:
  - `I:\wikiart_distinct5_latents_512_ema`
- verified test-image root:
  - `I:\wikiart_distinct5_samam_512_classview\test`
- verified pairing cache source:
  - `I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\prototype_pairing_top8.pt`
- remote `base` config was rewritten to those paths
- `k025` and `k000` still inherit from the rewritten remote `base` config

## Current live state

Observed at `2026-06-03 14:44:58` local time:

- active remote Python process:
  - `python` pid `89428`
  - start time `2026/6/3 14:40:30`
- GPU:
  - about `9648 MiB`
  - about `153 W`
  - about `96%` GPU util

Current save dir:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_h_base_seed42_b44`

Observed current artifacts:

- `config.json`
- `numeric_debug.jsonl`
- `logs\training_20260603_144043.csv`

Note:

- the currently running `base` arm was started by a direct foreground SSH
  invocation after debugging, so it does **not yet** have the intended
  `remote_train.log`
- the durable primary runtime evidence for this live arm is currently:
  - save dir presence
  - `logs\training_20260603_144043.csv`
  - `numeric_debug.jsonl`
  - GPU/process state captured above

## Still pending

- wait for the running `base` arm to finish and collect:
  - checkpoints
  - `full_eval\epoch_0001..0003\summary.json`
- then relaunch or continue packetized execution for:
  - `k025`
  - `k000`
- then run:
  - `tools\probe_path_stability.py`

## Policy read

This packet is still the highest-value unblocked mechanism closure after the
latest four-lane review reread (`R20260603L`). The paper should not expand its
kinetic/path-energy claim until this packet either lands cleanly or fails under
its own accept/reject rule.
