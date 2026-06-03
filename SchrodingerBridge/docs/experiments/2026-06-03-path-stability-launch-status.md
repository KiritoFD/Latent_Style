# Distinct5 Path-Stability Launch Status

Date: 2026-06-03

Status:

- `packet_landed__probe_promotion_condition_met`

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

Re-checked later on `2026-06-03` during local packet audit:

- no matching remote `python` process remained alive
- GPU had returned to idle:
  - about `114 MiB`
  - about `11.6 W`
  - about `0%` GPU util
- save dir still contained only:
  - `epoch_0001.pt`
  - `logs\training_20260603_144043.csv`
  - `logs\training_20260603_144732.csv`
  - `numeric_debug.jsonl`
- `numeric_debug.jsonl` still showed finite progress through at least:
  - `epoch 2 step 40`
- but there was still no:
  - `epoch_0002.pt`
  - `full_eval\...\summary.json`
  - `remote_train.log`

Current operational read:

- the packet has been launched and did real work on the remote 3060;
- the `base` arm is **not** currently healthy-running;
- the currently retained runtime surface is partial and mixed across foreground
  recovery attempts;
- the packet is therefore not yet admissible as a landed mechanism result.

## Clean rerun recovery

Observed later on `2026-06-03` during the recovery pass:

- the interrupted mixed-artifact save dir was archived to:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_h_base_seed42_b44_interrupted_20260603_1449`
- a clean launcher was written to:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\_launchers\aaai2027_path_kinetic_h_base_seed42_b44_base_clean.cmd`
- the `base` arm was relaunched from that clean surface
- a fresh save dir was recreated at:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_h_base_seed42_b44`
- retained fresh runtime evidence now includes:
  - `remote_train.log`
- verified healthy shortly after relaunch:
  - GPU about `9648 MiB`
  - GPU about `100%` util
  - GPU about `154 W`
- `remote_train.log` shows:
  - dataset load success
  - model init success
  - epoch `1/3` in active progress

Current operational read after recovery:

- the clean rerun is now the authoritative live `base` arm;
- the archived interrupted directory remains as provenance only;
- the next gate is still successful completion with retained checkpoints and
  `full_eval` summaries.

## Clean `base` rerun completion

Re-checked on `2026-06-03` after the recovered rerun finished:

- GPU had returned to idle:
  - about `114 MiB`
  - about `0%` util
  - about `11.7 W`
- the clean `base` save dir retained the full expected chain:
  - `remote_train.log`
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `full_eval\epoch_0001\summary.json`
  - `full_eval\epoch_0002\summary.json`
  - `full_eval\epoch_0003\summary.json`
- `remote_train.log` ends cleanly with:
  - full eval completed for `epoch_0003`
  - `Training completed.`
  - launcher exit code `0`

Retained full-scope metrics:

- `epoch_0001`
  - `clip_style = 0.6891`
  - `content_lpips = 0.4272`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 87.72s`
- `epoch_0002`
  - `clip_style = 0.6821`
  - `content_lpips = 0.4198`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 87.18s`
- `epoch_0003`
  - `clip_style = 0.6887`
  - `content_lpips = 0.4171`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 87.37s`

Current read after `base` landing:

- the clean `base` arm is now reviewer-safe as a retained artifact chain;
- this closes the earlier provenance hole for the `base` arm only;
- the mechanism packet is still incomplete until `k025`, `k000`, and the
  retained probe are landed.

## `k025` launch

Observed on `2026-06-03` after the clean `base` rerun landed:

- a dedicated launcher was created at:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\_launchers\aaai2027_path_kinetic_h_base_seed42_b44_k025.cmd`
- the matched weakened-kinetic arm was launched by direct foreground SSH
  invocation
- current save dir:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_h_base_seed42_b44_k025`
- retained runtime evidence already present:
  - `remote_train.log`
- early log and device checks confirm healthy execution:
  - log timestamp `2026-06-03 15:41:00`
  - dataset load success
  - model init success
  - training entered `epoch 1/3`
  - GPU about `9648 MiB`
  - GPU about `96%` util
  - GPU about `151 W`

## `k025` completion

Re-checked on `2026-06-03` after the weakened-kinetic run finished:

- GPU returned to idle before the next launch
- retained artifact chain is complete:
  - `remote_train.log`
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `full_eval\epoch_0001\summary.json`
  - `full_eval\epoch_0002\summary.json`
  - `full_eval\epoch_0003\summary.json`
- `remote_train.log` ends with:
  - full eval completed for `epoch_0003`
  - `Training completed.`

Retained full-scope metrics:

- `epoch_0001`
  - `clip_style = 0.6792`
  - `content_lpips = 0.4977`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 88.59s`
- `epoch_0002`
  - `clip_style = 0.6825`
  - `content_lpips = 0.4600`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 88.38s`
- `epoch_0003`
  - `clip_style = 0.6817`
  - `content_lpips = 0.4668`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 88.41s`

Current read after `k025` landing:

- weakening the kinetic weight to `0.25` does not obviously improve the current
  quality frontier over the clean `base` arm;
- the packet is still incomplete, so no mechanism claim should move yet;
- the matched `k000` arm remains required for the same-family readout.

## `k000` launch

Observed on `2026-06-03` immediately after `k025` completion:

- a dedicated launcher was created at:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\_launchers\aaai2027_path_kinetic_h_base_seed42_b44_k000.cmd`
- the no-kinetic arm was launched by direct foreground SSH invocation
- current save dir:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_h_base_seed42_b44_k000`
- retained runtime evidence already present:
  - `remote_train.log`
- early log and device checks confirm healthy execution:
  - log timestamp `2026-06-03 15:52:30`
  - dataset load success
  - model init success
  - GPU about `9648 MiB`
  - GPU about `96%` util
  - GPU about `150 W`

## `k000` completion

Re-checked on `2026-06-03` after the no-kinetic run finished:

- retained artifact chain is complete:
  - `remote_train.log`
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `full_eval\epoch_0001\summary.json`
  - `full_eval\epoch_0002\summary.json`
  - `full_eval\epoch_0003\summary.json`
- `remote_train.log` ends with:
  - full eval completed for `epoch_0003`
  - `Training completed.`

Retained full-scope metrics:

- `epoch_0001`
  - `clip_style = 0.6761`
  - `content_lpips = 0.5198`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 89.30s`
- `epoch_0002`
  - `clip_style = 0.6775`
  - `content_lpips = 0.4862`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 88.25s`
- `epoch_0003`
  - `clip_style = 0.6790`
  - `content_lpips = 0.5073`
  - `clip_dir = 0.0000`
  - `full_eval wall_total = 88.44s`

Current read after `k000` landing:

- removing the kinetic term is materially worse than the clean `base` arm on
  both style/content quality and the no-op-adjusted mechanism packet;
- the matched three-arm checkpoint surface is now complete and ready for the
  retained probe readout.

## Probe completion

Probe run executed on `2026-06-03` with a reviewer-safe matched setup:

- tool:
  - `tools\probe_path_stability.py`
- rollout mode:
  - `field`
- checkpoint selection:
  - `H_base = epoch_0001`
  - `H_k025 = epoch_0001`
  - `H_k000 = epoch_0001`
- output dir:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_stability_probe_h_base_seed42_b44_e1`
- retained outputs:
  - `summary.json`
  - `per_time_stats.csv`
  - `run_summary.csv`
  - `fig_velocity_over_time.pdf`

Key matched transfer-direction readout:

- `H_base`
  - `mean_endpoint_disp_l2 = 80.24`
  - `mean_path_length_l2 = 80.28`
  - `mean_peak_velocity_l2 = 80.41`
- `H_k025`
  - `mean_endpoint_disp_l2 = 111.71`
  - `mean_path_length_l2 = 111.72`
  - `mean_peak_velocity_l2 = 111.83`
- `H_k000`
  - `mean_endpoint_disp_l2 = 122.42`
  - `mean_path_length_l2 = 122.40`
  - `mean_peak_velocity_l2 = 122.51`

Promotion-rule read:

- the packet satisfies accept rule `1`
  - weakening or removing kinetic clearly raises transfer-direction velocity
    magnitude and endpoint/path displacement under matched `epoch_0001`
    checkpoints;
- the clean `base` arm therefore has concrete same-family Distinct5 support for
  the bounded claim that kinetic regularization acts as a practical path
  stabilizer in the current OMF/field regime;
- this does **not** justify a broader theorem claim beyond the current bounded
  wording.

## Still pending

- absorb the landed packet into manuscript wording and figure priorities
- open the next four-lane review cycle only after the current write/update pass
  is stable

## Policy read

This packet is still the highest-value unblocked mechanism closure after the
latest four-lane review reread (`R20260603L`). The paper should not expand its
kinetic/path-energy claim until this packet either lands cleanly or fails under
its own accept/reject rule.
