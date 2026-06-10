# WikiArts-5 SaMST Repro

Date: 2026-06-10

Scope:

- start a local WSL `SaMST` reproduction on the new `wikiarts5` train pool
- use the existing WSL env:
  - `/root/venvs/samam/bin/python`
- keep the run isolated with explicit stdout / stderr logs

Launch:

- launcher:
  - [run_wsl_samst_wikiarts5.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_wsl_samst_wikiarts5.py)
- underlying WSL shell:
  - [run_samst_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samst_wikiarts5_wsl.sh)
- dataset root:
  - `F:\wikiarts_5_full_notest`
- styles:
  - `Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e`
- batch size:
  - `1`
- image size:
  - `256`
- style size:
  - `512`
- epochs:
  - `100`

Current run:

- result root:
  - [samst_wikiarts5_wsl_20260610_172206](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206)
- stdout:
  - [samst_wikiarts5_wsl_20260610_172206.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samst_wikiarts5_wsl_20260610_172206.stdout.log)
- stderr:
  - [samst_wikiarts5_wsl_20260610_172206.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samst_wikiarts5_wsl_20260610_172206.stderr.log)
- launch meta:
  - [samst_wikiarts5_wsl_20260610_172206.stdout.meta.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samst_wikiarts5_wsl_20260610_172206.stdout.meta.json)

Process check right after launch:

- WSL process:
  - `pid=27526`
  - command:
    - `/root/venvs/samam/bin/python ... run_samst_distinct5_local.py --data-root /mnt/f/wikiarts_5_full_notest ...`
- local GPU sample:
  - `4915 MiB / 8188 MiB`
  - `util=64%`

Notes:

- the first launch attempt failed because PowerShell split the comma-separated `--styles` argument
- the second launch fixed that quoting issue but still used the old `save_interval=100` default
- the current authoritative run is the third launch:
  - same dataset / env / batch
  - but now with `save_interval=5`, so the every-5-epoch eval watcher can actually trigger
- every-5-epoch eval watcher:
  - [watch_wikiarts5_samst_eval_bundle.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_wikiarts5_samst_eval_bundle.py)
  - stdout:
    - [samst_wikiarts5_eval_every5_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samst_wikiarts5_eval_every5_20260610.stdout.log)
  - stderr:
    - [samst_wikiarts5_eval_every5_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samst_wikiarts5_eval_every5_20260610.stderr.log)
  - trigger rule:
    - once all five target-style checkpoint folders contain the same common `epoch_{5k}.model`, launch one full `CLIP-S / LPIPS` eval bundle for that epoch
- important read:
  - this reproduction trains target styles sequentially
  - the visible `epoch` in the live log is only the epoch of the currently active style
  - therefore `epoch=5` does not mean the first every-5-epoch eval should already exist
  - and even for the current style, `epoch=5` does not mean `epoch_5.model` already exists; that checkpoint is written only after the 5th epoch finishes
  - the first eval only fires after all five style checkpoint folders each contain `epoch_0005.model`
  - current alignment frontier:
    - `2 / 5` styles have already landed `epoch_5.model`
    - the active sequential slot has now moved on to `Minimalism`
- explicit alignment control added:
  - `--skip-styles-with-epoch-at-least 5`
  - `--stop-after-one-pending-style`
  - together these let the WSL runner advance exactly one still-unfinished style to the common `epoch 5` frontier per launch, instead of relying on an implicit multi-style sequential command
- current alignment read:
  - `Early_Renaissance` and `Impressionism` have already landed `epoch_5.model`
  - the active slot is still `Minimalism`
  - `Minimalism` is now in its `epoch 3` phase, but has not yet written `epoch_5.model`
- latest live read:
  - `Minimalism` has now entered `epoch 5`
  - but `epoch_5.model` still has not been written at this read
- newest transition:
  - `Minimalism` has now landed `epoch_5.model`
  - the active sequential slot has advanced again to `Rococo`
- current alignment frontier is now:
  - `3 / 5` styles have landed `epoch_5.model`
  - remaining:
    - `Rococo`
    - `Ukiyo_e`
- latest live read:
  - `Rococo` has now entered `epoch 2`
  - but `Rococo epoch_5.model` has not landed yet
- current live read:
  - `Rococo` is still the active slot
  - it remains below `epoch_5.model` at this read
  - current progress:
    - `Rococo` is now deep into `epoch 2`
    - but the fourth aligned `epoch_5.model` still has not landed
- current sequential frontier:
  - the active slot remains `Rococo`
  - `Ukiyo_e` has not started yet
- latest continuation read:
  - the active slot is still `Rococo`
  - live log has advanced into `epoch 3`
  - but `Rococo epoch_5.model` still has not landed
  - therefore the global every-5-epoch eval trigger is still correctly blocked at `3 / 5` aligned styles, not stalled
- newest continuation read:
  - the active slot is still `Rococo`
  - live log has now advanced into `epoch 4`
  - but the fourth aligned `epoch_5.model` still has not landed
  - so the every-5-epoch eval gate is still correctly waiting on `Rococo` first, then `Ukiyo_e`

<!-- WIKIARTS5_SAMST_AUTO_STATUS:START -->
## Auto Status

- Result root: [samst_wikiarts5_wsl_20260610_172206](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206)
- Live JSON: [samst_live_status.json](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206/samst_live_status.json)
- Active WSL process count: `1`
- Active WSL process:
  - `pid=31509` `etime=04:06:12`
- Eval watcher alive: `yes`
  - `pid=174368`
- Status watcher alive: `yes`
  - `pid=187052`
- Latest train log: [train_Rococo.log](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206/logs/train_Rococo.log)
- Active style: `Rococo`
- Latest logged progress:
  - `epoch=5`
  - `step=15200 / 18894`
  - `content/style/ae/total = 153575.04 / 81542.84 / 279.68 / 235117.87`
- Common saved epochs across all 5 styles:
  - `none yet`
- Eligible every-5-epoch eval points currently present:
  - `none yet`
- Per-style saved epoch checkpoints:
  - `Early_Renaissance: 1`
  - `Impressionism: 1`
  - `Minimalism: 1`
  - `Rococo: 0`
  - `Ukiyo_e: 0`
- First eval trigger condition:
  - all five styles must each have `epoch_0005.model` before the every-5-epoch eval watcher launches the first full bundle
- Important interpretation:
  - the displayed `epoch` comes from the currently active single-style train log only
  - this run trains styles sequentially, not all 5 styles in lockstep
  - so `epoch=5` for `Early_Renaissance` still does not imply a common `epoch_0005.model` exists across all 5 style folders
  - even for the current style, `epoch=5` means `the 5th epoch is in progress`; the `epoch_5.model` file is only written after that epoch finishes
- Last eval-watch event:
  - `{"event": "poll", "common_epochs": [], "per_style_epoch_counts": {"Early_Renaissance": 1, "Impressionism": 1, "Minimalism": 1, "Rococo": 0, "Ukiyo_e": 0}}`
- Local GPU sample:
  - `NVIDIA GeForce RTX 4070 Laptop GPU`
  - `4326 MiB / 8188 MiB`, `util=67%`
<!-- WIKIARTS5_SAMST_AUTO_STATUS:END -->
















































































































































































































































