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
- newest live read:
  - the active slot is still `Rococo`
  - live log has now entered `epoch 5`
  - but `Rococo epoch_5.model` is still not written at this read
  - so the first common every-5-epoch eval bundle is still blocked on the current style finishing this epoch
- newest stage transition:
  - `Rococo` has now landed `epoch_5.model`
  - the active sequential slot has advanced to `Ukiyo_e`
  - current alignment frontier is now:
    - `4 / 5` styles have landed `epoch_5.model`
  - remaining blocker before the first common every-5-epoch eval bundle:
    - `Ukiyo_e epoch_5.model`
- latest continuation read:
  - the active slot remains `Ukiyo_e`
  - live log is now in `epoch 1`
  - so the final blocker before the first common every-5-epoch eval bundle is no longer checkpoint writing on `Rococo`, but simply advancing `Ukiyo_e` to its own `epoch_5.model`
- newest continuation read:
  - the active slot remains `Ukiyo_e`
  - live log has now advanced into `epoch 2`
  - the first common every-5-epoch eval bundle is still blocked only by `Ukiyo_e epoch_5.model`
- latest continuation read:
  - the active slot remains `Ukiyo_e`
  - it is now deeper into `epoch 2`
  - no common `epoch_0005.model` frontier exists yet across all 5 styles
  - so the every-5-epoch eval watcher is still correctly waiting, not stalled
- first common epoch-5 eval repair:
  - the watcher did trigger the first common epoch-`5` eval bundle immediately after `Ukiyo_e epoch_5.model` landed
  - but the first several attempts failed for two concrete engineering reasons:
    - Windows console encoding inside the upstream `SaMST` test script
    - shared `content/` and `outputs/` directories being reused across retries
  - repair actions:
    - force UTF-8 for the `SaMST` test subprocess
    - route each target-style eval pass through its own temporary `content/` and `outputs/` subtree
  - repaired result:
    - `eval_bundle/eval_epoch5/epoch_0005/summary.json` now exists
    - `metrics.csv` now exists
    - generated image count:
      - `750`
  - current interpretation:
    - the first common every-5-epoch eval gate is now proven healthy
    - later common epochs can reuse the same repaired bundle path
- current closure read as of `2026-06-11`:
  - this run is **not converged**
  - the current closed packet is only the first common frontier:
    - `epoch_0005`
  - why it stops at `5` right now:
    - `SaMST` trains the five target styles sequentially
    - the watcher only evaluates epochs that exist in **all five** style checkpoint folders
    - at this stage, the only common saved epoch is still `5`
    - therefore `epoch_0005` is a valid first synchronized eval point, not a convergence claim
- wall-clock to the first common frontier:
  - run root created at:
    - `2026-06-10 17:22:06`
  - final aligned `Ukiyo_e/epoch_5.model` saved at about:
    - `2026-06-11 01:45:53`
  - train wall time to common `epoch_5`:
    - about `8h 23m 47s`
  - first successful full generation bundle after the repair:
    - `212.67s`
  - evaluator wall time on the retained `750` outputs:
    - `21.33s`
- `epoch_0005` metric summary:
  - transfer:
    - `CLIP-S = 0.68893`
    - `LPIPS = 0.62059`
    - `delta over IDT = +0.04901`
  - all-pairs:
    - `CLIP-S = 0.71815`
    - `LPIPS = 0.61283`
  - reading:
    - style is already strong enough to beat the wikiarts5 `IDT` floor on transfer
    - content preservation is still much weaker than `SaMAM` / current mainline variants on this split
- current continuation after the first common packet:
  - the automatic segmented controller is now active on the same result root
  - it is pushing the common frontier from `epoch_0005` toward `epoch_0010`
  - current active stage:
    - `Early_Renaissance`
    - live log already advanced into `epoch 7`
  - read:
    - the first common `epoch_0005` packet is closed
    - the baseline is now in the intended `every 5 common epochs -> eval -> continue` loop rather than a one-off midpoint run

<!-- WIKIARTS5_SAMST_AUTO_STATUS:START -->
## Auto Status

- Result root: [samst_wikiarts5_wsl_20260610_172206](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206)
- Live JSON: [samst_live_status.json](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206/samst_live_status.json)
- Active WSL process count: `1`
- Active WSL process:
  - `pid=42240` `etime=01:50:53`
- Eval watcher alive: `yes`
  - `pid=299424`
- Status watcher alive: `yes`
  - `pid=187052`
- Latest train log: [train_Impressionism.log](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206/logs/train_Impressionism.log)
- Active style: `Impressionism`
- Latest logged progress:
  - `epoch=7`
  - `step=3400 / 18894`
  - `content/style/ae/total = 223700.99 / 64376.94 / 208.60 / 288077.93`
- Common saved epochs across all 5 styles:
  - `5`
- Eligible every-5-epoch eval points currently present:
  - `5`
- Per-style saved epoch checkpoints:
  - `Early_Renaissance: 2`
  - `Impressionism: 1`
  - `Minimalism: 1`
  - `Rococo: 1`
  - `Ukiyo_e: 1`
- First eval trigger condition:
  - all five styles must each have `epoch_0005.model` before the every-5-epoch eval watcher launches the first full bundle
- Important interpretation:
  - the displayed `epoch` comes from the currently active single-style train log only
  - this run trains styles sequentially, not all 5 styles in lockstep
  - so `epoch=5` for `Early_Renaissance` still does not imply a common `epoch_0005.model` exists across all 5 style folders
  - even for the current style, `epoch=5` means `the 5th epoch is in progress`; the `epoch_5.model` file is only written after that epoch finishes
- Last eval-watch event:
  - `{"event": "poll", "common_epochs": [5], "per_style_epoch_counts": {"Early_Renaissance": 2, "Impressionism": 1, "Minimalism": 1, "Rococo": 1, "Ukiyo_e": 1}}`
- Local GPU sample:
  - `NVIDIA GeForce RTX 4070 Laptop GPU`
  - `4284 MiB / 8188 MiB`, `util=51%`
<!-- WIKIARTS5_SAMST_AUTO_STATUS:END -->





















































































































































































































































































































































































































