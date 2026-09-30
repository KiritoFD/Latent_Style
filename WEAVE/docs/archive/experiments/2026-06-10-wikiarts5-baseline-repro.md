# WikiArts-5 Baseline Repro

Date: 2026-06-10

Scope:

- local WSL baseline reproduction on the new full five-style RGB dataset
- keep local GPU usage under `6.5 GiB`
- start with `SaMAM patch_size=8`
- require the accelerated `mamba-ssm + causal-conv1d` route, not the pure-torch fallback

Dataset:

- RGB train root:
  - `F:\wikiarts_5_full_notest\train`
- flat view for `SaMAM`:
  - `F:\wikiarts_5_full_notest\train_flat\content`
  - `F:\wikiarts_5_full_notest\train_flat\style`
- eval root for periodic checkpoint tracking:
  - `F:\wikiart_distinct5_samam_512_classview\test`

WSL env:

- distro:
  - `f`
- confirmed usable python:
  - `/root/venvs/samam/bin/python`
- confirmed modules:
  - `torch 2.3.0+cu121`
  - `pytorch_lightning`
  - `mamba_ssm`
  - `causal_conv1d`

Entry points:

- env probe:
  - [wsl_find_python_env.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/wsl_find_python_env.sh)
- `SaMAM` direct WSL shell:
  - [run_samam_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samam_wikiarts5_wsl.sh)
- `SaMAM` segmented train/eval shell:
  - [run_samam_wikiarts5_segmented_eval_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samam_wikiarts5_segmented_eval_wsl.sh)
  - current default behavior:
    - keep extending past the old `MAX_STEPS` budget until the convergence rule is actually satisfied
    - old fixed-cap behavior is still available through `STOP_AT_MAX_STEPS=1`
- `SaMST` direct WSL shell:
  - [run_samst_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samst_wikiarts5_wsl.sh)
- `SaMAM` segmented curve aggregator:
  - [aggregate_samam_segmented_curve.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/aggregate_samam_segmented_curve.py)
- baseline live-status/doc refresher:
  - [update_wikiarts5_baseline_repro_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/update_wikiarts5_baseline_repro_status.py)

Current run:

- `SaMAM patch_size=8 segmented`
  - result root:
    - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wikiarts5_patch8_segmented_20260610_094447`
  - control log:
    - `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samam_wikiarts5_segmented.stderr.log`
  - current segment:
    - current train target is `8000`
    - latest settled point is `7750`
  - periodic eval contract:
    - every `250` steps
    - run `clip-s + lpips`
    - record inference wall time
    - write convergence curve
    - current convergence authority:
      - `transfer_clip_style + transfer_lpips`
      - not the raw all-pairs mean alone
  - root curve artifacts:
    - [curve_metrics.csv](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_metrics.csv)
    - [curve_metrics.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_metrics.json)
    - [baseline_live_status.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/baseline_live_status.json)
    - [clip_lpips_curve.png](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/clip_lpips_curve.png)
    - [timing_curve.png](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/timing_curve.png)
    - [curve_convergence.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_convergence.json)
  - convergence watcher:
    - [watch_samam_segmented_convergence.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/watch_samam_segmented_convergence.py)
    - initial stdout:
      - [samam_wikiarts5_segmented_convergence_watch_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_convergence_watch_20260610.stdout.log)
    - current auto-refresh watcher stdout:
      - [samam_wikiarts5_segmented_convergence_watch_refresh_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_convergence_watch_refresh_20260610.stdout.log)
    - current auto-refresh watcher stderr:
      - [samam_wikiarts5_segmented_convergence_watch_refresh_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_convergence_watch_refresh_20260610.stderr.log)
    - note:
      - the current watcher now also triggers [update_wikiarts5_baseline_repro_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/update_wikiarts5_baseline_repro_status.py) after each new settled curve state, so the auto block below is the authoritative live read
      - the convergence watcher is now also keyed on:
        - `transfer_clip_style`
        - `transfer_lpips`
      - this better matches the actual paper-facing transfer board and reduces the chance of stopping on an all-pairs artifact plateau
      - the same updater now also refreshes the `wikiarts5` page-1 summary packet automatically:
        - [2026-06-10-wikiarts5-page1-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-page1-read.md)
        - [fig_wikiarts5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_page1_summary.png)
      - a detached resume watcher is now also armed for this exact result root:
        - [watch_resume_wikiarts5_segmented_until_converged.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_resume_wikiarts5_segmented_until_converged.py)
        - stdout:
          - [samam_wikiarts5_segmented_resume_watch_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_resume_watch_20260610.stdout.log)
        - stderr:
          - [samam_wikiarts5_segmented_resume_watch_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_resume_watch_20260610.stderr.log)
        - purpose:
          - if the old controller exits because of the historical `MAX_STEPS=20000` cap while `curve_convergence.json` is still `false`, the watcher relaunches the segmented script on the same result root with the new `run-until-converged` behavior

First settled point:

- `step = 250`
  - `clip_style = 0.5832`
  - `content_lpips = 0.5929`
  - `clip_content = 0.6104`
  - `infer_wall_seconds = 74.55`
  - `metric_wall_seconds = 44.65`

Current settled frontier:

- best `CLIP-S` so far:
  - `step = 5750`
  - `clip_style = 0.6331`
  - `content_lpips = 0.3414`
  - `clip_content = 0.7915`
  - `infer_wall_seconds = 69.34`
  - `metric_wall_seconds = 42.72`
- best `LPIPS` so far:
  - `step = 7750`
  - `clip_style = 0.6276`
  - `content_lpips = 0.2972`
  - `clip_content = 0.8098`
  - `infer_wall_seconds = 70.72`
  - `metric_wall_seconds = 43.06`
- current latest settled point:
  - `step = 7750`
  - next segment is already training toward `8000`
- current convergence read:
  - `row_count = 31`
  - `best_step = 5750`
  - `last_pareto_step = 7750`
  - `since_last_pareto = 0`
  - `converged = false`

First memory read:

- local GPU sample after launch:
  - `2629 MiB / 8188 MiB`
  - interpretation:
    - comfortably below the requested `6.5 GiB` ceiling at early training time
    - continue to monitor through the first `250-step` checkpoint

Current device read:

- local WSL GPU sample during the `7750 -> 8000` segment:
  - `3104 MiB / 8188 MiB`
- interpretation:
  - still comfortably below the requested `6.5 GiB` ceiling
  - no second local Python training/eval process remains on this GPU

Implementation note:

- `SaMam` RGB training expects a flat file layout, not classview folders
- the new dataset therefore also has:
  - `F:\wikiarts_5_full_notest\train_flat\content`
  - `F:\wikiarts_5_full_notest\train_flat\style`

Remote mainline prep:

- current remote task:
  - `wikiarts5-latent-prep`
- target remote latent root:
  - `/mnt/i/wikiarts_5_full_notest_latents_ema/train`
- current state:
  - remote latent encoding completed on `2026-06-10`
  - remote packed manifest now exists:
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/manifest.json`
  - remote pairing cache now exists:
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

<!-- WIKIARTS5_BASELINE_AUTO_STATUS:START -->
## Auto Status

- Result root: [samam_wikiarts5_patch8_segmented_20260610_094447](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447)
- Curve CSV: [curve_metrics.csv](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_metrics.csv)
- Convergence JSON: [curve_convergence.json](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_convergence.json)
- CLIP/LPIPS curve: [clip_lpips_curve.png](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/clip_lpips_curve.png)
- Timing curve: [timing_curve.png](G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/timing_curve.png)
- Active WSL training process count: `0`
- Current training segment: `23000`
- Latest settled point:
  - `step=23000`
  - `clip_style / lpips / clip_content = 0.6188 / 0.2159 / 0.8695`
  - `infer_wall_seconds / metric_wall_seconds = 100.39 / 49.89`
- Best `CLIP-S`:
  - `step=5750`
  - `clip_style / lpips = 0.6331 / 0.3414`
- Best `LPIPS`:
  - `step=19500`
  - `clip_style / lpips = 0.6226 / 0.2124`
- Best transfer `CLIP-S`:
  - `step=5750`
  - `clip_style / lpips = 0.6173 / 0.3504`
- Best transfer `LPIPS`:
  - `step=19500`
  - `clip_style / lpips = 0.5999 / 0.2209`
- Latest transfer point:
  - `step=23000`
  - `clip_style / lpips = 0.5953 / 0.2246`
- Convergence snapshot:
  - `row_count = 92`
  - `best_step = 5750`
  - `last_pareto_step = 19500`
  - `since_last_pareto = 14`
  - `tail_flat = True`
  - `style_key / lpips_key = transfer_clip_style / transfer_lpips`
  - `converged = True`
- Local GPU sample:
  - `NVIDIA GeForce RTX 4070 Laptop GPU`
  - `3880 MiB / 8188 MiB`, `util=72%`
<!-- WIKIARTS5_BASELINE_AUTO_STATUS:END -->






































































