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
- `SaMST` direct WSL shell:
  - [run_samst_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samst_wikiarts5_wsl.sh)
- `SaMAM` segmented curve aggregator:
  - [aggregate_samam_segmented_curve.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/aggregate_samam_segmented_curve.py)

Current run:

- `SaMAM patch_size=8 segmented`
  - result root:
    - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wikiarts5_patch8_segmented_20260610_094447`
  - control log:
    - `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samam_wikiarts5_segmented.stderr.log`
  - current segment:
    - current train target is `5750`
    - latest settled point is `5500`
  - periodic eval contract:
    - every `250` steps
    - run `clip-s + lpips`
    - record inference wall time
    - write convergence curve
  - root curve artifacts:
    - [curve_metrics.csv](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_metrics.csv)
    - [curve_metrics.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_metrics.json)
    - [clip_lpips_curve.png](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/clip_lpips_curve.png)
    - [timing_curve.png](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/timing_curve.png)
    - [curve_convergence.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/curve_convergence.json)
  - convergence watcher:
    - [watch_samam_segmented_convergence.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/watch_samam_segmented_convergence.py)
    - stdout:
      - [samam_wikiarts5_segmented_convergence_watch_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/samam_wikiarts5_segmented_convergence_watch_20260610.stdout.log)

First settled point:

- `step = 250`
  - `clip_style = 0.5832`
  - `content_lpips = 0.5929`
  - `clip_content = 0.6104`
  - `infer_wall_seconds = 74.55`
  - `metric_wall_seconds = 44.65`

Current settled frontier:

- best `CLIP-S` so far:
  - `step = 5500`
  - `clip_style = 0.6328`
  - `content_lpips = 0.3268`
  - `clip_content = 0.7982`
  - `infer_wall_seconds = 69.86`
  - `metric_wall_seconds = 42.14`
- best `LPIPS` so far:
  - `step = 5000`
  - `clip_style = 0.6273`
  - `content_lpips = 0.3080`
  - `clip_content = 0.7973`
  - `infer_wall_seconds = 71.41`
  - `metric_wall_seconds = 43.98`
- current latest settled point:
  - `step = 5500`
  - next segment is already training toward `5750`
- current convergence read:
  - `best_step = 5500`
  - `last_pareto_step = 5500`
  - `since_last_pareto = 0`
  - `converged = false`

First memory read:

- local GPU sample after launch:
  - `2629 MiB / 8188 MiB`
  - interpretation:
    - comfortably below the requested `6.5 GiB` ceiling at early training time
    - continue to monitor through the first `250-step` checkpoint

Current device read:

- local WSL GPU sample during the `5500 -> 5750` segment:
  - `2896 MiB / 8188 MiB`
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
