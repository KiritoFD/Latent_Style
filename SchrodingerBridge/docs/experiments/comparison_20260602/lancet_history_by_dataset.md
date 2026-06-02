# LANCET History By Dataset

This note groups the currently relevant `LANCET` history by dataset and records
which points are used in the comparison figures.

## 1. legacy256_overfit50

Representative strict-summary points used in scatter plots:

- `S-add e8`
  - summary: [epoch_0008/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json)
  - role: stronger style among stable `0.45-0.47` LPIPS points
- `K1 original e7`
  - summary: [epoch_0007/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0007/summary.json)
  - role: stable content frontier around `LPIPS ~= 0.451`
- `steps_12`
  - summary: [steps_12/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/exp/review_additional_experiments/review_additional_experiments/step_count_sweep/steps_12/summary.json)
  - role: step-count sweep point nearly identical to `K1 original e7`

Artifact state:

- The modern strict-summary dirs above keep `metrics.csv` and `summary.json`
  but do not retain `images/`.
- Because target-wise `ArtFID` needs the generated images, the ArtFID bar chart
  uses archive proxies:
  - [archive epoch_0007](/G:/GitHub/Latent_Style/SchrodingerBridge/archives/old_experiment_dirs/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0/full_eval/epoch_0007/summary.json)
  - [archive epoch_0008](/G:/GitHub/Latent_Style/SchrodingerBridge/archives/old_experiment_dirs/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json)
- These archive points are close in family and training regime, but they are
  not the exact same modern strict-summary records. They are marked
  `targetwise_archive_proxy`.

## 2. wikiart512_5style

Representative points:

- `local WSL hist b32 e8`
  - summary: [summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8/full_eval_epoch_0008_b2_opt_nocls/summary.json)
  - target-wise ArtFID: [aggregate_targetwise_artfid.json](/G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8/full_eval_epoch_0008_b2_opt_nocls/aggregate_targetwise_artfid.json)
  - role: best verified historical style point
- `from-scratch e8`
  - summary: [summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/exp/timing_20260602/lancet_from_scratch_e8_generate750/summary.json)
  - role: later from-scratch point with lower style and weaker content than
    `hist b32 e8`

Artifact state:

- `local WSL hist b32 e8` is fully evidenced with summary, metrics, generated
  images, and target-wise ArtFID.
- `from-scratch e8` is retained for scatter context but not used in ArtFID bars
  because the best historical point is the more important comparison anchor.

## 3. distinct5_512

Representative points:

- `F e1`
  - role: current LPIPS-best point
- `K e1`
  - role: current style-best point
- `H e1`
  - role: balanced LPIPS point
- `H e2`
  - role: balanced style point

Scatter source:

- [clip_style_vs_1lpips_full_transfer_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv)

ArtFID source:

- [distinct5_aggregate_artfid_keypoints.remote.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.remote.csv)
- [distinct5_aggregate_artfid_keypoints.remote.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.remote.json)

Artifact state:

- These were aggregated remotely and only the result csv/json were pulled back.
- For the current bars we use `F e1` and `K e1` as the two main `LANCET`
  representatives.

## 4. SaMST evaluation status

The previously stopped local `SaMST` line is now evaluated for the datasets
used in this comparison:

- `legacy256_overfit50`
  - [protocol_a_800/summary.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst/protocol_a_800/summary.json)
- `wikiart512_5style`
  - [epoch_0015/summary.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/timing_20260602/samst_wikiart512_epoch15_generate750_png/epoch_0015/summary.json)
- `distinct5_512`
  - [epoch_0015/summary.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b2_e15_20260602/eval_epoch15/epoch_0015/summary.json)
