# Local Generated Media Owner Review - 2026-06-05

This is a manual owner-level review pass for local generated-image/video-heavy directories. The media-count scan was used only to rank candidates. Deletion decisions below require direct directory opening plus evidence checks.

## Reviewed And Retained

- `Related_Works\runs\cut_5x5\datasets`: retained as a dataset mirror. It has `manifest_overfit50.csv` and `manifest_val.csv`; the 56,305 jpg files are not generated-output trash.
- `SchrodingerBridge\exp\paper\paper_main_750_bundle`: retained as paper-facing bundle. It has 6,750 method/style jpgs. No replacement manifest was found, so it is not a deletion target.
- `Related_Works\baseline_pipeline\results\samam_wsl_mamba_512_scratch_clean_silent_b1_20k\formal_eval_750`: retained as formal eval curve evidence. It has curve metrics, per-step `summary.json`, `metrics.csv`, and ArtFID aggregate files, and it is referenced by docs/timing and comparison points.
- `SchrodingerBridge\exp\diagnostics\seedream_gap`: retained pending owner decision. It is diagnostic media under a diagnostics parent with CLIP geometry/separability docs, but the directory itself has no summary.
- `Related_Works\runs\cut_5x5\raw_results` and `raw_results_val`: retained pending owner decision. They are CUT baseline raw outputs with training logs and later summaries elsewhere in the CUT bundle.
- `SchrodingerBridge\exp\inference\inference_param_sweep_t01e8_quick` and `inference_param_sweep_t01e8_fine`: retained pending owner decision. They are small parameter-sweep media directories with no in-directory summary.
- `Related_Works\baseline_pipeline\results\s2wat\protocol_a_800` and `cut\protocol_a_800`: retained as protocol eval evidence with `summary.json` and `metrics.csv`.

## Delete Whitelist

The following directories are whitelisted because each was opened and found to contain only `_work` intermediate frame pngs, with no mp4/json/csv inside the directory and no text references to the timestamp:

- `Related_Works\runs\cut_5x5\video\head_20260404_140330`
- `Related_Works\runs\cut_5x5\video\head_20260404_140349`
- `Related_Works\runs\cut_5x5\video\head_20260404_140443`
- `Related_Works\runs\cut_5x5\video\head_20260404_140525`
- `Related_Works\runs\cut_5x5\video\head_20260404_140551`

The final video evidence is retained under `Cycle-NCE\video`, which has `summary.json` and mp4 outputs. The `Cycle-NCE\video\summary.json` points to `head_20260404_140655`, not these five local work-frame directories.

Expected release: about 3,065.463 MB.

## Still Pending

- `Related_Works\baseline_pipeline\results\cut\protocol_smoke_cut`
- `Related_Works\baseline_pipeline\results\samam_wsl_mamba_b2_15ep_15000\curve_eval_sb_5src`
- `SchrodingerBridge\exp\distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote\full_eval`
- `Related_Works\baseline_pipeline\results\timing_20260602\samst_wikiart512_curve_midpoints`
- The rest of the media candidate list below the current top group.
