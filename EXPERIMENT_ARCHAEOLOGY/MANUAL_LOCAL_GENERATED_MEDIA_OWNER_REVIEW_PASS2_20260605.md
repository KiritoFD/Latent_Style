# Local Generated Media Owner Review Pass 2 - 2026-06-05

This pass continues the generated-media owner review from `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_20260605.md`. No deletion was performed in this pass.

## Reviewed Directories

- `Related_Works\baseline_pipeline\results\cut\protocol_smoke_cut`: retained. It has `summary.json`, `metrics.csv`, `summary_grid.png`, class directories, and protocol outputs.
- `Related_Works\baseline_pipeline\results\samam_wsl_mamba_b2_15ep_15000\curve_eval_sb_5src`: retained. It has curve metrics, generation logs, and per-step summaries/metrics through `step_015000`.
- `SchrodingerBridge\exp\distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote\full_eval`: retained. It has epoch-level full-eval summaries/metrics and generated images.
- `Related_Works\baseline_pipeline\results\timing_20260602\samst_wikiart512_curve_midpoints`: retained. It has `epoch_0005` and `epoch_0010` summaries, metrics, ArtFID aggregates, and eval rerun log.
- `Related_Works\baseline_pipeline\results\samst\protocol_a_800`: retained. It has summary, metrics, ArtFID aggregates, summary grid, and class outputs.
- `Related_Works\baseline_pipeline\results\styleid\images`: retained despite exact duplicate files in class directories. It is explicitly referenced by `SchrodingerBridge\docs\experiments\2026-05-11-baseline-reproduction-progress.md`, and deleting it would make existing docs stale.
- `Related_Works\baseline_pipeline\results\s2wat\images`: retained. It is referenced by the same baseline reproduction progress doc and has no exact same-name/same-size duplicate replacement.
- `Related_Works\baseline_pipeline\results\seedream45_api\protocol_a_800`: retained. The fake eval checkpoint in this directory was already deleted earlier; the protocol images remain evidence.

## Duplicate Checks

- `styleid/images`: 1001 of 1001 aggregate files have exact same-name/same-size duplicates in class directories, but the aggregate path is documented, so no deletion without owner approval or doc/index migration.
- `s2wat/images`: 0 of 1000 aggregate files matched exact same-name/same-size duplicates under class/protocol directories.
- `seedream45_api/protocol_a_800/images`: 0 of 721 aggregate files matched exact same-name/same-size duplicates under class directories.

## Remaining Work

Continue down the generated-media candidate list after this pass. Deletion remains whitelist-only.
