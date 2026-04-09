# Run Record: overfit50-clipstyle-probe-v1

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/overfit50-clipstyle-probe-v1`
- Family: `overfit50`
- Tier: `C_incomplete`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `-`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | - |
| best_transfer_clip_style_epoch | - |
| best_transfer_classifier_acc | - |
| latest_transfer_content_lpips | - |
| matrix_eval_count_mean | - |
| matrix_complete_square | false |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-`
- Style band: `no_metric`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | - |
| train_csv_count | 0 |
| latest_train_epoch | - |
| latest_train_loss | - |
| latest_train_lr | - |
| checkpoint_count | 2 |
| snapshot_count | 0 |
| snapshot_model_hash_count | 0 |
| snapshot_losses_hash_count | 0 |
| snapshot_trainer_hash_count | 0 |

## Config Excerpt (Latest Snapshot)

### Model

- None

### Loss

- None

## Assessment

### Strengths

- None.

### Risks

- No parseable best style metric from summary outputs.
- Classifier metric is missing.
- Content LPIPS metric is missing.
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Backfill parseable `summary.json` output so this run can enter direct comparison.
