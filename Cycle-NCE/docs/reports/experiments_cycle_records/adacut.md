# Run Record: adacut

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/adacut`
- Family: `adacut`
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
- Family best: `adacut_overfit50-lightonly` (0.470728)
- Delta vs family best: `-`
- Style band: `no_metric`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | - |
| train_csv_count | 2 |
| latest_train_epoch | 33 |
| latest_train_loss | 6.801885 |
| latest_train_lr | 0.000099 |
| checkpoint_count | 3 |
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
