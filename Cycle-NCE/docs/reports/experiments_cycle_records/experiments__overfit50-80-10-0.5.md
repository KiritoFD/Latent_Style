# Run Record: overfit50-80-10-0.5

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-80-10-0.5`
- Family: `overfit50`
- Tier: `B_partial`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `19/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.452397 |
| best_transfer_clip_style_epoch | 10 |
| best_transfer_classifier_acc | 0.320000 |
| latest_transfer_content_lpips | 0.000000 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.099136`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.140858`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 10 |
| train_csv_count | 1 |
| latest_train_epoch | 10 |
| latest_train_loss | 0.246164 |
| latest_train_lr | 0.000015 |
| checkpoint_count | 0 |
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

- Content LPIPS is relatively low (<=0.45).

### Risks

- Metric exists but the run is not strict-comparable.
- Style score is below 0.50.
- Classifier accuracy is low (<0.60), style identity is unstable.
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
