# Run Record: full_300_strong-style-v1

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/full_300_strong-style-v1`
- Family: `full_300`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `21/25`
- Family rank (metric runs): `4/4`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.459679 |
| best_transfer_clip_style_epoch | 100 |
| best_transfer_classifier_acc | 0.170000 |
| latest_transfer_content_lpips | 0.295542 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 2 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.091855`
- Family best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs family best: `-0.091855`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 100 |
| train_csv_count | 1 |
| latest_train_epoch | 127 |
| latest_train_loss | 0.794683 |
| latest_train_lr | 0.000129 |
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

- Strict comparable with complete matrix and non-zero LPIPS.
- Content LPIPS is relatively low (<=0.45).

### Risks

- Style score is below 0.50.
- Classifier accuracy is low (<0.60), style identity is unstable.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
