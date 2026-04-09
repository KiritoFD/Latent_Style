# Run Record: adacut_overfit50-lightonly

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/adacut_overfit50-lightonly`
- Family: `adacut`
- Tier: `B_partial`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `1/2`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.470728 |
| best_transfer_clip_style_epoch | 100 |
| best_transfer_classifier_acc | 0.320000 |
| latest_transfer_content_lpips | 0.000000 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.080805`
- Family best: `adacut_overfit50-lightonly` (0.470728)
- Delta vs family best: `+0.000000`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 100 |
| train_csv_count | 2 |
| latest_train_epoch | - |
| latest_train_loss | - |
| latest_train_lr | - |
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
