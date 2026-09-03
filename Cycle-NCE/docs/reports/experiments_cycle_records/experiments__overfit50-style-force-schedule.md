# Run Record: overfit50-style-force-schedule

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-style-force-schedule`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `4/25`
- Family rank (metric runs): `4/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.543730 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 0.260000 |
| latest_transfer_content_lpips | 0.582351 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.007803`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.049525`
- Style band: `high(>=0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 20 |
| train_csv_count | 1 |
| latest_train_epoch | 24 |
| latest_train_loss | 8.587028 |
| latest_train_lr | 0.000083 |
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
- Style score is in the current high band (>=0.54).

### Risks

- Classifier accuracy is low (<0.60), style identity is unstable.
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
