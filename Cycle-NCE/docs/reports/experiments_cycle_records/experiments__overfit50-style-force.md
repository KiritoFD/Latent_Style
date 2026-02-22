# Run Record: overfit50-style-force

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-style-force`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `24/25`
- Family rank (metric runs): `24/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.443286 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 0.070000 |
| latest_transfer_content_lpips | 0.257564 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.108247`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.149970`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 20 |
| train_csv_count | 1 |
| latest_train_epoch | 20 |
| latest_train_loss | 0.402105 |
| latest_train_lr | 0.000011 |
| checkpoint_count | 4 |
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
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
