# Run Record: overfit50-upscale-balance-v2

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-upscale-balance-v2`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `15/25`
- Family rank (metric runs): `14/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.504675 |
| best_transfer_clip_style_epoch | 40 |
| best_transfer_classifier_acc | 0.500000 |
| latest_transfer_content_lpips | 0.441309 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.046858`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.088580`
- Style band: `mid(0.50-0.52)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 40 |
| train_csv_count | 6 |
| latest_train_epoch | 45 |
| latest_train_loss | 7.346957 |
| latest_train_lr | 0.000015 |
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
- Style score is above 0.50.
- Content LPIPS is relatively low (<=0.45).

### Risks

- Classifier accuracy is low (<0.60), style identity is unstable.
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
