# Run Record: overfit50-upscale-balance-v2-re

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-upscale-balance-v2-re`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `22/25`
- Family rank (metric runs): `20/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.450307 |
| best_transfer_clip_style_epoch | 40 |
| best_transfer_classifier_acc | 0.100000 |
| latest_transfer_content_lpips | 0.265898 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.101226`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.142949`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 40 |
| train_csv_count | 1 |
| latest_train_epoch | 40 |
| latest_train_loss | 9.056237 |
| latest_train_lr | 0.000010 |
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
