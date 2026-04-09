# Run Record: overfit50-upscale-balance-v2-re-0.1cycle

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-upscale-balance-v2-re-0.1cycle`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `14/25`
- Family rank (metric runs): `13/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.506272 |
| best_transfer_clip_style_epoch | 80 |
| best_transfer_classifier_acc | 0.230000 |
| latest_transfer_content_lpips | 0.434735 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.045261`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.086983`
- Style band: `mid(0.50-0.52)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 80 |
| train_csv_count | 4 |
| latest_train_epoch | - |
| latest_train_loss | - |
| latest_train_lr | - |
| checkpoint_count | 9 |
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
