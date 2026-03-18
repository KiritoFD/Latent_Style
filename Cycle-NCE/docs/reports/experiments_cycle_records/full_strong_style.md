# Run Record: full_strong_style

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/full_strong_style`
- Family: `full_other`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `19/25`
- Family rank (metric runs): `1/1`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.475696 |
| best_transfer_clip_style_epoch | 50 |
| best_transfer_classifier_acc | 0.110000 |
| latest_transfer_content_lpips | 0.325611 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 1 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.075837`
- Family best: `full_strong_style` (0.475696)
- Delta vs family best: `+0.000000`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 50 |
| train_csv_count | 1 |
| latest_train_epoch | 50 |
| latest_train_loss | 0.879213 |
| latest_train_lr | 0.000188 |
| checkpoint_count | 5 |
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
