# Run Record: overfit50-style-force-balance-v1-cycle4

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-style-force-balance-v1-cycle4`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `9/25`
- Family rank (metric runs): `9/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.521914 |
| best_transfer_clip_style_epoch | 40 |
| best_transfer_classifier_acc | 0.520000 |
| latest_transfer_content_lpips | 0.675383 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.029619`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.071341`
- Style band: `mid_high(0.52-0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 40 |
| train_csv_count | 1 |
| latest_train_epoch | 40 |
| latest_train_loss | 5.345668 |
| latest_train_lr | 0.000010 |
| checkpoint_count | 1 |
| snapshot_count | 1 |
| snapshot_model_hash_count | 1 |
| snapshot_losses_hash_count | 1 |
| snapshot_trainer_hash_count | 1 |

## Config Excerpt (Latest Snapshot)

### Model

| key | value |
|---|---|
| `style_ref_gain` | `1` |

### Loss

| key | value |
|---|---|
| `w_code` | `10` |
| `w_cycle` | `10` |
| `w_nce` | `1` |

## Assessment

### Strengths

- Strict comparable with complete matrix and non-zero LPIPS.
- Style score is above 0.50.
- Snapshot hashes are stable across captured snapshots.

### Risks

- Classifier accuracy is low (<0.60), style identity is unstable.
- Content LPIPS is high (>=0.60), indicating stronger content drift.
- No summary_history rounds; trend stability cannot be verified.

### Suggested Next Step

- Reduce style-induced drift first (lower aggressive style pressure or add stronger structure/content constraints).
