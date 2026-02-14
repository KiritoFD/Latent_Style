# Run Record: overfit50-style-distill-struct-v1-20

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-style-distill-struct-v1-20`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `3/25`
- Family rank (metric runs): `3/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.547901 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 0.970000 |
| latest_transfer_content_lpips | 0.670641 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.003633`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.045355`
- Style band: `high(>=0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 20 |
| train_csv_count | 1 |
| latest_train_epoch | 20 |
| latest_train_loss | 10.999520 |
| latest_train_lr | 0.000011 |
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
- Style score is in the current high band (>=0.54).
- Classifier accuracy is strong (>=0.85).
- Snapshot hashes are stable across captured snapshots.

### Risks

- Content LPIPS is high (>=0.60), indicating stronger content drift.
- No summary_history rounds; trend stability cannot be verified.

### Suggested Next Step

- Reduce style-induced drift first (lower aggressive style pressure or add stronger structure/content constraints).
