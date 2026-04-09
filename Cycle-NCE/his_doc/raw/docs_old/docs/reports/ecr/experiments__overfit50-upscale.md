# Run Record: overfit50-upscale

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/overfit50-upscale`
- Family: `overfit50`
- Tier: `B_partial`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `1/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.593255 |
| best_transfer_clip_style_epoch | 40 |
| best_transfer_classifier_acc | 0.930000 |
| latest_transfer_content_lpips | 0.000000 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `+0.041722`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `+0.000000`
- Style band: `high(>=0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 40 |
| train_csv_count | 2 |
| latest_train_epoch | 40 |
| latest_train_loss | 0.302137 |
| latest_train_lr | 0.000010 |
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

- Style score is in the current high band (>=0.54).
- Classifier accuracy is strong (>=0.85).
- Content LPIPS is relatively low (<=0.45).

### Risks

- Metric exists but the run is not strict-comparable.
- No summary_history rounds; trend stability cannot be verified.
- No src_snapshot captured; reproducibility trace is limited.

### Suggested Next Step

- Keep as candidate and validate with more history rounds plus fixed eval protocol.
