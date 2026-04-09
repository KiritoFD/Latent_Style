# Run Record: overfit50-v5-mse-sharp-style_back

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/overfit50-v5-mse-sharp-style_back`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `8/25`
- Family rank (metric runs): `8/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.526751 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 1.000000 |
| latest_transfer_content_lpips | 0.606507 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 1 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.024783`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.066505`
- Style band: `mid_high(0.52-0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 20 |
| train_csv_count | 2 |
| latest_train_epoch | - |
| latest_train_loss | - |
| latest_train_lr | - |
| checkpoint_count | 1 |
| snapshot_count | 2 |
| snapshot_model_hash_count | 1 |
| snapshot_losses_hash_count | 2 |
| snapshot_trainer_hash_count | 1 |

## Config Excerpt (Latest Snapshot)

### Model

| key | value |
|---|---|
| `normalize_style_spatial_maps` | `true` |
| `style_delta_lowfreq_gain` | `0.35` |
| `style_gate_floor` | `0.7` |
| `style_ref_gain` | `1` |
| `style_spatial_dec_gain_32` | `0.25` |
| `style_spatial_pre_gain_16` | `0.4` |
| `style_texture_gain` | `0.35` |
| `use_decoder_adagn` | `true` |
| `use_decoder_spatial_inject` | `true` |
| `use_delta_highpass_bias` | `true` |
| `use_output_style_affine` | `true` |
| `use_style_delta_gate` | `true` |
| `use_style_spatial_highpass` | `false` |

### Loss

| key | value |
|---|---|
| `w_code` | `4` |
| `w_cycle` | `8` |
| `w_distill` | `8` |
| `w_edge` | `0.25` |
| `w_nce` | `3.5` |
| `w_push` | `1` |
| `w_struct` | `0.75` |

## Assessment

### Strengths

- Strict comparable with complete matrix and non-zero LPIPS.
- Style score is above 0.50.
- Classifier accuracy is strong (>=0.85).

### Risks

- Content LPIPS is high (>=0.60), indicating stronger content drift.
- Code/config drift exists across snapshots.

### Suggested Next Step

- Reduce style-induced drift first (lower aggressive style pressure or add stronger structure/content constraints).
