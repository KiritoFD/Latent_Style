# Run Record: overfit50-style-distill-struct-v4

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/overfit50-style-distill-struct-v4`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `16/25`
- Family rank (metric runs): `15/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.499004 |
| best_transfer_clip_style_epoch | 60 |
| best_transfer_classifier_acc | 0.550000 |
| latest_transfer_content_lpips | 0.433740 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.052529`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.094251`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 60 |
| train_csv_count | 6 |
| latest_train_epoch | - |
| latest_train_loss | - |
| latest_train_lr | - |
| checkpoint_count | 3 |
| snapshot_count | 6 |
| snapshot_model_hash_count | 1 |
| snapshot_losses_hash_count | 2 |
| snapshot_trainer_hash_count | 3 |

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
- Content LPIPS is relatively low (<=0.45).

### Risks

- Style score is below 0.50.
- Classifier accuracy is low (<0.60), style identity is unstable.
- No summary_history rounds; trend stability cannot be verified.
- Code/config drift exists across snapshots.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
