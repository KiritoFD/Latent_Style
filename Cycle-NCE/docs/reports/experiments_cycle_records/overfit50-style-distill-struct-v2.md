# Run Record: overfit50-style-distill-struct-v2

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/overfit50-style-distill-struct-v2`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `7/25`
- Family rank (metric runs): `7/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.528368 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 0.850000 |
| latest_transfer_content_lpips | 0.551213 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.023165`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.064887`
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
| snapshot_count | 1 |
| snapshot_model_hash_count | 1 |
| snapshot_losses_hash_count | 1 |
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
| `w_cycle` | `3` |
| `w_distill` | `10` |
| `w_nce` | `2` |
| `w_push` | `1` |
| `w_struct` | `3` |

## Assessment

### Strengths

- Strict comparable with complete matrix and non-zero LPIPS.
- Style score is above 0.50.
- Classifier accuracy is strong (>=0.85).
- Snapshot hashes are stable across captured snapshots.

### Risks

- No summary_history rounds; trend stability cannot be verified.

### Suggested Next Step

- Keep as candidate and validate with more history rounds plus fixed eval protocol.
