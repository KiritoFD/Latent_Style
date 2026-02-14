# Run Record: full_300_distill_low_only_v1

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/full_300_distill_low_only_v1`
- Family: `full_300`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `1/25`
- Family rank (metric runs): `1/4`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.551533 |
| best_transfer_clip_style_epoch | 50 |
| best_transfer_classifier_acc | 1.000000 |
| latest_transfer_content_lpips | 0.697327 |
| matrix_eval_count_mean | 30 |
| matrix_complete_square | true |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `+0.000000`
- Family best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs family best: `+0.000000`
- Style band: `high(>=0.54)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 50 |
| train_csv_count | 8 |
| latest_train_epoch | 58 |
| latest_train_loss | 0.271563 |
| latest_train_lr | 0.000137 |
| checkpoint_count | 2 |
| snapshot_count | 8 |
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
| `style_texture_gain` | `0.45` |
| `use_decoder_adagn` | `true` |
| `use_decoder_spatial_inject` | `true` |
| `use_delta_highpass_bias` | `true` |
| `use_output_style_affine` | `true` |
| `use_style_delta_gate` | `true` |
| `use_style_spatial_blur` | `true` |
| `use_style_spatial_highpass` | `false` |

### Loss

| key | value |
|---|---|
| `distill_cross_domain_only` | `true` |
| `distill_low_only` | `true` |
| `w_code` | `6` |
| `w_cycle` | `0.3` |
| `w_delta_tv` | `0.004` |
| `w_distill` | `0.25` |
| `w_edge` | `0.08` |
| `w_nce` | `0.35` |
| `w_push` | `1.5` |
| `w_struct` | `0.08` |

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
