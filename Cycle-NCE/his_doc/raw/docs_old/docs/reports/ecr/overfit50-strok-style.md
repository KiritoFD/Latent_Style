# Run Record: overfit50-strok-style

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/overfit50-strok-style`
- Family: `overfit50`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `12/25`
- Family rank (metric runs): `12/25`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.516445 |
| best_transfer_clip_style_epoch | 20 |
| best_transfer_classifier_acc | 0.930000 |
| latest_transfer_content_lpips | 0.386163 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | true |
| history_rounds | 2 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.035088`
- Family best: `overfit50-upscale` (0.593255)
- Delta vs family best: `-0.076810`
- Style band: `mid(0.50-0.52)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 40 |
| train_csv_count | 1 |
| latest_train_epoch | 42 |
| latest_train_loss | 0.365439 |
| latest_train_lr | 0.000053 |
| checkpoint_count | 2 |
| snapshot_count | 1 |
| snapshot_model_hash_count | 1 |
| snapshot_losses_hash_count | 1 |
| snapshot_trainer_hash_count | 1 |

## Config Excerpt (Latest Snapshot)

### Model

| key | value |
|---|---|
| `normalize_style_spatial_maps` | `false` |
| `style_delta_lowfreq_gain` | `0.1` |
| `style_gate_floor` | `0` |
| `style_ref_gain` | `1` |
| `style_spatial_dec_gain_32` | `0.25` |
| `style_spatial_pre_gain_16` | `0.4` |
| `style_texture_gain` | `0.45` |
| `use_decoder_adagn` | `true` |
| `use_decoder_spatial_inject` | `true` |
| `use_delta_highpass_bias` | `true` |
| `use_output_style_affine` | `false` |
| `use_style_delta_gate` | `true` |
| `use_style_spatial_blur` | `true` |
| `use_style_spatial_highpass` | `false` |

### Loss

| key | value |
|---|---|
| `distill_cross_domain_only` | `true` |
| `distill_low_only` | `true` |
| `w_color_moment` | `6` |
| `w_cycle` | `0.3` |
| `w_delta_tv` | `0.002` |
| `w_distill` | `1` |
| `w_edge` | `0.2` |
| `w_nce` | `1` |
| `w_semigroup` | `0.05` |
| `w_stroke_gram` | `60` |
| `w_struct` | `0.2` |
| `w_style_spatial_tv` | `0.001` |

## Assessment

### Strengths

- Strict comparable with complete matrix and non-zero LPIPS.
- Style score is above 0.50.
- Classifier accuracy is strong (>=0.85).
- Content LPIPS is relatively low (<=0.45).
- Snapshot hashes are stable across captured snapshots.

### Risks

- None.

### Suggested Next Step

- Use this run as baseline and test moderate style-strength increases with strict eval cadence.
