# Run Record: full_300_gridfix_v2

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/full_300_gridfix_v2`
- Family: `full_300`
- Tier: `A_strict`
- Strict comparable: `yes`
- Strict rank: `20/25`
- Family rank (metric runs): `3/4`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.462860 |
| best_transfer_clip_style_epoch | 50 |
| best_transfer_classifier_acc | 0.370000 |
| latest_transfer_content_lpips | 0.324233 |
| matrix_eval_count_mean | 30 |
| matrix_complete_square | true |
| history_rounds | 1 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.088674`
- Family best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs family best: `-0.088674`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 150 |
| train_csv_count | 21 |
| latest_train_epoch | 191 |
| latest_train_loss | 0.594955 |
| latest_train_lr | 0.000066 |
| checkpoint_count | 8 |
| snapshot_count | 7 |
| snapshot_model_hash_count | 2 |
| snapshot_losses_hash_count | 2 |
| snapshot_trainer_hash_count | 4 |

## Config Excerpt (Latest Snapshot)

### Model

| key | value |
|---|---|
| `normalize_style_spatial_maps` | `false` |
| `style_delta_lowfreq_gain` | `0.1` |
| `style_gate_floor` | `0` |
| `style_ref_gain` | `1` |
| `style_spatial_dec_gain_32` | `0.08` |
| `style_spatial_pre_gain_16` | `0.32` |
| `style_texture_gain` | `0.28` |
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
| `w_code` | `0` |
| `w_color_moment` | `4` |
| `w_cycle` | `0.2` |
| `w_delta_tv` | `0.0025` |
| `w_distill` | `0` |
| `w_edge` | `0.3` |
| `w_nce` | `0.6` |
| `w_push` | `0` |
| `w_semigroup` | `0` |
| `w_stroke_gram` | `55` |
| `w_struct` | `0.2` |
| `w_style_spatial_tv` | `0.002` |

## Assessment

### Strengths

- Strict comparable with complete matrix and non-zero LPIPS.
- Content LPIPS is relatively low (<=0.45).

### Risks

- Style score is below 0.50.
- Classifier accuracy is low (<0.60), style identity is unstable.
- Code/config drift exists across snapshots.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
