# Run Record: main-style-distill-struct-v1

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/main-style-distill-struct-v1`
- Family: `experiments_misc`
- Tier: `C_incomplete`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `-`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | - |
| best_transfer_clip_style_epoch | - |
| best_transfer_classifier_acc | - |
| latest_transfer_content_lpips | - |
| matrix_eval_count_mean | - |
| matrix_complete_square | false |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-`
- Family best: `-`
- Delta vs family best: `-`
- Style band: `no_metric`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | - |
| train_csv_count | 1 |
| latest_train_epoch | 7 |
| latest_train_loss | 6.965157 |
| latest_train_lr | 0.000200 |
| checkpoint_count | 0 |
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

- Snapshot hashes are stable across captured snapshots.

### Risks

- No parseable best style metric from summary outputs.
- Classifier metric is missing.
- Content LPIPS metric is missing.
- No summary_history rounds; trend stability cannot be verified.

### Suggested Next Step

- Backfill parseable `summary.json` output so this run can enter direct comparison.
