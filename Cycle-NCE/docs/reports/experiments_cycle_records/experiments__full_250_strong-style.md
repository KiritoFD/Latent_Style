# Run Record: full_250_strong-style

- Generated: 2026-02-13 15:56:59
- Path: `experiments-cycle/experiments/full_250_strong-style`
- Family: `full_250`
- Tier: `B_partial`
- Strict comparable: `no`
- Strict rank: `-`
- Family rank (metric runs): `1/2`

## Metrics

| metric | value |
|---|---|
| best_transfer_clip_style | 0.000000 |
| best_transfer_clip_style_epoch | 250 |
| best_transfer_classifier_acc | 0.280000 |
| latest_transfer_content_lpips | 0.000000 |
| matrix_eval_count_mean | 50 |
| matrix_complete_square | false |
| history_rounds | 0 |

## Comparison

- Global strict best: `full_300_distill_low_only_v1` (0.551533)
- Delta vs global strict best: `-0.551533`
- Family best: `full_250_strong-style` (0.000000)
- Delta vs family best: `+0.000000`
- Style band: `low(<0.50)`

## Artifacts And Traceability

| item | value |
|---|---|
| has_full_eval | true |
| latest_epoch | 250 |
| train_csv_count | 4 |
| latest_train_epoch | 250 |
| latest_train_loss | 8.239717 |
| latest_train_lr | 0.000005 |
| checkpoint_count | 5 |
| snapshot_count | 4 |
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

- Content LPIPS is relatively low (<=0.45).
- Snapshot hashes are stable across captured snapshots.

### Risks

- Metric exists but the run is not strict-comparable.
- Style score is below 0.50.
- Classifier accuracy is low (<0.60), style identity is unstable.
- No summary_history rounds; trend stability cannot be verified.

### Suggested Next Step

- Prioritize classifier consistency (rebalance loss weights before raising style gain).
