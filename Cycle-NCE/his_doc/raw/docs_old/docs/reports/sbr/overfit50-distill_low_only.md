# 实验快照分析：overfit50-distill_low_only

- 实验路径：`experiments-cycle/overfit50-distill_low_only`
- 分析口径：`基线=src_snapshot_20260210_235141` -> `最终=src_snapshot_20260211_103123`（仅看最后快照）
- 快照数量：`2`
- 实验画像：`强探索迭代型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.518469 |
| best_transfer_classifier_acc | 0.930000 |
| latest_transfer_content_lpips | 0.545553 |
| strict 可比 | 是 |
| history_rounds | 1 |
| history_style_delta(last-first) | 0.000000 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 是 | 327 | 238 |
| `losses.py` | 是 | 168 | 6 |
| `trainer.py` | 是 | 25 | 8 |
| `run.py` | 是 | 7 | 1 |

## 基线到最终：配置变化

- 配置变更键数：`49`
- 高影响键变更数：`14`

| key | baseline | last |
|---|---|---|
| `loss.color_patch_size` | `-` | `1` |
| `loss.cycle_edge_strength` | `-` | `0.050000` |
| `loss.cycle_lowpass_strength` | `-` | `0.050000` |
| `loss.cycle_ramp_epochs` | `12` | `15` |
| `loss.distill_cross_domain_only` | `-` | `true` |
| `loss.distill_low_only` | `-` | `true` |
| `loss.edge_ramp_epochs` | `-` | `15` |
| `loss.edge_warmup_epochs` | `-` | `5` |
| `loss.nce_max_tokens` | `512` | `1024` |
| `loss.nce_spatial_size` | `16` | `32` |
| `loss.nce_warmup_epochs` | `5` | `0` |
| `loss.push_margin` | `0.200000` | `-` |
| `loss.semigroup_cross_domain_only` | `-` | `true` |
| `loss.semigroup_detach_midpoint` | `-` | `false` |
| `loss.semigroup_h_max` | `-` | `1.000000` |
| `loss.semigroup_h_min` | `-` | `0.250000` |
| `loss.semigroup_loss_type` | `-` | `l1` |
| `loss.semigroup_lowpass_strength` | `-` | `0.250000` |
| `loss.semigroup_ramp_epochs` | `-` | `15` |
| `loss.semigroup_warmup_epochs` | `-` | `5` |
| `loss.stroke_patch_randomize` | `-` | `true` |
| `loss.stroke_patch_sizes` | `-` | `[3, 5]` |
| `loss.struct_lowpass_strength` | `-` | `0.150000` |
| `loss.struct_ramp_epochs` | `10` | `15` |
| `loss.struct_warmup_epochs` | `4` | `5` |
| `loss.w_code` | `4.000000` | `-` |
| `loss.w_color_moment` | `-` | `6.000000` |
| `loss.w_cycle` | `8.000000` | `0.300000` |
| `loss.w_delta_tv` | `-` | `0.002000` |
| `loss.w_distill` | `8.000000` | `1.000000` |
| `loss.w_edge` | `0.250000` | `0.200000` |
| `loss.w_gram` | `80.000000` | `0.000000` |
| `loss.w_moment` | `2.000000` | `0.000000` |
| `loss.w_nce` | `3.500000` | `1.000000` |
| `loss.w_push` | `1.000000` | `-` |
| `loss.w_semigroup` | `-` | `0.050000` |
| `loss.w_stroke_gram` | `-` | `60.000000` |
| `loss.w_struct` | `0.750000` | `0.200000` |
| `loss.w_style_spatial_tv` | `-` | `0.001000` |
| `model.normalize_style_spatial_maps` | `true` | `false` |
| `model.residual_gain` | `0.220000` | `0.400000` |
| `model.style_delta_lowfreq_gain` | `0.350000` | `0.100000` |
| `model.style_force_gain` | `0.600000` | `0.700000` |
| `model.style_gate_floor` | `0.700000` | `0.000000` |
| `model.style_texture_gain` | `0.350000` | `0.450000` |
| `model.upsample_mode` | `-` | `bilinear` |
| `model.use_downsample_blur` | `-` | `true` |
| `model.use_output_style_affine` | `true` | `false` |
| `model.use_style_spatial_blur` | `-` | `true` |

## 人工解读

- 这是本轮改动最大的 run：model/loss/trainer/run 全线变化，属于配方迁移而非小调参。
- model 新增 _predict_delta + integrate(step_size) 路径，loss 引入 stroke/color/semigroup/style_spatial_tv。
- 配置层 49 个键变化，核心是压低 w_cycle/w_struct/w_nce/w_distill，并打开 stroke/color/semigroup 系。

## 结论（该实验）

- 风格分数中上。
