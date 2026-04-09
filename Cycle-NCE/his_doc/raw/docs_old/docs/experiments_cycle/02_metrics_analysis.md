# Step 2: Metrics Analysis (Style / Content / Classifier)

## 口径说明
- 基础表：`docs/experiments_cycle/data/runs_metrics.csv`
- 主指标：`best_transfer_clip_style`
- 辅助指标：`best_transfer_classifier_acc`、`latest_transfer_content_lpips`
- 本文主要使用 `strict` 子集：
  - `matrix_complete_square=True`
  - `best_transfer_clip_style` 可解析
  - `latest_transfer_content_lpips > 0`
- 原因：剔除缺失 summary、LPIPS=0 的旧评测格式，降低误判。

样本规模：
- 全部可解析指标 run：`35`
- `strict` 可比 run：`25`

## 风格分数分布（strict）
- `clip_style >= 0.50`：`15/25`
- `clip_style >= 0.52`：`9/25`
- `clip_style >= 0.54`：`4/25`
- `clip_style >= 0.56`：`0/25`

结论：当前实验池在可比口径下，`0.54` 已经是较高段位，`0.56+` 还没有稳定样本。

## Top Runs（按 best_transfer_clip_style）

| run | path | best_clip_style | cls_acc | content_lpips | eval_count | snapshots |
|---|---|---:|---:|---:|---:|---:|
| full_300_distill_low_only_v1 | full_300_distill_low_only_v1 | 0.551533 | 1.000 | 0.697327 | 30 | 8 |
| overfit50-style-force-balance-v1 | experiments/overfit50-style-force-balance-v1 | 0.549101 | 0.890 | 0.764001 | 50 | 2 |
| overfit50-style-distill-struct-v1-20 | experiments/overfit50-style-distill-struct-v1-20 | 0.547901 | 0.970 | 0.670641 | 50 | 1 |
| overfit50-style-force-schedule | experiments/overfit50-style-force-schedule | 0.543730 | 0.260 | 0.582351 | 50 | 0 |
| overfit50-style-distill-struct-v2 | experiments/overfit50-style-distill-struct-v2 | 0.534868 | 0.880 | 0.645370 | 50 | 1 |
| overfit50-strong_structure | overfit50-strong_structure | 0.534638 | 0.120 | 0.581129 | 50 | 1 |
| overfit50-style-distill-struct-v2 | overfit50-style-distill-struct-v2 | 0.528368 | 0.850 | 0.551213 | 50 | 1 |
| overfit50-v5-mse-sharp-style_back | overfit50-v5-mse-sharp-style_back | 0.526751 | 1.000 | 0.606507 | 50 | 2 |
| overfit50-style-force-balance-v1-cycle4 | experiments/overfit50-style-force-balance-v1-cycle4 | 0.521914 | 0.520 | 0.675383 | 50 | 1 |
| overfit50-distill_low_only | overfit50-distill_low_only | 0.518469 | 0.930 | 0.545553 | 50 | 2 |
| overfit50-style-distill-struct-v3 | overfit50-style-distill-struct-v3 | 0.516456 | 0.670 | 0.510532 | 50 | 2 |
| overfit50-strok-style | overfit50-strok-style | 0.516445 | 0.930 | 0.386163 | 50 | 1 |
| full_300-map16+32 | full_300-map16+32 | 0.509921 | 0.780 | 0.424217 | 50 | 0 |

## 关于 `full_300-map16+32`
- 在 strict 口径下全局排名：`13/25`
- 在 `full_300` 家族内排名：`2/4`
  - `#1 full_300_distill_low_only_v1`：0.551533（但 LPIPS=0.697，且 eval_count=30，仅单点评估）
  - `#2 full_300-map16+32`：0.509921（LPIPS=0.424，history=6，稳定性更好）
- 结论：
  - `map16+32` 不是风格最高，但在“内容保真 + 持续评估轮次”上更均衡。

## 轨迹（有 summary_history 的 8 个 run）

| run | path | rounds | first | last | best | best_epoch | delta(last-first) | last_cls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_300-map16+32 | full_300-map16+32 | 6 | 0.502842 | 0.508573 | 0.509921 | 200 | +0.005731 | 0.760 |
| full_300_gridfix_v2 | full_300_gridfix_v2 | 1 | 0.462860 | 0.462860 | 0.462860 | 50 | +0.000000 | 0.370 |
| full_300_strong-style-v1 | full_300_strong-style-v1 | 2 | 0.458917 | 0.459679 | 0.459679 | 100 | +0.000761 | 0.160 |
| full_strong_style | full_strong_style | 1 | 0.475696 | 0.475696 | 0.475696 | 50 | +0.000000 | 0.110 |
| overfit50-distill_low_only | overfit50-distill_low_only | 1 | 0.518469 | 0.518469 | 0.518469 | 20 | +0.000000 | 0.930 |
| overfit50-strok-style | overfit50-strok-style | 2 | 0.516445 | 0.485160 | 0.516445 | 20 | -0.031285 | 0.520 |
| overfit50-v5-mse-sharp | overfit50-v5-mse-sharp | 2 | 0.439869 | 0.441041 | 0.441041 | 40 | +0.001172 | 0.080 |
| overfit50-v5-mse-sharp-style_back | overfit50-v5-mse-sharp-style_back | 1 | 0.526751 | 0.526751 | 0.526751 | 20 | +0.000000 | 1.000 |

轨迹层观察：
- `full_300-map16+32` 有明显平台期（200 epoch 后提升很小），但没有崩。
- `overfit50-strok-style` 出现后期回落，说明需要更密集 full_eval 或提前停训。
- 多数 run 只有 1 次 full_eval，无法判断泛化趋势。

## 指标关系（strict，n=25）
- corr(`clip_style`, `content_lpips`) = `+0.94`
- corr(`clip_style`, `classifier_acc`) = `+0.73`

解释：
- 当前实验族里，“风格更强”通常伴随更高 LPIPS（内容漂移更大）。
- 单纯推高 clip_style 的路径，内容稳定性风险显著。

## 阶段性结论
- 你想冲 `clip_style=0.65`，按当前分布差距很大（strict 口径最高仅 `0.5515`）。
- 如果目标是“风格提升且不破图”，当前最值得继续跟进的是：
  - `overfit50-strok-style`（0.516 / 0.93 / 0.386）
  - `full_300-map16+32`（0.510 / 0.78 / 0.424，history最完整）
- `0.54+` 的若干高分 run 多伴随高 LPIPS（0.58~0.76），更像“强风格但内容牺牲”配置。
