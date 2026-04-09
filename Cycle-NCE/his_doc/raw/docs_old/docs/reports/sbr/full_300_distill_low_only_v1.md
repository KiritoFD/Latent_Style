# 实验快照分析：full_300_distill_low_only_v1

- 实验路径：`experiments-cycle/full_300_distill_low_only_v1`
- 分析口径：`基线=src_snapshot_20260211_001416` -> `最终=src_snapshot_20260211_085727`（仅看最后快照）
- 快照数量：`8`
- 实验画像：`稳定复现型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.551533 |
| best_transfer_classifier_acc | 1.000000 |
| latest_transfer_content_lpips | 0.697327 |
| strict 可比 | 是 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 否 | 0 | 0 |
| `trainer.py` | 否 | 0 | 0 |
| `run.py` | 否 | 0 | 0 |

## 基线到最终：配置变化

- 配置变更键数：`0`
- 高影响键变更数：`0`
- 无配置差异。

## 人工解读

- 8 个快照首尾完全一致（代码和配置都不变），属于稳定长跑而不是边训边改。
- 它是当前 strict 集合 top1（style=0.5515），但 LPIPS=0.6973 偏高，说明风格收益伴随内容漂移。

## 结论（该实验）

- 风格分数处于当前高位。
- 内容漂移偏高（LPIPS 高），不适合作为稳态生产基线。
