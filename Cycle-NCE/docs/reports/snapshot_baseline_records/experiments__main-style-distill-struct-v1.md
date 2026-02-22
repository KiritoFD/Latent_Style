# 实验快照分析：main-style-distill-struct-v1

- 实验路径：`experiments-cycle/experiments/main-style-distill-struct-v1`
- 分析口径：`基线=src_snapshot_20260210_004525` -> `最终=src_snapshot_20260210_004525`（仅看最后快照）
- 快照数量：`1`
- 实验画像：`单快照（基线=最终）`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | - |
| best_transfer_classifier_acc | - |
| latest_transfer_content_lpips | - |
| strict 可比 | 否 |

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

- 该实验只有一个快照，基线与最终一致，无法从快照层面判断中途策略漂移。
- 若要做归因，建议增加中期快照或固定 epoch 的 full_eval 对照。

## 结论（该实验）

- 缺少可用 style 指标，不建议纳入竞争排序。
