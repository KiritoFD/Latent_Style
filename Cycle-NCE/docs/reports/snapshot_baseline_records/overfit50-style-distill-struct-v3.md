# 实验快照分析：overfit50-style-distill-struct-v3

- 实验路径：`experiments-cycle/overfit50-style-distill-struct-v3`
- 分析口径：`基线=src_snapshot_20260210_115542` -> `最终=src_snapshot_20260210_121412`（仅看最后快照）
- 快照数量：`2`
- 实验画像：`稳定复现型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.516456 |
| best_transfer_classifier_acc | 0.670000 |
| latest_transfer_content_lpips | 0.510532 |
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

- 两次快照首尾无变化，属于完全定版训练。

## 结论（该实验）

- 风格分数中上。
