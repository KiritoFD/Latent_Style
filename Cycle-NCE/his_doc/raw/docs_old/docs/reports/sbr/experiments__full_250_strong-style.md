# 实验快照分析：full_250_strong-style

- 实验路径：`experiments-cycle/experiments/full_250_strong-style`
- 分析口径：`基线=src_snapshot_20260210_012059` -> `最终=src_snapshot_20260210_014354`（仅看最后快照）
- 快照数量：`4`
- 实验画像：`轻量工程改动型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.000000 |
| best_transfer_classifier_acc | 0.280000 |
| latest_transfer_content_lpips | 0.000000 |
| strict 可比 | 否 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 否 | 0 | 0 |
| `trainer.py` | 否 | 0 | 0 |
| `run.py` | 是 | 62 | 0 |

## 基线到最终：配置变化

- 配置变更键数：`0`
- 高影响键变更数：`0`
- 无配置差异。

## 人工解读

- model/losses/trainer 未变，只改 run.py 的 CPU 线程与 affinity 控制，属于系统侧优化。
- 该实验 style=0 且 LPIPS=0，评测口径不完整，不能用于主排序结论。

## 结论（该实验）

- 风格分数不高。
- 内容保持相对更稳。
- 分类一致性偏弱，风格身份稳定性不足。
