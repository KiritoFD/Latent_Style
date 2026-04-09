# 实验快照分析：overfit50-style-force-balance-v1

- 实验路径：`experiments-cycle/experiments/overfit50-style-force-balance-v1`
- 分析口径：`基线=src_snapshot_20260209_215752` -> `最终=src_snapshot_20260209_220733`（仅看最后快照）
- 快照数量：`2`
- 实验画像：`轻量工程改动型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.549101 |
| best_transfer_classifier_acc | 0.890000 |
| latest_transfer_content_lpips | 0.764001 |
| strict 可比 | 是 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 否 | 0 | 0 |
| `trainer.py` | 否 | 0 | 0 |
| `run.py` | 是 | 17 | 0 |

## 基线到最终：配置变化

- 配置变更键数：`0`
- 高影响键变更数：`0`
- 无配置差异。

## 人工解读

- 只改 run.py：增加 CPU 线程配置入口；模型与损失未变。
- 最终 style 很高（0.5491）但 LPIPS 很高（0.7640），仍是“高风格高漂移”型。

## 结论（该实验）

- 风格分数处于当前高位。
- 内容漂移偏高（LPIPS 高），不适合作为稳态生产基线。
