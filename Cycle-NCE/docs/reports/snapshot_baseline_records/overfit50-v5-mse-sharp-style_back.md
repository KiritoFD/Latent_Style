# 实验快照分析：overfit50-v5-mse-sharp-style_back

- 实验路径：`experiments-cycle/overfit50-v5-mse-sharp-style_back`
- 分析口径：`基线=src_snapshot_20260210_232152` -> `最终=src_snapshot_20260210_235121`（仅看最后快照）
- 快照数量：`2`
- 实验画像：`轻量工程改动型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.526751 |
| best_transfer_classifier_acc | 1.000000 |
| latest_transfer_content_lpips | 0.606507 |
| strict 可比 | 是 |
| history_rounds | 1 |
| history_style_delta(last-first) | 0.000000 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 是 | 14 | 1 |
| `trainer.py` | 否 | 0 | 0 |
| `run.py` | 否 | 0 | 0 |

## 基线到最终：配置变化

- 配置变更键数：`0`
- 高影响键变更数：`0`
- 无配置差异。

## 人工解读

- 仅 losses.py 小幅更新：distill 从统一 L1 改成可选 low-only + cross-domain-only 聚合。
- 模型与 trainer 不变，属于单一损失开关试验；style 与 cls 仍高，但 LPIPS 偏高。

## 结论（该实验）

- 风格分数中上。
- 内容漂移偏高（LPIPS 高），不适合作为稳态生产基线。
