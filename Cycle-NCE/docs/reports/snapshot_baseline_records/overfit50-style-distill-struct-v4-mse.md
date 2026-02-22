# 实验快照分析：overfit50-style-distill-struct-v4-mse

- 实验路径：`experiments-cycle/overfit50-style-distill-struct-v4-mse`
- 分析口径：`基线=src_snapshot_20260210_211143` -> `最终=src_snapshot_20260210_223207`（仅看最后快照）
- 快照数量：`3`
- 实验画像：`中等改动型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.497539 |
| best_transfer_classifier_acc | 0.640000 |
| latest_transfer_content_lpips | 0.437998 |
| strict 可比 | 是 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 是 | 27 | 0 |
| `trainer.py` | 是 | 74 | 2 |
| `run.py` | 否 | 0 | 0 |

## 基线到最终：配置变化

- 配置变更键数：`0`
- 高影响键变更数：`0`
- 无配置差异。

## 人工解读

- 配置不变，但 losses/trainer 有实现迭代：补入 delta_tv、cycle_edge_strength、summary_history 聚合。
- 属于“实现补丁 + 评测增强”，不是目标函数大迁移；最终 style=0.4975，略低于 v4。

## 结论（该实验）

- 风格分数不高。
- 内容保持相对更稳。
