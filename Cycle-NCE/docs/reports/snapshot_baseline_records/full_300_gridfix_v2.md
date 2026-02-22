# 实验快照分析：full_300_gridfix_v2

- 实验路径：`experiments-cycle/full_300_gridfix_v2`
- 分析口径：`基线=src_snapshot_20260211_111219` -> `最终=src_snapshot_20260211_202159`（仅看最后快照）
- 快照数量：`7`
- 实验画像：`强探索迭代型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.462860 |
| best_transfer_classifier_acc | 0.370000 |
| latest_transfer_content_lpips | 0.324233 |
| strict 可比 | 是 |
| history_rounds | 1 |
| history_style_delta(last-first) | 0.000000 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 是 | 50 | 0 |
| `losses.py` | 是 | 51 | 60 |
| `trainer.py` | 是 | 23 | 78 |
| `run.py` | 是 | 1 | 6 |

## 基线到最终：配置变化

- 配置变更键数：`7`
- 高影响键变更数：`4`

| key | baseline | last |
|---|---|---|
| `loss.style_loss_source` | `student` | `-` |
| `loss.w_code` | `6.000000` | `0.000000` |
| `loss.w_distill` | `0.700000` | `0.000000` |
| `loss.w_gram` | `0.000000` | `-` |
| `loss.w_moment` | `0.000000` | `-` |
| `loss.w_push` | `1.000000` | `0.000000` |
| `loss.w_semigroup` | `0.040000` | `0.000000` |

## 人工解读

- loss 路径收敛到 student 主路，且最后快照把 w_distill/w_code/w_push/w_semigroup 直接降为 0。
- trainer 同步删掉 gram/moment/idt 指标项，实验目标从“全套约束”转向“窄损失组合”。
- 最终 style 不高（0.4629）但 LPIPS 低（0.3242），偏内容保持。

## 结论（该实验）

- 风格分数不高。
- 内容保持相对更稳。
- 分类一致性偏弱，风格身份稳定性不足。
