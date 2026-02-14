# 实验快照分析：overfit50-style-distill-struct-v4

- 实验路径：`experiments-cycle/overfit50-style-distill-struct-v4`
- 分析口径：`基线=src_snapshot_20260210_121539` -> `最终=src_snapshot_20260210_211128`（仅看最后快照）
- 快照数量：`6`
- 实验画像：`强探索迭代型`

## 结果概览

| 指标 | 数值 |
|---|---:|
| best_transfer_clip_style | 0.499004 |
| best_transfer_classifier_acc | 0.550000 |
| latest_transfer_content_lpips | 0.433740 |
| strict 可比 | 是 |

## 基线到最终：代码变化

| 文件 | 是否变化 | 增加行 | 删除行 |
|---|---|---:|---:|
| `model.py` | 否 | 0 | 0 |
| `losses.py` | 是 | 51 | 8 |
| `trainer.py` | 是 | 61 | 82 |
| `run.py` | 是 | 17 | 3 |

## 基线到最终：配置变化

- 配置变更键数：`12`
- 高影响键变更数：`5`

| key | baseline | last |
|---|---|---|
| `loss.cycle_ramp_epochs` | `30` | `12` |
| `loss.cycle_warmup_epochs` | `20` | `5` |
| `loss.nce_ramp_epochs` | `30` | `10` |
| `loss.nce_warmup_epochs` | `20` | `5` |
| `loss.struct_ramp_epochs` | `25` | `10` |
| `loss.struct_warmup_epochs` | `15` | `4` |
| `loss.w_cycle` | `3.000000` | `8.000000` |
| `loss.w_distill` | `10.000000` | `8.000000` |
| `loss.w_edge` | `-` | `0.250000` |
| `loss.w_gram` | `120.000000` | `80.000000` |
| `loss.w_nce` | `2.000000` | `3.500000` |
| `loss.w_struct` | `3.000000` | `0.750000` |

## 人工解读

- 最后快照把目标从 struct 主导切向 cycle/nce 更强：w_cycle 3→8、w_struct 3→0.75、w_nce 2→3.5。
- loss 新增 cycle/struct 的可配置对齐形式（l1/mse + lowpass 混合），同时 run.py 偏向 CPU 低负载。
- 最终 style=0.499，未优于 v3，说明这轮迁移方向收益不足。

## 结论（该实验）

- 风格分数不高。
- 内容保持相对更稳。
- 分类一致性偏弱，风格身份稳定性不足。
