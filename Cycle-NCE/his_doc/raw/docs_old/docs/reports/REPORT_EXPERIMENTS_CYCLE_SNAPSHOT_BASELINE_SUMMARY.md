# Experiments-Cycle 快照基线分析总结（中文）

- 范围：所有存在 `src_snapshot_*` 的实验（共 18 个）
- 口径：每个实验仅比较“最早快照（基线）”与“最后快照（最终）”
- 单实验文档索引：`docs/reports/REPORT_EXPERIMENTS_CYCLE_SNAPSHOT_BASELINE_INDEX.md`

## 1) 结构分层

| 实验画像 | 数量 |
|---|---:|
| 稳定复现型 | 2 |
| 轻量工程改动型 | 3 |
| 中等改动型 | 1 |
| 强探索迭代型 | 3 |
| 单快照（基线=最终） | 9 |

## 2) 改动最重的实验（基线->最终）

| run | path | snapshots | 代码改动行(+-) | 配置变更键数 | 画像 |
|---|---|---:|---:|---:|---|
| overfit50-distill_low_only | `overfit50-distill_low_only` | 2 | 780 | 49 | 强探索迭代型 |
| full_300_gridfix_v2 | `full_300_gridfix_v2` | 7 | 269 | 7 | 强探索迭代型 |
| overfit50-style-distill-struct-v4 | `overfit50-style-distill-struct-v4` | 6 | 222 | 12 | 强探索迭代型 |
| overfit50-style-distill-struct-v4-mse | `overfit50-style-distill-struct-v4-mse` | 3 | 103 | 0 | 中等改动型 |
| full_250_strong-style | `experiments/full_250_strong-style` | 4 | 62 | 0 | 轻量工程改动型 |
| overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | 2 | 17 | 0 | 轻量工程改动型 |
| overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | 2 | 15 | 0 | 轻量工程改动型 |
| full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | 8 | 0 | 0 | 稳定复现型 |

## 3) 结合指标看稳定候选

| run | path | best_style | cls | lpips | 画像 |
|---|---|---:|---:|---:|---|
| full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | 0.551533 | 1.000000 | 0.697327 | 稳定复现型 |
| overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | 0.549101 | 0.890000 | 0.764001 | 轻量工程改动型 |
| overfit50-style-distill-struct-v1-20 | `experiments/overfit50-style-distill-struct-v1-20` | 0.547901 | 0.970000 | 0.670641 | 单快照（基线=最终） |
| overfit50-style-distill-struct-v2 | `experiments/overfit50-style-distill-struct-v2` | 0.534868 | 0.880000 | 0.645370 | 单快照（基线=最终） |
| overfit50-strong_structure | `overfit50-strong_structure` | 0.534638 | 0.120000 | 0.581129 | 单快照（基线=最终） |
| overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 0.528368 | 0.850000 | 0.551213 | 单快照（基线=最终） |
| overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | 0.526751 | 1.000000 | 0.606507 | 轻量工程改动型 |
| overfit50-style-force-balance-v1-cycle4 | `experiments/overfit50-style-force-balance-v1-cycle4` | 0.521914 | 0.520000 | 0.675383 | 单快照（基线=最终） |
| overfit50-distill_low_only | `overfit50-distill_low_only` | 0.518469 | 0.930000 | 0.545553 | 强探索迭代型 |
| overfit50-style-distill-struct-v3 | `overfit50-style-distill-struct-v3` | 0.516456 | 0.670000 | 0.510532 | 稳定复现型 |

## 4) 人工结论

- 真正“边训边改配方”的核心 run 主要是：`overfit50-distill_low_only`、`full_300_gridfix_v2`、`overfit50-style-distill-struct-v4`。
- `full_300_distill_low_only_v1` 是典型稳定基线：快照多但首尾一致，说明是固定配方训练。
- 多个高分 run（如 `overfit50-style-force-balance-v1`）风格分高但 LPIPS 偏高，不适合作为内容保真基线。
- 单快照实验（9 个）无法从快照角度做“中途策略变化”归因，只能看最终结果。

## 5) 建议的下一步

1. 先锁定稳定候选（快照稳定 + strict 可比）作为主线复现基线。
2. 对强探索 run，按“代码冻结、只改配置”再做分离回放，减少归因耦合。
3. 对单快照 run 增加中期快照与固定 epoch full_eval，补齐可追溯链路。
