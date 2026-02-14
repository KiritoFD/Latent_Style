# Experiments-Cycle 代码改动影响深度报告（中文）

- 目标：评估“每个代码改动”对风格效果与相关指标（style / cls / lpips / eval_count / history）的影响。
- 数据：`docs/reports/data/snapshot_baseline_vs_last.json` + `docs/experiments_cycle/data/runs_metrics.csv`。
- 范围：所有有快照实验 18 个，其中有真实代码改动的实验 7 个。
- 说明：多数 run 没有严格同配置 A/B；以下“影响”是基于同族对照与代码语义的经验归因，已标注置信度。

## 1) 全量快照实验指标总览（18个）

| run | path | snapshots | 画像 | best_style | cls | lpips | eval_count | history | strict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | 8 | 稳定复现 | 0.551533 | 1.000000 | 0.697327 | 30 | 0 | 是 |
| full_300_gridfix_v2 | `full_300_gridfix_v2` | 7 | 强探索改动 | 0.462860 | 0.370000 | 0.324233 | 30 | 1 | 是 |
| overfit50-style-distill-struct-v4 | `overfit50-style-distill-struct-v4` | 6 | 强探索改动 | 0.499004 | 0.550000 | 0.433740 | 50 | 0 | 是 |
| full_250_strong-style | `experiments/full_250_strong-style` | 4 | 轻量改动 | 0.000000 | 0.280000 | 0.000000 | 50 | 0 | 否 |
| overfit50-style-distill-struct-v4-mse | `overfit50-style-distill-struct-v4-mse` | 3 | 中等改动 | 0.497539 | 0.640000 | 0.437998 | 50 | 0 | 是 |
| overfit50-distill_low_only | `overfit50-distill_low_only` | 2 | 强探索改动 | 0.518469 | 0.930000 | 0.545553 | 50 | 1 | 是 |
| overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | 2 | 轻量改动 | 0.549101 | 0.890000 | 0.764001 | 50 | 0 | 是 |
| overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | 2 | 轻量改动 | 0.526751 | 1.000000 | 0.606507 | 50 | 1 | 是 |
| overfit50-style-distill-struct-v3 | `overfit50-style-distill-struct-v3` | 2 | 稳定复现 | 0.516456 | 0.670000 | 0.510532 | 50 | 0 | 是 |
| overfit50-style-distill-struct-v1-20 | `experiments/overfit50-style-distill-struct-v1-20` | 1 | 单快照 | 0.547901 | 0.970000 | 0.670641 | 50 | 0 | 是 |
| overfit50-style-distill-struct-v2 | `experiments/overfit50-style-distill-struct-v2` | 1 | 单快照 | 0.534868 | 0.880000 | 0.645370 | 50 | 0 | 是 |
| overfit50-strong_structure | `overfit50-strong_structure` | 1 | 单快照 | 0.534638 | 0.120000 | 0.581129 | 50 | 0 | 是 |
| overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 1 | 单快照 | 0.528368 | 0.850000 | 0.551213 | 50 | 0 | 是 |
| overfit50-style-force-balance-v1-cycle4 | `experiments/overfit50-style-force-balance-v1-cycle4` | 1 | 单快照 | 0.521914 | 0.520000 | 0.675383 | 50 | 0 | 是 |
| overfit50-strok-style | `overfit50-strok-style` | 1 | 单快照 | 0.516445 | 0.930000 | 0.386163 | 50 | 2 | 是 |
| overfit50-v5-mse-sharp | `overfit50-v5-mse-sharp` | 1 | 单快照 | 0.441041 | 0.080000 | 0.241780 | 50 | 2 | 是 |
| full_250_strong-style | `full_250_strong-style` | 1 | 单快照 | 0.000000 | 0.460000 | 0.000000 | 50 | 0 | 否 |
| main-style-distill-struct-v1 | `experiments/main-style-distill-struct-v1` | 1 | 单快照 | - | - | - | - | 0 | 否 |

## 2) 同族对照：改动与指标变化

| 对照 | A(style/cls/lpips) | B(style/cls/lpips) | Δstyle | Δcls | Δlpips | 观察 |
|---|---|---|---:|---:|---:|---|
| v3->v4：cycle/nce 加强 + struct 降低 | 0.516456/0.670000/0.510532 | 0.499004/0.550000/0.433740 | -0.017451 | -0.120000 | -0.076792 | 仅作经验对照，非严格 A/B 因果 |
| v4->v4-mse：在 v4 上加入 edge/delta_tv 等实现修正 | 0.499004/0.550000/0.433740 | 0.497539/0.640000/0.437998 | -0.001465 | 0.090000 | 0.004258 | 仅作经验对照，非严格 A/B 因果 |
| v5->v5-style_back：distill 聚合方式调整 | 0.441041/0.080000/0.241780 | 0.526751/1.000000/0.606507 | 0.085709 | 0.920000 | 0.364726 | 仅作经验对照，非严格 A/B 因果 |
| force-balance v1->cycle4：后续训练轮次对照 | 0.549101/0.890000/0.764001 | 0.521914/0.520000/0.675383 | -0.027187 | -0.370000 | -0.088618 | 仅作经验对照，非严格 A/B 因果 |
| distill_low_only->strok-style：相近配方的同类对照 | 0.518469/0.930000/0.545553 | 0.516445/0.930000/0.386163 | -0.002024 | 0.000000 | -0.159390 | 仅作经验对照，非严格 A/B 因果 |
| full_300 分支：gridfix_v2 vs distill_low_only_v1 | 0.462860/0.370000/0.324233 | 0.551533/1.000000/0.697327 | 0.088674 | 0.630000 | 0.373094 | 仅作经验对照，非严格 A/B 因果 |

## 3) 逐实验代码改动影响（7个改动实验）

### full_300_gridfix_v2（`full_300_gridfix_v2`）

- 改动文件：`model.py, losses.py, trainer.py, run.py`；配置变更键：`7`；快照：`src_snapshot_20260211_111219 -> src_snapshot_20260211_202159`。
- 指标：style=`0.462860`，cls=`0.370000`，lpips=`0.324233`，eval_count=`30`，history_rounds=`1`。
- 代码上从 teacher/code 辅助路径收缩到 student 主路径，并在配置里把 w_distill/w_code/w_push/w_semigroup 归零。
- 指标表现为 style 与 cls 显著偏低（0.4629 / 0.37），但 lpips 很低（0.3242）。
- 影响判断：该类“去 teacher/code”改动更像是在换取内容保持，牺牲了风格强度与可辨识性（高置信度）。

### overfit50-style-distill-struct-v4（`overfit50-style-distill-struct-v4`）

- 改动文件：`losses.py, trainer.py, run.py`；配置变更键：`12`；快照：`src_snapshot_20260210_121539 -> src_snapshot_20260210_211128`。
- 指标：style=`0.499004`，cls=`0.550000`，lpips=`0.433740`，eval_count=`50`，history_rounds=`0`。
- 核心变更是损失权重重心从 struct 向 cycle/nce 偏移（w_cycle↑, w_nce↑, w_struct↓）并引入 edge。
- 与 v3 对照：style -0.0175，cls -0.12，lpips -0.0768。
- 影响判断：该权重迁移让内容更稳（lpips 下降）但风格表达和分类一致性下降（中高置信度）。

### full_250_strong-style（`experiments/full_250_strong-style`）

- 改动文件：`run.py`；配置变更键：`0`；快照：`src_snapshot_20260210_012059 -> src_snapshot_20260210_014354`。
- 指标：style=`0.000000`，cls=`0.280000`，lpips=`0.000000`，eval_count=`50`，history_rounds=`0`。
- 仅 run.py 的 CPU 环境线程/affinity 改动，模型与损失不变。
- 该 run 指标口径不完整（style=0, lpips=0），无法做有效风格归因。
- 影响判断：应先补齐评测，再讨论改动对风格影响（高置信度）。

### overfit50-style-distill-struct-v4-mse（`overfit50-style-distill-struct-v4-mse`）

- 改动文件：`losses.py, trainer.py`；配置变更键：`0`；快照：`src_snapshot_20260210_211143 -> src_snapshot_20260210_223207`。
- 指标：style=`0.497539`，cls=`0.640000`，lpips=`0.437998`，eval_count=`50`，history_rounds=`0`。
- 在 v4 基础上增加 cycle_edge_strength / delta_tv 等实现修正，配置层基本不变。
- 与 v4 对照：style -0.0015，cls +0.09，lpips +0.0043。
- 影响判断：属于“小幅结构化修正”，主要改善 cls，几乎不改 style 上限（中置信度）。

### overfit50-distill_low_only（`overfit50-distill_low_only`）

- 改动文件：`model.py, losses.py, trainer.py, run.py`；配置变更键：`49`；快照：`src_snapshot_20260210_235141 -> src_snapshot_20260211_103123`。
- 指标：style=`0.518469`，cls=`0.930000`，lpips=`0.545553`，eval_count=`50`，history_rounds=`1`。
- 这是最大规模迁移：model 新增 integrate(step_size)，loss 引入 stroke/color/semigroup 等新项，配置 49 键变化。
- 最终指标（0.5185/0.93/0.5456）说明该配方可达到较好的 style+cls 平衡，但内容漂移仍中高。
- 与 strok-style 对照时 style 基本相当但 lpips 更高，提示还存在可优化的内容保真空间（中置信度）。

### overfit50-style-force-balance-v1（`experiments/overfit50-style-force-balance-v1`）

- 改动文件：`run.py`；配置变更键：`0`；快照：`src_snapshot_20260209_215752 -> src_snapshot_20260209_220733`。
- 指标：style=`0.549101`，cls=`0.890000`，lpips=`0.764001`，eval_count=`50`，history_rounds=`0`。
- 仅 run.py 的 CPU 线程控制改动，目标是训练稳定性和资源控制，不直接改损失函数。
- 最终 style 很高（0.5491）但 lpips 也高（0.7640）。
- 影响判断：当前指标更像原始配方属性，run.py 工程改动不是决定性风格因素（中置信度）。

### overfit50-v5-mse-sharp-style_back（`overfit50-v5-mse-sharp-style_back`）

- 改动文件：`losses.py`；配置变更键：`0`；快照：`src_snapshot_20260210_232152 -> src_snapshot_20260210_235121`。
- 指标：style=`0.526751`，cls=`1.000000`，lpips=`0.606507`，eval_count=`50`，history_rounds=`1`。
- 仅 losses.py 的 distill 聚合策略变更（支持 low-only + cross-domain-only）。
- 与 v5 对照：style +0.0857，cls +0.92，同时 lpips +0.3647。
- 影响判断：这是典型“强风格提升伴随强内容漂移”的单点改动（高置信度）。

## 4) 总结：哪些改动在拉高风格，哪些在改善内容

- 明显拉高 style 的单点改动：`v5 -> v5-style_back` 的 distill 聚合策略（style +0.0857），但 lpips 同时显著上升。
- 更偏内容保持的改动：`v3 -> v4` 这类加大 cycle/edge 约束的迁移，通常降低 lpips，但可能牺牲 style/cls。
- 风格与识别双高但漂移大的代表：`overfit50-style-force-balance-v1`、`full_300_distill_low_only_v1`（二者 lpips 都偏高）。
- 目前更接近平衡点的路线：`overfit50-distill_low_only` 与 `overfit50-strok-style` 系（style>0.516、cls>=0.93，lpips 中低）。

## 5) 下一步计划（可执行）

### 5.1 目标与门槛

- 目标：在 `style >= 0.53` 前提下，把 `lpips` 压到 `<= 0.50`，同时 `cls >= 0.85`。
- 统一评测：固定 `eval_count=50`、完整矩阵、输出 `summary.json + summary_history.json`。

### 5.2 三阶段实验计划

1. 阶段A（复现实证，1-2天）
- 复现 `overfit50-distill_low_only` 与 `overfit50-strok-style`，冻结代码，只验证结果稳定性。
- 同时复现 `full_300_distill_low_only_v1`，确认其 high-style/high-lpips 是否稳定可重现。

2. 阶段B（单因素改动，2-3天）
- 在 `overfit50-distill_low_only` 基线上只扫以下单因子：`w_cycle`、`w_struct`、`w_stroke_gram`、`w_color_moment`。
- 每次只改一个参数组，记录 style/cls/lpips 的方向性，建立“参数-指标灵敏度表”。

3. 阶段C（组合收敛，2天）
- 选择阶段B里最优 2-3 组组合，做 50 epoch 完整评测。
- 只保留同时满足门槛（style/cls/lpips）的候选进入最终报告。

### 5.3 风险控制

- 禁止在同一轮同时改 `model.py + losses.py + 关键权重`，避免再次出现不可归因的耦合改动。
- 所有 run 强制保存首中末快照，确保后续可以做“基线->最终”的可追溯分析。
