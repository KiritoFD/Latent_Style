# Experiments-Cycle 论文级手工深度报告（v2）

- 版本：v2（手工逐项复核版）
- 日期：2026-02-13
- 范围：`experiments-cycle/` 全量 53 个实验
- 核心目标：评估“代码/配置变更对风格效果（style/cls/lpips）的影响”，并给出下一步可执行研究计划
- 手工证据源：
- `docs/reports/experiments_cycle_records/*.md`（53 份单实验记录）
- `docs/reports/snapshot_baseline_records/*.md`（18 份快照基线到最终分析）
- `docs/experiments_cycle/data/runs_metrics.csv`
- `docs/experiments_cycle/data/history_rounds.csv`
- `docs/reports/data/snapshot_baseline_vs_last.json`

## 摘要

本次按“论文审稿”标准进行手工复盘，重点不在“跑了多少实验”，而在“哪些实验可以相信、可复现、可归因”。结论如下：

1. 53 个实验中，严格可比样本只有 25 个（Tier A）；其余 28 个要么 LPIPS=0（Tier B），要么缺失可解析指标（Tier C）。
2. 严格可比上限是 `full_300_distill_low_only_v1`（style=0.5515），但 LPIPS=0.6973，属于“高风格高漂移”，不是可直接落地的平衡解。
3. 严格集合里 style 与 LPIPS 高正相关（`corr=+0.9395`），当前体系仍存在“提风格靠牺牲内容保真”的强耦合。
4. 真正有代码变更并能做“基线->最终快照”归因的核心只有 7 个实验；其中 `overfit50-distill_low_only` 与 `full_300_gridfix_v2` 变更幅度最大。
5. 最有价值的平衡候选不是 top-style，而是 `overfit50-strok-style`、`overfit50-distill_low_only`、`full_300-map16+32` 这一类“style 可接受 + cls 可接受 + lpips 可控”的路线。
6. 下一步不应继续大规模混改，而应执行“单变量 + 固定评测口径 + 强制快照追踪”的协议化实验。

## 1. 审计口径与可比性

### 1.1 指标

- `best_transfer_clip_style`：风格强度，越高越好。
- `best_transfer_classifier_acc`：风格类别一致性，越高越好。
- `latest_transfer_content_lpips`：内容漂移，越低越好。
- `matrix_eval_count_mean`：评测样本口径（本批主要为 50，少量为 30）。
- `history_rounds`：多轮 full_eval 轨迹长度。

### 1.2 分层

- Tier A（严格可比）：style 可解析 + 完整 matrix + LPIPS>0。
- Tier B（部分可比）：style 可解析，但 LPIPS=0 或口径不足。
- Tier C（不可比）：无可解析 style/cls/lpips。

### 1.3 总量

- 总实验：53
- Tier A：25
- Tier B：10
- Tier C：18

## 2. 全局结果

### 2.1 严格集合（Tier A）统计

- 样本数：25
- style 阈值覆盖：
- `>=0.50`: 15/25
- `>=0.52`: 9/25
- `>=0.54`: 4/25
- `>=0.56`: 0/25
- 相关系数：
- `corr(style, lpips)=+0.939530`
- `corr(style, cls)=+0.731278`

### 2.2 平衡候选（兼顾 style/cls/lpips）

约束 `cls>=0.75` 且 `lpips<=0.60` 下，仅 4 个：

| run | path | style | cls | lpips |
|---|---|---:|---:|---:|
| overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 0.528368 | 0.850 | 0.551213 |
| overfit50-distill_low_only | `overfit50-distill_low_only` | 0.518469 | 0.930 | 0.545553 |
| overfit50-strok-style | `overfit50-strok-style` | 0.516445 | 0.930 | 0.386163 |
| full_300-map16+32 | `full_300-map16+32` | 0.509921 | 0.780 | 0.424217 |

## 3. 快照时间线（以最早快照为基线）

最早可追溯快照是 `experiments/overfit50-style-force-balance-v1` 的 `src_snapshot_20260209_215752`。按“只看每个实验最后一个快照”的要求，18 个有快照实验时间线如下。

基线参考值（最早快照对应实验最终指标）：
- style=0.549101
- cls=0.890000
- lpips=0.764001

| 序号 | baseline 快照 | run(path) | 快照数 | 首尾代码变化 | style/cls/lpips | 相对最早基线（d_style / d_cls / d_lpips） | 手工结论 |
|---:|---|---|---:|---|---|---|---|
| 1 | 20260209_215752 | `experiments/overfit50-style-force-balance-v1` | 2 | 仅 `run.py`(+17) | 0.5491 / 0.89 / 0.7640 | 0 / 0 / 0 | 早期高风格样本，漂移极高 |
| 2 | 20260209_232130 | `experiments/overfit50-style-force-balance-v1-cycle4` | 1 | 无 | 0.5219 / 0.52 / 0.6754 | -0.0272 / -0.37 / -0.0886 | 风格与分类双降，漂移仍高 |
| 3 | 20260210_001057 | `experiments/overfit50-style-distill-struct-v1-20` | 1 | 无 | 0.5479 / 0.97 / 0.6706 | -0.0012 / +0.08 / -0.0934 | 高风格高分类，但漂移仍高 |
| 4 | 20260210_004525 | `experiments/main-style-distill-struct-v1` | 1 | 无 | 无指标 | - | 指标缺失，无法纳入结论 |
| 5 | 20260210_012059 | `experiments/full_250_strong-style` | 4 | 仅 `run.py`(+62) | 0.0000 / 0.28 / 0.0000 | -0.5491 / -0.61 / -0.7640 | 评测口径异常，不可比 |
| 6 | 20260210_105103 | `experiments/overfit50-style-distill-struct-v2` | 1 | 无 | 0.5349 / 0.88 / 0.6454 | -0.0142 / -0.01 / -0.1186 | 高风格分支，漂移仍大 |
| 7 | 20260210_111235 | `overfit50-style-distill-struct-v2` | 1 | 无 | 0.5284 / 0.85 / 0.5512 | -0.0207 / -0.04 / -0.2128 | 较 v2(exp) 漂移明显回收 |
| 8 | 20260210_115542 | `overfit50-style-distill-struct-v3` | 2 | 无 | 0.5165 / 0.67 / 0.5105 | -0.0326 / -0.22 / -0.2535 | 向低漂移移动，但分类下降 |
| 9 | 20260210_121539 | `overfit50-style-distill-struct-v4` | 6 | `losses/trainer/run` 重改 | 0.4990 / 0.55 / 0.4337 | -0.0501 / -0.34 / -0.3303 | 结构回收明显，风格掉到 0.50 以下 |
| 10 | 20260210_141451 | `full_250_strong-style` | 1 | 无 | 0.0000 / 0.46 / 0.0000 | -0.5491 / -0.43 / -0.7640 | 评测口径异常，不可比 |
| 11 | 20260210_203005 | `overfit50-strong_structure` | 1 | 无 | 0.5346 / 0.12 / 0.5811 | -0.0145 / -0.77 / -0.1829 | 风格高但分类崩溃 |
| 12 | 20260210_211143 | `overfit50-style-distill-struct-v4-mse` | 3 | `losses/trainer` 中改 | 0.4975 / 0.64 / 0.4380 | -0.0516 / -0.25 / -0.3260 | 相比 v4，cls 回升，style 未回升 |
| 13 | 20260210_223503 | `overfit50-v5-mse-sharp` | 1 | 无 | 0.4410 / 0.08 / 0.2418 | -0.1081 / -0.81 / -0.5222 | 极低漂移，但风格/分类几乎失效 |
| 14 | 20260210_232152 | `overfit50-v5-mse-sharp-style_back` | 2 | 仅 `losses.py`(+14/-1) | 0.5268 / 1.00 / 0.6065 | -0.0224 / +0.11 / -0.1575 | 单点损失改动把风格拉回，但漂移激增 |
| 15 | 20260210_235141 | `overfit50-distill_low_only` | 2 | `model/losses/trainer/run` 大改 | 0.5185 / 0.93 / 0.5456 | -0.0306 / +0.04 / -0.2184 | 平衡候选之一，但变量耦合重 |
| 16 | 20260211_001416 | `full_300_distill_low_only_v1` | 8 | 无任何改动 | 0.5515 / 1.00 / 0.6973 | +0.0024 / +0.11 / -0.0667 | strict top1，但高漂移 |
| 17 | 20260211_103151 | `overfit50-strok-style` | 1 | 无 | 0.5164 / 0.93 / 0.3862 | -0.0327 / +0.04 / -0.3778 | 最均衡候选之一 |
| 18 | 20260211_111219 | `full_300_gridfix_v2` | 7 | `model/losses/trainer/run` 大改 | 0.4629 / 0.37 / 0.3242 | -0.0862 / -0.52 / -0.4398 | 明显偏内容保真，风格能力下滑 |

## 4. 7 个“代码有变化”实验的因果拆解

以下全部基于手工源码 diff（`baseline snapshot -> last snapshot`）与指标对照。

### 4.1 `overfit50-v5-mse-sharp -> overfit50-v5-mse-sharp-style_back`

- 代码变化：
- 仅 `losses.py` 变化，引入 `distill_low_only` 与 `distill_cross_domain_only` 的按样本聚合。
- `model.py`、`trainer.py`、`run.py` 未改，配置也未改。
- 指标变化：
- style：0.441041 -> 0.526751（+0.085710）
- cls：0.080000 -> 1.000000（+0.920000）
- lpips：0.241780 -> 0.606507（+0.364727）
- 结论：
- 这是最“干净”的单点损失改动证据：风格与分类同时大幅提升，但内容漂移显著恶化。

### 4.2 `overfit50-style-distill-struct-v3 -> v4`

- 代码变化：
- `losses.py` 增加 `_per_sample_alignment`，cycle/struct 支持 `l1/mse + lowpass_strength` 混合。
- `run.py` 增加 CPU 线程/worker/pin_memory 约束，训练资源策略改变。
- `trainer.py` 的累计与日志机制有明显改动。
- 配置核心变化：
- `w_cycle 3 -> 8`
- `w_nce 2 -> 3.5`
- `w_struct 3 -> 0.75`
- 新增 `w_edge=0.25`
- warmup/ramp 全部缩短。
- 指标变化：
- style：0.516456 -> 0.499004（-0.017452）
- cls：0.670000 -> 0.550000（-0.120000）
- lpips：0.510532 -> 0.433740（-0.076792）
- 结论：
- 这组改动明确“回收内容漂移”成功，但以 style 和 cls 下滑为代价。

### 4.3 `overfit50-style-distill-struct-v4 -> v4-mse`

- 代码变化：
- 配置不变。
- `losses.py` 新增 `cycle_edge_strength`、`w_delta_tv` 路径。
- `trainer.py` 新增 `summary_history.json` 聚合写入。
- 指标变化：
- style：0.499004 -> 0.497539（-0.001465）
- cls：0.550000 -> 0.640000（+0.090000）
- lpips：0.433740 -> 0.437998（+0.004258）
- 结论：
- 风格几乎不变，cls 小幅回升，lpips 小幅变差，属于“实现和评测增强型”改动，不是强性能跃迁。

### 4.4 `overfit50-strok-style -> overfit50-distill_low_only`

- 代码变化（后者）：
- `model.py` 重构为 `_predict_delta + integrate(step_size)` 路径，支持步长控制。
- `losses.py` 引入 stroke/color 分解、semigroup、style_spatial_tv、多项低频/跨域开关。
- `trainer.py/run.py` 同步新增 semigroup/stroke/color/delta_tv 的日志与 full_eval step_size 参数。
- 配置变化量：49 键（高耦合）。
- 指标变化（用相近路线对照）：
- style：0.516445 -> 0.518469（+0.002024）
- cls：0.930000 -> 0.930000（0）
- lpips：0.386163 -> 0.545553（+0.159390）
- 结论：
- 风格/分类基本守住，但内容漂移明显上升；该实验最大问题是“改动太多，因果不可拆”。

### 4.5 `full_300_distill_low_only_v1` 对比 `full_300_gridfix_v2`

- `full_300_distill_low_only_v1`：
- 8 快照首尾完全一致，配方稳定。
- 指标：0.551533 / 1.000000 / 0.697327（高风格高漂移）。
- `full_300_gridfix_v2`：
- 去掉 `w_distill/w_code/w_push/w_semigroup`（归零），teacher/code 路径大幅收缩。
- `losses.py` 移除 gram/moment/idt 分支，风格统计约束更窄。
- 指标：0.462860 / 0.370000 / 0.324233。
- 差值（gridfix-v2 - distill_low_only_v1）：
- style：-0.088673
- cls：-0.630000
- lpips：-0.373094
- 结论：
- 典型“内容保真换风格能力”路线，适合做低漂移参照，不适合作为高风格主线。

### 4.6 `experiments/overfit50-style-force-balance-v1`

- 代码变化：
- 仅 `run.py` 新增 CPU 线程设置入口。
- 指标：0.549101 / 0.890000 / 0.764001。
- 结论：
- 改动本身是工程侧，不是损失机理变更；该 run 的高风格更可能来自原有配方，而非线程控制逻辑。

### 4.7 `experiments/full_250_strong-style`

- 代码变化：
- 仅 `run.py`：新增 `cpu_env_threads`、`cpu_affinity` 等系统侧控制。
- 指标：0 / 0.28 / 0（不可比）。
- 结论：
- 在当前评测结果缺失情况下，无法建立“代码改动 -> 风格效果”结论。

## 5. History 轨迹稳定性

有 `summary_history` 的仅 8 个 run。真正有趋势信息价值的是下列 4 个：

### 5.1 `full_300-map16+32`（6 轮）

- style：0.5028 -> 0.5046 -> 0.5092 -> 0.5099 -> 0.5082 -> 0.5086
- cls：0.72 -> 0.76 -> 0.78 -> 0.76 -> 0.76 -> 0.76
- lpips：0.4063 -> 0.4197 -> 0.4318 -> 0.4275 -> 0.4235 -> 0.4242
- 解读：150~200 epoch 已接近上限，后期基本平台期。

### 5.2 `overfit50-strok-style`（2 轮）

- epoch20：style 0.5164 / cls 0.93 / lpips 0.5373
- epoch40：style 0.4852 / cls 0.52 / lpips 0.3862
- 解读：继续训练会掉 style/cls、降 lpips，存在明显早停窗口。

### 5.3 `overfit50-v5-mse-sharp`（2 轮）

- style 基本不变（0.4399 -> 0.4410）
- cls 持续低位（0.08 -> 0.08）
- lpips 基本稳定（0.2421 -> 0.2418）
- 解读：低漂移稳定，但风格能力无增长迹象。

### 5.4 `full_300_strong-style-v1`（2 轮）

- style 微增（0.4589 -> 0.4597）
- cls 微降（0.17 -> 0.16）
- lpips 微升（0.2910 -> 0.2955）
- 解读：改进幅度极小，趋势价值有限。

## 6. Tier A（25 个）逐实验评注

| 排名 | run | path | style | cls | lpips | 手工诊断 |
|---:|---|---|---:|---:|---:|---|
| 1 | full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | 0.551533 | 1.000 | 0.697327 | 风格上限样本；漂移高，不宜直接当生产基线 |
| 2 | overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | 0.549101 | 0.890 | 0.764001 | 高风格高漂移极端样本 |
| 3 | overfit50-style-distill-struct-v1-20 | `experiments/overfit50-style-distill-struct-v1-20` | 0.547901 | 0.970 | 0.670641 | 高风格高分类，但漂移仍高 |
| 4 | overfit50-style-force-schedule | `experiments/overfit50-style-force-schedule` | 0.543730 | 0.260 | 0.582351 | 仅 style 高，分类一致性不足 |
| 5 | overfit50-style-distill-struct-v2 | `experiments/overfit50-style-distill-struct-v2` | 0.534868 | 0.880 | 0.645370 | 高风格路线，漂移偏高 |
| 6 | overfit50-strong_structure | `overfit50-strong_structure` | 0.534638 | 0.120 | 0.581129 | 分类崩溃，风格可信度低 |
| 7 | overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 0.528368 | 0.850 | 0.551213 | 当前可行候选之一，但仍需压漂移 |
| 8 | overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | 0.526751 | 1.000 | 0.606507 | 单点 loss 改动高收益，高漂移代价 |
| 9 | overfit50-style-force-balance-v1-cycle4 | `experiments/overfit50-style-force-balance-v1-cycle4` | 0.521914 | 0.520 | 0.675383 | 风格尚可，分类与漂移双风险 |
| 10 | overfit50-distill_low_only | `overfit50-distill_low_only` | 0.518469 | 0.930 | 0.545553 | 平衡候选，需拆分变量验证因果 |
| 11 | overfit50-style-distill-struct-v3 | `overfit50-style-distill-struct-v3` | 0.516456 | 0.670 | 0.510532 | 中位基线，后续 v4 走向可回溯 |
| 12 | overfit50-strok-style | `overfit50-strok-style` | 0.516445 | 0.930 | 0.386163 | 全局最均衡候选之一，建议主线复现实验 |
| 13 | full_300-map16+32 | `full_300-map16+32` | 0.509921 | 0.780 | 0.424217 | full 体系最稳中位基线（history=6） |
| 14 | overfit50-upscale-balance-v2-re-0.1cycle | `experiments/overfit50-upscale-balance-v2-re-0.1cycle` | 0.506272 | 0.230 | 0.434735 | 分类不足，需先补 cls |
| 15 | overfit50-upscale-balance-v2 | `experiments/overfit50-upscale-balance-v2` | 0.504675 | 0.500 | 0.441309 | style 可用，cls 不达标 |
| 16 | overfit50-style-distill-struct-v4 | `overfit50-style-distill-struct-v4` | 0.499004 | 0.550 | 0.433740 | v3->v4 回收漂移成功，style 下穿 0.5 |
| 17 | overfit50-style-distill-struct-v4-mse | `overfit50-style-distill-struct-v4-mse` | 0.497539 | 0.640 | 0.437998 | cls 有回升，style 无实质抬升 |
| 18 | overfit50-upscale-styleid-v1 | `experiments/overfit50-upscale-styleid-v1` | 0.482200 | 0.450 | 0.366808 | 低漂移但风格不足 |
| 19 | full_strong_style | `full_strong_style` | 0.475696 | 0.110 | 0.325611 | 内容好，风格身份几乎不可用 |
| 20 | full_300_gridfix_v2 | `full_300_gridfix_v2` | 0.462860 | 0.370 | 0.324233 | 内容保真优先路线，不适合冲 style |
| 21 | full_300_strong-style-v1 | `full_300_strong-style-v1` | 0.459679 | 0.170 | 0.295542 | 风格分类能力弱 |
| 22 | overfit50-upscale-balance-v2-re | `experiments/overfit50-upscale-balance-v2-re` | 0.450307 | 0.100 | 0.265898 | 低漂移但风格/分类偏弱 |
| 23 | overfit50-upscale-struct | `experiments/overfit50-upscale-struct` | 0.444692 | 0.120 | 0.254958 | 结构保守，风格不足 |
| 24 | overfit50-style-force | `experiments/overfit50-style-force` | 0.443286 | 0.070 | 0.257564 | 风格身份失败样本 |
| 25 | overfit50-v5-mse-sharp | `overfit50-v5-mse-sharp` | 0.441041 | 0.080 | 0.241780 | strict 最低漂移之一，但风格能力最低档 |

## 7. Tier B（10 个）逐实验评注

共同问题：LPIPS=0 或矩阵口径异常，不能支持“内容保真”结论。

| run | path | style | cls | lpips | 手工诊断 |
|---|---|---:|---:|---:|---|
| overfit50-upscale | `experiments/overfit50-upscale` | 0.593255 | 0.930 | 0.000000 | style 最高但 LPIPS=0，不能纳入 strict 结论 |
| overfit50-no-idt | `experiments/overfit50-no-idt` | 0.478453 | 0.430 | 0.000000 | 部分指标可看，内容指标不可用 |
| adacut_overfit50-lightonly | `adacut_overfit50-lightonly` | 0.470728 | 0.320 | 0.000000 | adacut 家族可比性不足 |
| overfit50-80-10-0.5 | `experiments/overfit50-80-10-0.5` | 0.452397 | 0.320 | 0.000000 | 基线型探索，证据不完整 |
| overfit50 | `experiments/overfit50` | 0.449813 | 0.260 | 0.000000 | 早期试验，保留参考意义 |
| overfit50-upscale-styleid-v2 | `experiments/overfit50-upscale-styleid-v2` | 0.446567 | 0.080 | 0.000000 | 分类极弱，且 LPIPS 不可用 |
| 50-no-distill | `50-no-distill` | 0.441478 | 0.080 | 0.000000 | 去 distill 路线未给出有效风格收益 |
| adacut_overfit50 | `adacut_overfit50` | 0.441186 | 0.100 | 0.000000 | 家族末端样本，参考价值有限 |
| full_250_strong-style | `experiments/full_250_strong-style` | 0.000000 | 0.280 | 0.000000 | style/lpips 皆 0，疑似评测链路问题 |
| full_250_strong-style | `full_250_strong-style` | 0.000000 | 0.460 | 0.000000 | 与上同名不同路径，需统一命名与评测 |

## 8. Tier C（18 个）逐实验评注

共同问题：无可解析 style 指标，不能进入任何性能排序。

| run | path | 家族 | 主要缺失 |
|---|---|---|---|
| adacut | `adacut` | adacut | summary 指标不可解析 |
| adacut_overfit | `adacut_overfit` | adacut | 无完整 full_eval 指标 |
| adacut_overfit0 | `adacut_overfit0` | adacut | 指标链路缺失 |
| main-style-distill-struct-v1 | `experiments/main-style-distill-struct-v1` | experiments_misc | 仅单快照，无可解析指标 |
| full-300-3060-313 | `full-300-3060-313` | full-300 | train/full_eval 证据都不足 |
| full_300_strong-style-v2 | `full_300_strong-style-v2` | full_300 | 有 full_eval 但输出不可解析 |
| baseline_50e | `ablation50_repro_cwdfix/baseline_50e` | other | ablation 基线缺可比指标 |
| overfit50-clipstyle-probe-v1 | `overfit50-clipstyle-probe-v1` | overfit50 | probe 输出未入标准 summary |
| overfit50-clipstyle-probe-v2 | `overfit50-clipstyle-probe-v2` | overfit50 | 同上 |
| overfit50-clipstyle-probe-v3 | `overfit50-clipstyle-probe-v3` | overfit50 | 同上 |
| small-exp-...145326 | `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_145326` | small-exp | 无可解析 summary |
| small-exp-...150950 | `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150950` | small-exp | 无可解析 summary |
| small-exp-...151448 | `experiments/small-exp-overfit50_e13_hires6_spatialproto_v1-overfit50_e13_hires6_spatialproto_v1-bd128-dsp1-hp0p2-whf3p0-wprob0p8-wproto0p2-wcyc8p0-20260209_151448` | small-exp | 无可解析 summary |
| small-exp-...151947 | `experiments/small-exp-overfit50_e14_hires6_weakcls_v1-overfit50_e14_hires6_weakcls_v1-bd128-dsp1-hp0p18-whf3p4-wprob0p35-wproto0p2-wcyc8p0-20260209_151947` | small-exp | 无可解析 summary |
| small-exp-e15 | `experiments/small-exp-overfit50_e15_style_only_from_smoke` | small-exp | 无可解析 summary |
| small-exp-e16 | `experiments/small-exp-overfit50_e16_style_only_flowboost` | small-exp | 无可解析 summary |
| small-exp-e17 | `experiments/small-exp-overfit50_e17_style_forcepath` | small-exp | 无可解析 summary |
| small-exp-smoke-e12 | `experiments/small-exp-smoke_e12_skipfusion_v2-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150800` | small-exp | 无可解析 summary |

## 9. 综合讨论

### 9.1 风格提升与漂移冲突是真实存在的，不是偶然点

- 在 Tier A（25 个）里，style 与 lpips 的高正相关非常显著。
- 高 style 头部样本几乎都落在高 lpips 区间（0.58~0.76）。
- 这说明当前损失设计仍偏“风格注入优先”，内容约束强度不足。

### 9.2 可执行主线应该选择“平衡候选”，而非 style top1

建议主线候选顺序：
1. `overfit50-strok-style`
2. `overfit50-distill_low_only`
3. `full_300-map16+32`

这三条线在 style、cls、lpips 三维上更接近可工程化平衡。

### 9.3 当前最大短板是“可归因性”，不是“实验数量”

- 53 个实验足够多，但只有 7 个实验可做代码改动归因。
- 很多 run 同时改代码、改损失、改训练配置，导致结论混叠。
- Tier C 的 18 个实验几乎都因输出不可解析而无法进入比较。

## 10. 下一步计划（可直接执行）

### 10.1 协议先行（必须先做）

1. 固定评测协议：统一 `eval_count=50`，强制完整 matrix，强制输出 `summary.json + summary_history.json`。
2. 固定快照协议：每 run 至少 3 个快照（首/中/末），且必须含 `model.py/losses.py/trainer.py/run.py/config.json`。
3. 固定变更协议：单轮只允许一种变化类型（仅代码或仅配置），禁止混改。
4. 固定命名协议：消除重名 run（如两个 `full_250_strong-style`）造成的解释歧义。

### 10.2 Phase A：基线复现（3 条）

1. 复现 `overfit50-strok-style`，验证 history 早停窗口。
2. 复现 `overfit50-distill_low_only`，拆分其 49 个配置键中的高影响子集。
3. 复现 `full_300-map16+32`，作为 full 体系稳定对照。

### 10.3 Phase B：单变量扫描（在固定代码上）

- 仅在 `overfit50-distill_low_only` 最后快照代码上做：
- `w_cycle`: 0.2 / 0.3 / 0.5
- `w_struct`: 0.15 / 0.2 / 0.3
- `w_stroke_gram`: 40 / 60
- `w_color_moment`: 4 / 6
- `w_semigroup`: 0 / 0.03 / 0.05
- `w_delta_tv`: 0.001 / 0.002 / 0.003

### 10.4 Phase C：代码改动最小化验证（仅 3 条）

1. 复测 `v5->style_back` 的 distill 聚合改动，但配合更强内容约束，目标压 lpips。
2. 复测 `v3->v4` 的 cycle/struct 权重迁移，验证 cls 下滑是否可逆。
3. 在 `full_300_gridfix_v2` 上单独恢复 `w_distill` 或 `w_code`，检测 style/cls 恢复斜率。

### 10.5 继续/淘汰门槛

- 继续门槛：`style>=0.53` 且 `cls>=0.85` 且 `lpips<=0.50`
- 淘汰门槛：
- style 上升但 lpips 恶化超过 +0.06
- 或 cls 下降超过 -0.05

## 11. 结论

本批实验“数量足够，证据密度不足”。  
最重要的下一步不是继续堆 run，而是把实验协议化、可归因化。  
在现有证据下，建议以 `overfit50-strok-style` / `overfit50-distill_low_only` / `full_300-map16+32` 作为三条主线基线，进入严格单变量扫描阶段。

