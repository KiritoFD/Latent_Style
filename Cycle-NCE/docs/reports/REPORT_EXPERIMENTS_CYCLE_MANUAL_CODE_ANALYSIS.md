# Experiments-Cycle 手工代码改动影响分析报告（中文）

- 分析日期：2026-02-13
- 分析方式：手工逐个对比 `最早快照 -> 最后快照`，直接阅读 `model.py / losses.py / trainer.py / run.py / config.json` 的代码和配置差异，再对照指标。
- 指标口径：`best_transfer_clip_style`、`best_transfer_classifier_acc`、`latest_transfer_content_lpips`、`matrix_eval_count_mean`、`history_rounds`。
- 说明：以下结论基于历史实验结果，绝大部分不是严格单变量 A/B，因果强度会在文中标注。

## 1) 手工核查范围（有代码改动的 7 个实验）

| run | rel_path | 代码改动文件 | 配置改动 | style | cls | lpips | eval_count | history |
|---|---|---|---:|---:|---:|---:|---:|---:|
| full_300_gridfix_v2 | `full_300_gridfix_v2` | model/losses/trainer/run | 7 | 0.462860 | 0.370000 | 0.324233 | 30 | 1 |
| overfit50-style-distill-struct-v4 | `overfit50-style-distill-struct-v4` | losses/trainer/run | 12 | 0.499004 | 0.550000 | 0.433740 | 50 | 0 |
| overfit50-style-distill-struct-v4-mse | `overfit50-style-distill-struct-v4-mse` | losses/trainer | 0 | 0.497539 | 0.640000 | 0.437998 | 50 | 0 |
| overfit50-distill_low_only | `overfit50-distill_low_only` | model/losses/trainer/run | 49 | 0.518469 | 0.930000 | 0.545553 | 50 | 1 |
| overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | losses | 0 | 0.526751 | 1.000000 | 0.606507 | 50 | 1 |
| overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | run | 0 | 0.549101 | 0.890000 | 0.764001 | 50 | 0 |
| full_250_strong-style | `experiments/full_250_strong-style` | run | 训练项有变化 | 0.000000 | 0.280000 | 0.000000 | 50 | 0 |

## 2) 逐实验手工分析

### 2.1 full_300_gridfix_v2

- 手工看到的核心改动：
- `config.json` 把 `w_distill: 0.7 -> 0.0`、`w_code: 6.0 -> 0.0`、`w_push: 1.0 -> 0.0`、`w_semigroup: 0.04 -> 0.0`，同时去掉 `style_loss_source`。
- `losses.py` 中 teacher 分支从“总是参与”改成“仅当 `w_distill` 或 `w_code` > 0 才参与”，并移除 gram/moment 路径与 idt 相关输出。
- `model.py` 增加 `build_model_from_config` 工厂，`trainer.py` 改为通过该工厂建模，日志也去掉 gram/moment/idt。
- `training` 还改了 `batch_size`、`log_interval`，并把 `full_eval_interval` 设为 0、`full_eval_on_last_epoch` 设为 false。
- 指标解读：
- 结果是 style=0.4629、cls=0.37、lpips=0.3242。
- 这组结果明显偏“内容保留”，风格能力和分类一致性明显弱。
- 影响判断：
- 高置信度：去 teacher/code/push 后，style 与 cls 下滑，lpips 下降。
- 高风险点：最后配置关闭了 full_eval，最终指标可能更多反映“早期评测点”，不是最后训练状态。

### 2.2 overfit50-style-distill-struct-v4

- 手工看到的核心改动：
- `config.json` 权重明显换挡：`w_cycle 3->8`、`w_nce 2->3.5`、`w_struct 3->0.75`、新增 `w_edge=0.25`，warmup/ramp 也整体变短。
- `losses.py` 新增 `_per_sample_alignment`，允许 cycle/struct 用 `l1/mse + lowpass_strength` 混合。
- `trainer.py` 和 `run.py` 增加了偏 CPU 低负载的训练控制（线程、worker、pin_memory 逻辑调整）。
- 指标解读：
- v4 自身：style=0.4990、cls=0.55、lpips=0.4337。
- 与 v3（0.5165/0.67/0.5105）相比：style -0.0175，cls -0.12，lpips -0.0768。
- 影响判断：
- 中高置信度：这次从 struct 主导转向 cycle/nce 主导的改动，换来了更低 lpips，但 style 和 cls 都下降。

### 2.3 overfit50-style-distill-struct-v4-mse

- 手工看到的核心改动：
- `config.json` 首尾一致，没有参数变化。
- `losses.py` 增加 `cycle_edge_strength`、`w_delta_tv` 相关项。
- `trainer.py` 增加 `delta_tv` 记录，并补了 `summary_history.json` 聚合写入逻辑。
- 指标解读：
- v4-mse：0.4975 / 0.64 / 0.4380。
- 对比 v4：style -0.0015、cls +0.09、lpips +0.0043。
- 影响判断：
- 中置信度：实现修补更像“提升分类稳定性”的小改动，对 style 上限基本无帮助。

### 2.4 overfit50-distill_low_only

- 手工看到的核心改动（改动最大）：
- `model.py` 从单次 forward 逻辑扩展到 `_predict_delta + integrate(step_size)` 路径，结构组织明显重构。
- `losses.py` 新增 stroke/color 多尺度分解、semigroup、style_spatial_tv、distill_low_only/cross_domain_only 等多条监督路径。
- `trainer.py` 和 `run.py` 同步扩展日志与 full_eval 参数（含 `step_size`）。
- `config.json` 变动 49 项，不仅 loss/model 改动大，训练与评测配置也大变（batch、workers、compile、test_image_dir、lpips 开关、save_dir 等）。
- 指标解读：
- 本 run：0.5185 / 0.93 / 0.5456。
- 与 `overfit50-strok-style`（0.5164 / 0.93 / 0.3862）相比：style +0.0020、cls 持平、lpips +0.1594。
- 影响判断：
- 中置信度：它确实提升到“style+cls 都不错”的区间，但内容漂移仍较高。
- 归因警告：配置层混入了大量非损失因素，不能把收益完全归因到某一个 loss 改动。

### 2.5 overfit50-v5-mse-sharp-style_back

- 手工看到的核心改动：
- 只动了 `losses.py`：distill 从统一 L1 变为支持 `distill_low_only` 与 `distill_cross_domain_only` 的按样本聚合。
- `config.json`、`model.py`、`trainer.py`、`run.py` 都没有变化。
- 指标解读：
- 本 run：0.5268 / 1.00 / 0.6065。
- 对比 `overfit50-v5-mse-sharp`（0.4410 / 0.08 / 0.2418）：style +0.0857、cls +0.92、lpips +0.3647。
- 影响判断：
- 高置信度：这是“显著提升风格和分类，但显著增大内容漂移”的典型单点损失改动。

### 2.6 experiments/overfit50-style-force-balance-v1

- 手工看到的核心改动：
- 只有 `run.py` 增加 `_set_cpu_threads` 并在入口调用，属于训练资源控制。
- 其余核心算法文件不变，配置也不变。
- 指标解读：
- 本 run 指标 0.5491 / 0.89 / 0.7640，style 高但 lpips 极高。
- 对比 `...cycle4`（0.5219 / 0.52 / 0.6754）显示后续版本整体退化。
- 影响判断：
- 中置信度：run.py 的工程改动不是主要风格增益来源，主要影响训练稳定性和吞吐，不是损失机理。

### 2.7 experiments/full_250_strong-style

- 手工看到的核心改动：
- `run.py` 增加 `cpu_env_threads`、`cpu_affinity` 处理，线程/亲和性控制更细。
- `config.json` 训练项也有变（batch、compile、CPU 线程参数），但 model/loss/trainer 没变。
- 指标解读：
- style=0、lpips=0，属于非可比口径。
- 影响判断：
- 高置信度：在当前指标形态下，无法评价“代码改动对风格效果”的真实影响。

## 3) 综合实验结果（从“代码改动 -> 风格效果”角度）

- 可以确认“会明显拉高风格”的改动：
- `v5 -> v5-style_back` 这类 distill 聚合改动，确实显著抬升 style 与 cls，但 lpips 同步恶化。
- 可以确认“更偏内容保持”的改动：
- `v3 -> v4` 这类加强 cycle/edge、削弱 struct 的迁移，使 lpips 下降，但 style/cls 同时下降。
- 当前较均衡候选：
- `overfit50-distill_low_only` 与 `overfit50-strok-style` 这条线在 style/cls 上较稳，但 lpips 仍有优化空间。
- 当前最重要的归因问题：
- 多个 run 同时改了“代码 + 损失权重 + 训练/评测配置”，导致结论容易混叠，必须做单变量回放。

## 4) 下一步计划（按重要性排序）

### 4.1 先做“可归因”实验设计

1. 固定代码快照：每轮只允许改配置，不改 `model.py/losses.py/trainer.py/run.py`。  
2. 固定评测口径：`eval_count=50`、完整矩阵、强制输出 `summary.json + summary_history.json`。  
3. 固定数据与训练基础项：`test_image_dir`、`num_workers`、`batch_size`、`use_compile` 不在同轮变更。  

### 4.2 主线实验（建议从 overfit50 系开始）

1. 基线 B1：`overfit50-distill_low_only`（固定最后快照代码）。  
2. 基线 B2：`overfit50-strok-style`（同评测口径复现）。  
3. 单变量扫描 S1：`w_cycle` 只扫 3 个点（0.2 / 0.3 / 0.5）。  
4. 单变量扫描 S2：`w_struct` 只扫 3 个点（0.15 / 0.2 / 0.3）。  
5. 单变量扫描 S3：`w_stroke_gram` 与 `w_color_moment` 只做 2x2 组合。  

### 4.3 通过门槛（继续/淘汰标准）

- 继续条件：`style >= 0.53` 且 `cls >= 0.85` 且 `lpips <= 0.50`。  
- 淘汰条件：style 提升但 lpips 恶化超过 `+0.06`，或 cls 下降超过 `-0.05`。  

### 4.4 full_300 系策略

1. 先冻结 `full_300_distill_low_only_v1` 作为“高 style 参考点”。  
2. 只做小范围内容回收实验（不改 teacher/code 主干），目标是压 lpips，不追 style 再冲高。  
3. 禁止复制 `full_300_gridfix_v2` 那种同时关掉多条监督路径的激进改动。  

## 5) 结论一句话

- 目前最值得推进的不是“继续大改代码”，而是基于已验证路线做严格单变量实验，把 `style` 保住的同时把 `lpips` 压下来。  
