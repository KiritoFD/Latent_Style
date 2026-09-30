# 620 4 天全量消融实验计划

## Summary

在已有的 `tools/ablation256`（217 组、1 epoch smoke）基础上，设计并实施一个 **512–1024 组、4 天、远程 RTX 3060** 的收敛级全量消融。核心目标是以 **Softmax Oversmoothing 理论** 为指导，系统评估 620 架构中每个模块、每个超参、每种 attention 变体对白化（WFI）、风格（CLIP-S）、内容（LPIPS）的真实贡献，最终输出可执行脚本、实验矩阵和结果汇总。

计划特点：
- **先探测 VRAM，再自动分配合适 batch**，避免 OOM 导致的时间浪费。
- **两阶段调度**：第一阶段 1 epoch smoke 快速筛选（约 1 天）；第二阶段对 promising 配置做 early-stopping 收敛训练（约 3 天）。
- **理论与工程结合**：除原有 A–M 维度外，新增基于 Softmax Oversmoothing 的 N–R 维度（temperature、sharpen、sinkhorn/gumbel、top-k、RMSNorm、style-bias、CFG 等）。
- **全自动化脚本**：配置生成、VRAM 探测、训练、评估、early stopping、结果收集一键运行。

## Current State Analysis

### 已有资产

1. **代码**：`src/blocks620.py`、`src/model620.py`、`src/config_schema.py` 已支持：
   - 6 种 `style_attn_mode`（softmax/gated/gated_raw/relu2/style_select/sparsemax）
   - `style_attn_temperature`、`style_attn_topk`
   - `endpoint_head_mode`（velocity / endpoint_lowhigh）+ `endpoint_film_enabled` + `endpoint_style_hidden_dim`
   - `style_film_enabled`、`style_shortcut_alpha`、`style_cross_attn_skip_coarse`
   - `style_condition_source`（latent / target_dino_patches）+ DINO adapter/MoE
   - 多种 loss weight、bridge 参数、数据参数
2. **历史消融**：`tools/ablation256/ablation256/configs/` 已有 217 组 1-epoch 配置，覆盖 A–M 维度。
3. **历史结论**：`docs/620/fog/ablation_audit/final_report.md` 指出：
   - `style_condition_source=latent` 通过 WFI 门，`target_dino_patches` 严重白化。
   - `edge_weight=0`、`swd_noise_sigma=0.02`、`endpoint_style_hidden_dim=128` 更优。
   - 容量升级（dim=128）当前基线上无收益。
4. **VRAM 探测**：`tools/experiments/make_vram_probe_cfg.py` + `run_phase616_vram_probe.sh` 可快速估算峰值显存。
5. **评估**：`tools/run_eval_with_wfi.py` 可输出 WFI/CLIP-S/LPIPS。

### 关键缺口

1. 当前 ablation256 是 **1 epoch smoke**，未按"收敛"训练，可能低估/高估某些维度。
2. 未系统探索 **Softmax Oversmoothing** 相关变量（temperature、sharpen scale、Sinkhorn/Gumbel hard、RMSNorm、attention entropy regularization）。
3. 未根据 VRAM 自动为每组配置选择 batch size，远程 3060 10.8GB 对不同架构（dim=64/128、depth=4/6）容量差异大。
4. 未实现 early stopping：4 天内无法让所有 1024 组都跑 8–10 epoch。

## Proposed Changes

### Phase 0: 历史与文档复盘（0.5 天，只读）

- 读取 `docs/620/fog/ablation_audit/` 全部报告，提取已验证结论和冲突点。
- 读取 `src/lancet_blocks.py` 中历史 attention 实现（`CrossAttnAdaGN`、`SemanticCrossAttn`、`_sinkhorn_attention`、`_gumbel_hard_attention`、RMSNorm 位置等）。
- 读取 `tools/ablation256/generate_ablation256.py` 和已有 `matrix.csv`。
- 输出：`docs/620/massive_ablation/history_digest_for_massive.md`，列出**必须复测**、**可直接继承**、**新增探索**的维度。

### Phase 1: 扩展消融矩阵至 512–1024 组

保留 A–M 组中经 Phase 0 判定为"必须复测"的维度，并新增 N–R 组。

#### 保留/精简后的 A–M 组（约 250 组）

| 组 | 维度 | 取值 | 组数 | 说明 |
|---|---|---|---|---|
| A | `style_attn_mode` × `style_cross_attn_gate_init` | 6 × 4 | 24 | 复测，但基线改为 latent + hd128 |
| B | `endpoint_head_mode` × `endpoint_film_enabled` × `endpoint_style_hidden_dim` × `endpoint_film_init_std` | 约 32 | 32 | 重点复测 hd64/128/256/512 + init 0/0.01/0.02/0.05 |
| C | `style_film_enabled` × `style_shortcut_alpha` | 2 × 3 | 6 | 新增 shortcut 0.0/0.5/1.0 |
| D | `single_step_swd_weight` × `single_step_edge_weight` × `swd_noise_sigma` | 4 × 2 × 2 | 16 | 去掉已验证有害的 edge=0.5 |
| E | `training_target_projection_mode` × `low_mode` × `low_anchor` | 3 × 3 × 2 | 18 | 保留 |
| F | `style_condition_source` × `style_dino_adapter_enabled` × `style_moe_enabled` | 2 × 2 × 2 | 8 | 验证 DINO 在收敛训练下是否仍有害 |
| G | `num_res_blocks` × `base_dim` | 3 × 2 | 6 | 保留，但自动降 batch |
| H | `learning_rate` × `batch_size` × `virtual_length_multiplier` | 3 × 2 × 3 | 18 | 保留，但 batch 由 VRAM 探针决定 |
| I | `w_attn_entropy_reg` × `endpoint_energy_band_weight` × `input_anchor_noise_std` | 3 × 2 × 2 | 12 | 保留 |
| J | `endpoint_high_scale` × `endpoint_velocity_floor` × `endpoint_lowpass_kernel` | 3 × 3 × 3 | 27 | 保留 |
| K | `style_attn_temperature` × `style_attn_topk` × `style_cross_attn_skip_coarse` | 3 × 3 × 2 | 18 | 保留 |
| L | `pairing_cache_topk` × `pairing_cache_cross_only` | 3 × 2 | 6 | 保留 |
| M | `bridge_sigma` × `t_sampling_power` × `t_min` | 3 × 2 × 2 | 12 | 保留 |

#### 新增 N–R 组：Softmax Oversmoothing 专题（约 300–500 组）

| 组 | 维度 | 取值 | 组数 | 理论依据 |
|---|---|---|---|---|
| N | `style_attn_mode` × `style_attn_temperature` × `style_attn_sharpen_scale` | 6 × 4 × 3 | 72 | 温度降低 / sharpen 提升可抑制平均化 |
| O | `style_attn_topk` × `style_attn_temperature` | 4 × 4 | 16 | top-k 直接减少参与平均的 token 数 |
| P | `semantic_attn_routing_mode` × `semantic_sinkhorn_iters` × `semantic_gumbel_tau` | 3 × 3 × 3 | 27 | 引入 Sinkhorn / Gumbel hard 路由 |
| Q | cross-attention 后归一化：`post_ca_norm`（RMSNorm/LayerNorm/None）× `post_ca_scale` | 3 × 3 | 9 | 恢复被 softmax 压缩的方差 |
| R | `style_bias_proj` 强度 × `film_init_std` × `pre_film_q` 开关 | 3 × 3 × 2 | 18 | 绕过 Q@K^T 瓶颈直接加 style bias |
| S | inference CFG：`cfg_target_scale` 扫描 | 5 | 5 | CFG 可解耦无条件灰白先验 |
| T | 组合优化：从 A–S 中选出 top-32 配置做 factorial 组合 | 32 | 32 | 验证交互效应 |

合计约 **520–750 组**。若时间允许，追加 U 组（历史 lancet_blocks 中的模块，如 `CrossAttnAdaGN`、`GWOTAttention`、`StyleRoutingSkip`）复测，可达 1024 组。

### Phase 2: VRAM 探针与 Batch 自适应

目标：远程 RTX 3060 显存保守控制在 **8–9GB**，避免 OOM。

1. **不逐个实验实际跑**：按架构特征聚类，同一类配置共享显存预算。关键特征：
   - `base_dim` = 64 / 128
   - `num_res_blocks` = 2 / 4 / 6
   - `endpoint_style_hidden_dim` = 64 / 128 / 256 / 512
   - `style_attn_mode` = softmax / gated / gated_raw / relu2 / style_select / sparsemax
   - `batch_size` 候选：8 / 16 / 24 / 32 / 48
2. 复用 `tools/experiments/make_vram_probe_cfg.py`，为**每个聚类**生成一个**只跑 20 步**的探针配置。每组聚类运行 20 step，记录 `peak_vram_gb`。
3. 探针脚本保存结果到 `docs/620/massive_ablation/vram_probe_results.csv`。
4. 根据探针结果，为每类实验配置**自动选择最大可用 batch**，规则：
   - peak < 8.0GB → 保持候选 batch
   - 8.0 ≤ peak < 9.0GB → 可接受，但 prefer 降一档
   - peak ≥ 9.5GB 或 OOM → 继续降档
   - 目标：最终训练峰值稳定在 **8–9GB**，最高不超过 9.5GB
   - 最小 batch 不低于 4
5. 输出：`tools/massive_ablation/batch_assignments.json`。

### Phase 3: 收敛训练与 Early Stopping 脚本

#### 脚本 1：配置生成

`tools/massive_ablation/generate_massive_ablation.py`
- 输入：Phase 1 的维度定义 + Phase 2 的 batch 分配。
- 输出：
  - `tools/massive_ablation/configs/*.json`（约 512–1024 个）
  - `tools/massive_ablation/matrix.csv`
  - `tools/massive_ablation/vram_probe_commands.sh`
  - `tools/massive_ablation/launch_smoke.sh`（1 epoch）
  - `tools/massive_ablation/launch_converge.sh`（early stopping）
  - `tools/massive_ablation/collect_results.sh`

#### 脚本 2：Smoke 阶段（约 1 天）

`tools/massive_ablation/launch_smoke.sh`
- 所有 512–1024 组跑 1 epoch。
- 每组记录：train time、peak VRAM、final loss、eval 指标（如果 `full_eval_each_epoch=True`）。
- 若显存不足，自动跳过并记录 FAILED。

#### 脚本 3：Promising 筛选

`tools/massive_ablation/select_promising.py`
- 从 smoke 结果中，按多目标打分选择 top-128 进入收敛阶段：
  - 必须满足 WFI < 0.45（放宽门）
  - 优先：WFI 低、CLIP-S 高、LPIPS 低、训练时间短
  - 同时保证覆盖每个维度的 best/worst，避免全部集中在同一区域
- 输出：`tools/massive_ablation/promising_list.json`

#### 脚本 4：收敛训练阶段（约 3 天）

`tools/massive_ablation/launch_converge.sh`
- 对 promising_list 中的配置，启动 early stopping 训练：
  - `num_epochs` 上限 10
  - `full_eval_stop_on_convergence=True`
  - `full_eval_convergence_patience=4`
  - `full_eval_convergence_flat_eps_style=0.003`
  - `full_eval_convergence_flat_eps_lpips=0.010`
  - `full_eval_each_epoch=True`
- 每完成一个 epoch 自动跑 WFI 评估。
- 保存所有 epoch 的 checkpoint，供后续分析。

### Phase 4：评估与结果收集

1. `tools/massive_ablation/collect_results.sh`
   - 遍历所有实验目录，读取每个 epoch 的 `full_eval/wfi_eval_report.json` 或 `metrics.csv`。
   - 汇总字段：name, group, epoch, WFI, CLIP-S, LPIPS, ΔWFI, train_time, peak_vram, best_epoch。
   - 输出：`docs/620/massive_ablation/results_all.csv`。
2. `tools/massive_ablation/analyze_dimension_effects.py`
   - 对每个维度做单因子效应分析（控制其他变量时的平均 ΔWFI）。
   - 输出每个维度的 best/worst 取值、Pareto 前沿。

### Phase 5：分析与报告

输出：`docs/620/massive_ablation/final_report.md`
- 执行摘要与最终推荐配置
- 512–1024 组实验矩阵说明
- VRAM 探针结果与 batch 分配策略
- Smoke 阶段 vs 收敛阶段指标对比
- 每个维度的独立效应与交互效应
- Softmax Oversmoothing 专题结论
- 与历史基线（Round 1、hd512、Phase 5 推荐配置）的并排对比
- 设计取舍：KEEP / RESTORE / REMOVE / NEED_MORE_DATA

## Assumptions & Decisions

| 决策 | 选择 | 理由 |
|---|---|---|
| 基线条件源 | `style_condition_source=latent` | 历史消融已证明 DINO patches 在当前架构下加剧白化 |
| 基线 endpoint | `endpoint_lowhigh` + `endpoint_film_enabled=true` + `hd=128` | Phase 5 推荐配置通过 WFI 门且更简 |
| 基线 edge loss | `single_step_edge_weight=0.0` | 历史证明 edge=0 三赢 |
| 基线 SWD | `single_step_swd_weight=8.0`，`swd_noise_sigma=0.02` | 历史最优 |
| batch 自适应 | 通过 VRAM 探针为每组选最大可用 batch | 远程 3060 显存有限，不同架构容量差异大 |
| 收敛策略 | early stopping，最多 10 epoch | 4 天预算内平衡真实收敛与覆盖度 |
| 两阶段 | smoke 筛选 + 收敛训练 | 避免在大量无效配置上浪费收敛时间 |
| promising 数量 | 128 | 4 天中 3 天用于收敛，单卡约可跑 120–150 组收敛实验 |
| 是否复测全部 217 组 ablation256 | 是，但基线更新为 latent/hd128 | 原 ablation256 使用 target_dino_patches + hd512，结论可能在新基线下变化 |
| 是否引入历史 lancet_blocks | 作为 U 组可选 | 若时间允许，复测 CrossAttnAdaGN / Sinkhorn / Gumbel 等历史模块 |

## Verification Steps

1. **脚本可生成性**：运行 `generate_massive_ablation.py --dry-run` 后，应输出预期配置数、组分布、无命名冲突。
2. **VRAM 探针可靠性**：随机选 5 组运行探针，确认 peak_vram 与后续真实训练峰值误差 < 10%。
3. **Smoke 完整性**：launch_smoke.sh 跑 10 组抽样，确认每组都有 `train.log` 和 `metrics.csv`。
4. **Early stopping 正确性**：选 2 组 promising 跑收敛，确认训练在指标 plateau 后 4 epoch 内停止。
5. **结果收集完整性**：collect_results.sh 跑完后，`results_all.csv` 行数 ≥ 实际完成实验数 × 平均 epoch 数，无重复行。
6. **报告准确性**：final_report.md 中每个设计取舍都有具体实验名和指标引用。

## 预期输出文件

- `tools/massive_ablation/generate_massive_ablation.py`
- `tools/massive_ablation/select_promising.py`
- `tools/massive_ablation/analyze_dimension_effects.py`
- `tools/massive_ablation/vram_probe_commands.sh`
- `tools/massive_ablation/launch_smoke.sh`
- `tools/massive_ablation/launch_converge.sh`
- `tools/massive_ablation/collect_results.sh`
- `tools/massive_ablation/configs/*.json`
- `tools/massive_ablation/matrix.csv`
- `tools/massive_ablation/batch_assignments.json`
- `tools/massive_ablation/promising_list.json`
- `docs/620/massive_ablation/history_digest_for_massive.md`
- `docs/620/massive_ablation/vram_probe_results.csv`
- `docs/620/massive_ablation/results_all.csv`
- `docs/620/massive_ablation/final_report.md`
