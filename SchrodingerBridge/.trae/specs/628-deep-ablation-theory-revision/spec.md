# 628 深度消融理论修正 Spec (v2 扩充版)

## Why

628 已完成的消融（I1-I10 推理 + T1-T8 训练 smoke）初步证明当前架构处于 Pareto 前沿，但 v1 spec 的 40 组破坏性消融存在三大盲区：

1. **损失消融不完整**：L1-L12 遗漏了 `w_flow`（FM 主导命题的核心！）、`coupling_structure_edge_weight`、`coupling_structure_hybrid_stats_weight` 等默认启用的损失项；且未探索 28 个默认关闭但可能有效的损失项（`w_contrast_preserve`、`w_hf_energy`、`w_anisotropic_kinetic`、`w_freq_split_cosine` 等）
2. **权重扫描过浅**：P1-P6 每个参数只有 3 档（含 0），缺少中间档和极端档，无法绘制完整的"参数-指标"敏感性曲线，无法定位饱和点和最优点
3. **架构开关未充分探索**：D1-D18 只覆盖静态组件，未覆盖关键 mode 切换（`style_attn_mode`、`endpoint_head_mode`、`transport_prediction_mode`、`kinetic_penalty_mode`、`bridge_path_mode`、`swd_distance_mode`、`t_sampling_mode` 等）和未启用开关（`endpoint_film_enabled`、`velocity_hf_residual_enabled`、`spectral_brownian_enabled` 等）

本次目标：**通过约 114 组完整的轻度+破坏性消融矩阵，定量探明每个损失项/架构组件/参数对 clip/lpips 的边际贡献与饱和点，进而修正数学理论**。

## What Changes

### 新增工作（不修改生产代码）

#### 1. D 类架构消融：18 → 30 组（+12）

**保留 D1-D18**（已有配置生成），新增：
- D19-D22：`style_attn_mode` 切换（softmax → gated_raw / relu2 / style_select / sparsemax）
- D23：`endpoint_head_mode` 切换（velocity → endpoint_lowhigh）
- D24：`transport_prediction_mode` 切换（velocity → endpoint）— 训练侧验证 XPred 命题
- D25：`training_target_projection_mode` 切换（legacy → dwt）
- D26：`kinetic_penalty_mode` 切换（global_l2 → per_band）
- D27：`terminal_swd_mode` 切换（standard → high_freq）
- D28：`bridge_path_mode` 切换（vertical → tri_band）
- D29：`swd_distance_mode` 切换（cdf → squared）
- D30：`t_sampling_mode` 切换（uniform_power → logit_normal）

#### 2. L 类损失消融：12 → 36 组（+24）

**保留 L1-L12**，新增关闭类（L13-L16）：
- L13：`w_flow=0`（**关键！验证 FM 主导命题**）
- L14：`coupling_structure_edge_weight=0`（默认 1.0，未消融）
- L15：`coupling_structure_hybrid_stats_weight=0`（默认 0.5，未消融）
- L16：`source_endpoint_aux_weight` + `endpoint_energy_band_weight` 联合关闭

**新增启用探索类（E1-E24）**— 探索 28 个默认关闭损失项的启用效果：
- **内容保真类**：E1 `w_contrast_preserve=1.0`、E2 `w_channel_variance=1.0`、E3 `w_hf_energy=1.0`、E4 `w_content_lowpass_anchor=1.0`、E5 `w_content_edge_anchor=1.0`、E6 `w_pixel_color_match=1.0`
- **风格强化类**：E7 `w_velocity_magnitude=1.0`、E8 `w_residual_style_direction=1.0`、E9 `w_style_contrastive=1.0`、E10 `w_style_energy_floor=1.0`、E11 `w_hsv_saturation=1.0`、E12 `w_output_variance=1.0`
- **方向约束类**：E13 `w_directional_cosine=1.0`、E14 `w_freq_split_cosine=1.0`、E15 `w_endpoint_velocity_reg=1.0`、E16 `w_spectral_amplitude=1.0`
- **物理约束类**：E17 `w_anisotropic_kinetic=1.0`、E18 `w_stokes_viscous=1.0`、E19 `w_curvature=1.0`、E20 `w_lowfreq_velocity=1.0`
- **正则与蒸馏类**：E21 `w_attn_entropy_reg=0.5`、E22 `w_style_strength_reg=0.5`、E23 `w_variance_penalty=1.0`、E24 `w_plain_path_distill=1.0`

#### 3. P 类参数扫描：18 → 36 组（+18）

**保留 P1-P6**（已有 18 组），新增权重扫描：
- **P7 `spectral_w_hh` 扫描**：0.5 / 1.0 / 3.0 / 6.0（验证高频权重饱和点）
- **P8 `spectral_w_ll` 扫描**：0.1 / 0.5 / 1.0 / 2.0（验证低频权重对 LPIPS 的影响）
- **P9 `terminal_swd_weight` 扫描**：0.05 / 0.5 / 1.0 / 2.0（验证 SWD 权重饱和点）
- **P10 `w_kinetic` 扫描**：0.5 / 2.0 / 4.0 / 8.0（验证 kinetic 权重饱和点）
- **P11 `bridge_sigma` 扫描**：0.0 / 0.05 / 0.08 / 0.1（验证 σ 魔法阈值，训练侧）
- **P12 `single_step_edge_weight` 扫描**：0.05 / 0.5 / 1.0 / 2.0（验证 edge 权重饱和点）
- **P13 `w_flow` 扫描**：0.1 / 0.3 / 0.5 / 2.0（**关键！验证 FM 权重降低能否突破天花板**）
- **P14 `w_endpoint_content` 扫描**：0.5 / 2.0 / 4.0 / 8.0（验证内容损失权重饱和点）
- **P15 `coupling_structure_cost_weight` 扫描**：0.5 / 2.0 / 4.0 / 8.0（验证 OT 结构权重饱和点）
- **P16 `style_attn_num_tokens` 扫描**：64 / 128 / 512 / 1024（验证 token 数上限，训练侧）
- **P17 `style_attn_sharpen_scale` 扫描**：0 / 2.5 / 5.0 / 10.0（验证 sharpen 饱和点，训练侧）
- **P18 `style_cross_attn_gate_init` 扩展**：0.1 / 0.5 / 1.0（扩展 P6，验证大 gate 效果）

#### 4. 推理消融补全：8 → 12 组（+4）

**保留推理消融 #1-#8**，新增：
- #9：`bridge_path_mode` 推理切换（vertical → tri_band）
- #10：`swd_distance_mode` 推理切换（cdf → squared）— 推理侧验证
- #11：`full_eval_num_steps` 扫描（4 / 8 / 16 / 32）— 验证 ODE 步数饱和点
- #12：`full_eval_style_strength` 扫射（0.5 / 1.0 / 1.5 / 2.0）— 验证风格强度饱和点

#### 5. 理论修正

基于 114 组消融结果：
- 校准"五层乘积保守机制"各因子（α_gate / α_attn / α_norm / α_init / α_proj）
- 验证/推翻 6 个数学命题
- 绘制完整 Pareto 前沿（114 新点 + 历史点）
- 更新 docs/628/ablation_conclusions.md + docs/622/history/10_unified_mathematical_model.md
- 给出 Phase 5 理论指引

### 不变项
- 不修改 src/ 下任何生产代码
- 不重训 T5 baseline（T5 ep7 作为所有消融的固定起点）
- 不更换数据集或评估管线
- 不创建新 docs/628 文档（结果汇总进 `docs/628/ablation_conclusions.md` 现有文件）

## Impact

- **Affected code**: `628_gen_destructive_configs.py`（扩充 ABLATIONS 列表）+ `configs/ablations/628_destructive/`（输出）+ `exp/628_ablation/destructive/`（结果）
- **Affected docs**: `docs/628/ablation_conclusions.md`（追加 Phase 3-7）+ `docs/622/history/10_unified_mathematical_model.md`（追加第 7 章）
- **VRAM 约束**: batch=16, bf16, T5 峰值 8.9 GB（远低于 12 GB 上限）
- **时间预算**: 114 组 × (3 epoch 训练 ~6 min + 评估 ~2 min) ≈ 910 min ≈ 15 h
  - 分 3 批执行，每批 ~5 h，可跨 2 天完成
  - 推理消融 12 组不占训练 GPU，可并行

## ADDED Requirements

### Requirement: D 类架构消融扩充（D19-D30）

系统 SHALL 执行 12 组新增架构 mode 切换消融，每组从 T5 ep7 续训 3 epoch。

#### Scenario: mode 切换消融
- **WHEN** 执行 D19-D30 中任一配置
- **THEN** 在 T5 ep7 基础上切换对应 mode，续训 3 epoch
- **AND** 与 T5 ep10 baseline 对比 Δclip / Δlpips
- **AND** 记录训练稳定性（loss 曲线、是否发散）

### Requirement: L 类损失消融扩充（L13-L16 + E1-E24）

系统 SHALL 执行 4 组新增关闭消融（L13-L16）和 24 组启用探索消融（E1-E24）。

#### Scenario: 损失关闭消融
- **WHEN** 执行 L13（w_flow=0）
- **THEN** 验证 "FM 主导条件" 命题 — 若 clip 不变，证明 FM loss 确实主导 style 梯度
- **AND** 记录训练是否发散（FM 是主稳定器，关闭可能不稳定）

#### Scenario: 损失启用探索
- **WHEN** 执行 E1-E24 中任一配置
- **THEN** 在 T5 ep7 基础上启用对应损失项（权重=1.0 或 0.5）
- **AND** 与 T5 ep10 baseline 对比 Δclip / Δlpips
- **AND** 标注：✅ 提升 / ⚠️ 无显著变化 / ❌ 退化

### Requirement: P 类参数扫描扩充（P7-P18）

系统 SHALL 执行 18 组新增参数权重扫描，每个参数 4 档（含极端值）。

#### Scenario: 权重饱和点定位
- **WHEN** 执行 P7-P18 中任一参数扫描
- **THEN** 绘制参数值 vs (clip, lpips) 曲线
- **AND** 标注饱和点（继续增大无收益）和最优点（Pareto 最佳）
- **AND** P13（w_flow 扫描）验证 "降低 FM 权重能否突破 clip 天花板" 假设

### Requirement: 推理消融补全扩充（#9-#12）

系统 SHALL 执行 4 组新增推理消融。

#### Scenario: 推理参数扫描
- **WHEN** 执行推理消融 #11（num_steps 扫描）
- **THEN** 在 T5 ep7 上不重训，仅切换推理步数
- **AND** 绘制步数 vs (clip, lpips, 推理时间) 曲线
- **AND** 验证 ODE 步数饱和点（预期 8→16 边际递减，16→32 无显著变化）

### Requirement: 理论修正与命题验证

系统 SHALL 基于全部 114 组消融结果定量验证或推翻 6 个数学命题：

1. **命题 1（Gate Collapse 必然性）**: D10 + P6 + P18 + E21 验证
2. **命题 2（GN 白化定理）**: D11/D12 + E1/E2 验证
3. **命题 3（SWD 梯度正交性）**: L3/L4/L11 + P9 + E24 验证
4. **命题 4（训练-输出不匹配）**: D24 + 推理消融 #1 验证
5. **命题 5（有效 style 维度极低）**: P6 + P16 + P18 + 推理消融 #7 验证
6. **命题 6（三难困境）**: 全部 114 组结果汇总验证 Pareto 前沿形状
7. **命题 7（FM 主导条件，新增）**: L13 + P13 + E20 验证 — 降低/移除 FM 权重能否打破保守吸引子

## MODIFIED Requirements

### Requirement: 628 消融结论文档扩展

`docs/628/ablation_conclusions.md` SHALL 追加：

- **Phase 3: Destructive Ablation (D1-D30)** — 30 组架构组件必要性矩阵
- **Phase 4: Loss Ablation (L1-L16 + E1-E24)** — 40 组损失项开关/启用矩阵
- **Phase 5: Parameter Sweep (P1-P18)** — 18 参数 × 多档位敏感性曲线
- **Phase 6: Inference Ablation Supplement (#1-#12)** — 12 组推理消融
- **Phase 7: Theory Revision** — 五层乘积模型校准 + 7 命题验证表 + Pareto 前沿
