# 620 全面消融实验与设计审计 Spec

## Why
620 模型已积累大量相互叠加的设计变体（round1 诊断、白化修复、attention 改造、endpoint-FiLM、H4–H7 等）。Git 历史与近期初步消融显示，部分历史结论在新基线下可能不再成立：例如 `endpoint_film_hd512` 并非全局最优，`velocity` head 与 `softmax` attention 反而可能更优。为避免在不稳定的设计负债上继续堆叠，必须系统对比 git 历史实现 + 当前模型消融 + 历史基线，确定每个模块去留。

## What Changes
- 完成对所有相关 git 分支（当前分支、`codex/tokenizer-clean-c3058eab`、`origin/attn`、`origin/SWD`、`origin/Style8_Moment+SWD` 等）的历史实现与结果摘要。
- 在本地 RTX 4070 上以统一口径完成 620 消融矩阵（核心维度 + 扩展维度）。
- 综合 git 历史结论与当前消融结果，产出设计取舍报告：每个模块标记 KEEP / RESTORE / REMOVE / NEED_MORE_DATA。
- 生成最小有效配置 `configs/620_spatial_bridge_ablation_recommended.json`。
- 产出完整详尽报告 `docs/620/fog/ablation_audit/final_report.md`。
- **BREAKING**: 可能推翻此前 `620-whitening-fix` 中确立的 `endpoint_film_hd512` 为最优的结论，并替换为更简配置。

## Impact
- Affected specs: `620-whitening-fix`
- Affected code: `src/blocks620.py`, `src/model620.py`, `src/config_schema.py`, `src/losses620.py`, `src/style_encoder620.py`, `configs/620_*.json`
- Affected docs: `docs/620/fog/ablation_audit/`

## ADDED Requirements

### Requirement: 完成 Git 历史调研
系统 SHALL 完整探索当前分支、`codex/tokenizer-clean-c3058eab`、`origin/attn`、`origin/SWD`、`origin/Style8_Moment+SWD`、`origin/re-SWD`、`origin/multistep-texture`、`origin/exp/style-injection-priority-proto-sep` 等分支的提交历史、代码差异与文档结论。

#### Scenario: 历史摘要完成
- **WHEN** 开始消融审计
- **THEN** 必须输出 `docs/620/fog/ablation_audit/git_history_digest.md`
- **THEN** 文档必须包含：分支地图、实验时间线、设计演进树、历史教训、对当前消融的建议

### Requirement: 建立 620 设计维度消融矩阵
系统 SHALL 为 620 的主要设计选择建立可独立开关的消融维度，并在本地以统一口径运行 smoke 实验。

#### Scenario: 消融矩阵覆盖完整
- **WHEN** 开始 620 消融审计
- **THEN** 必须覆盖以下核心维度：
  - `style_attn_mode`: gated / softmax / gated_raw / relu2 / style_select / sparsemax
  - `style_film_enabled`: true / false
  - `endpoint_head_mode`: velocity / endpoint_lowhigh
  - `endpoint_film_enabled`: true / false
  - `endpoint_style_hidden_dim`: 128 / 256 / 512
  - `style_cross_attn_gate_init`: 0.05 / 0.3 / 0.5
  - `base_dim`: 64 / 128
  - `num_res_blocks`: 4 / 6
  - `single_step_swd_weight`: 0 / 2 / 8 / 16
  - `swd_noise_sigma`: 0 / 0.02
  - `single_step_edge_weight`: 0 / 0.1
  - `style_condition_source`: latent / target_dino_patches

### Requirement: 以统一口径对比历史基线
系统 SHALL 将每次消融结果与以下历史基线并排对比：
- Round 1 历史基线 `base_swd8`（dim=64×4, SWD=8, 8 epoch）
- 白化修复前基线 `620_film_v5_gated_local_smoke`
- 此前自评最优 `endpoint_film_hd512`
- Phase 2 新发现候选最优（如 `velocity` head + `gate_init=0.05`）

#### Scenario: 基线对比表生成
- **WHEN** 每次消融实验完成
- **THEN** 必须输出与基线并排的 WFI、CLIP-S、content LPIPS、ΔWFI、训练时间、参数量

### Requirement: 产出设计取舍报告
系统 SHALL 综合 git 历史 + 当前消融结果，为每个设计维度给出明确建议。

#### Scenario: 设计取舍决策
- **WHEN** 消融矩阵完成
- **THEN** 对每个维度必须给出：
  - 独立贡献与交互影响
  - 建议动作：KEEP / RESTORE / REMOVE / NEED_MORE_DATA
  - 证据引用、置信度、与 git 历史结论的对照

### Requirement: 固化最小有效配置并验证
系统 SHALL 将消融结论固化为新的 formal 配置模板，并验证其指标不低于此前最优。

#### Scenario: 新 formal 配置生成
- **WHEN** 设计取舍报告完成
- **THEN** 必须生成 `configs/620_spatial_bridge_ablation_recommended.json`
- **THEN** 必须训练 1 epoch smoke 并跑 WFI 评估
- **THEN** 必须确认 WFI < 0.40 且 CLIP-S ≥ 0.695

## MODIFIED Requirements

### Requirement: 620 后续优化必须基于消融结论
620 的后续优化（text、cross-attn 大改、DINO 扩展等）不再默认在现有 full-feature 分支上进行，而是基于消融后的最小有效配置。

**Reason**: 避免在无效模块上继续堆叠复杂度。
**Migration**: 所有后续实验以 `configs/620_spatial_bridge_ablation_recommended.json` 为基线。

## REMOVED Requirements

### Requirement: 所有既有设计模块默认保留
**Reason**: 历史设计（如 gated_raw attention、hf_residual、3-epoch over-train、endpoint_film_hd512）已被新的消融证据证明并非最优，不应默认保留。
**Migration**: 通过消融报告明确每个模块的去留，并在配置模板中删除 REMOVE 项。
