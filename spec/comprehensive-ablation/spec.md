# Feature Specification: 全面消融机制实验与确定性结论文档

**Created**: 2026-06-27
**Status**: Draft
**Input**: 用户要求："先建立全面消融机制并且实施实验，产出一个文档确定性的告诉我每个机制，乃至组合起来，会有什么效果。"

## Overview

对 Schrödinger Bridge 风格迁移模型（620_spectral_ode + FM 架构）进行系统性消融实验，穷举所有独立机制及其关键组合，产出一份确定性结论文档，回答：**每个机制单独启用/禁用时的效果是什么？哪几个机制组合起来效果最好？是否存在协同或拮抗效应？**

基线：Phase4 T5 (B2 V2 + D2 + D4) epoch_0007，all_pairs clip_style=0.7307, content_lpips=0.3403。

## User Scenarios & Testing

### User Story 1 — 推理侧单因素消融 (Priority: P1)

作为研究者，我需要知道每个推理侧参数（从默认值变为有效值时）对 clip_style / content_lpips / WFI 的独立贡献，以便确定哪些参数值得保留、哪些无效、哪些有害。

**Why this priority**: 推理侧消融无需重新训练（~2min/config），是最快获得确定性结论的路径，且Phase4已有部分数据可直接复用。

**Independent Test**: 每个参数独立改变，其余参数保持T5 ep7基线值，跑eval后对比三指标变化。

**Acceptance Scenarios**:

1. **Given** T5 ep7基线 (clip=0.7307, lpips=0.3403), **When** 单独改变 endpoint_adain_scale 从 1.0→0.0 (禁用N1), **Then** 产出该参数的独立Δclip/Δlpips/ΔWFI
2. **Given** T5 ep7基线, **When** 单独改变 patch_adain_kernel 从 0→16, **Then** 产出该参数的独立效果
3. **Given** 全部推理侧单因素结果, **Then** 按效果排序，标注"有效/无效/有害"

---

### User Story 2 — 训练侧单因素消融 (Priority: P1)

作为研究者，我需要知道每个训练侧机制（loss项、训练策略、架构选择）对最终指标的独立贡献，以便确定训练配方中哪些成分是必要的。

**Why this priority**: 训练侧消融需要1 epoch训练（~30min/实验），是理解因果关系的必要步骤。625 Phase3已证明推理侧参数无法突破架构天花板，训练侧是突破口。

**Independent Test**: 每个训练侧参数独立改变（在T5配置基础上），训练1 epoch，跑eval后对比基线。

**Acceptance Scenarios**:

1. **Given** T5训练配置, **When** 单独添加 w_contrast_preserve=2.0, **Then** 训练1 epoch后产出Δclip/Δlpips
2. **Given** T5训练配置, **When** 单独启用 gate_warmup_steps=500, **Then** 训练1 epoch后产出效果
3. **Given** 全部训练侧单因素结果, **Then** 按效果排序，标注"必要/可选/有害"

---

### User Story 3 — 关键2因素组合实验 (Priority: P2)

作为研究者，我需要知道top有效机制组合时是否产生协同（1+1>2）或拮抗（1+1<2）效应，以便设计最优配置。

**Why this priority**: 在单因素结果基础上，2因素组合能验证机制间交互，同时控制实验量在可行范围内。

**Independent Test**: 选取单因素top-3有效机制，做2×2交叉组合，与单因素加和预期对比。

**Acceptance Scenarios**:

1. **Given** 推理侧top-2有效参数A和B, **When** A+B同时启用, **Then** 对比实际Δ与预期Δ(A)+Δ(B)，判断协同/拮抗/独立
2. **Given** 训练侧top-2有效参数C和D, **When** C+D同时启用训练1 epoch, **Then** 对比实际效果与加和预期
3. **Given** 推理+训练最优组合, **Then** 产出综合最优配置及预期指标

---

### User Story 4 — 产出确定性结论文档 (Priority: P1)

作为研究者，我需要一份完整的结论文档，以表格形式列出每个机制的独立效果、组合效果、推荐保留/移除，以及最终推荐配置。

**Why this priority**: 这是用户的最终交付物，所有实验都服务于这份文档。

**Independent Test**: 文档包含所有实验数据表格、机制排序、组合分析、推荐配置。

**Acceptance Scenarios**:

1. **Given** 所有实验完成, **Then** 文档包含：机制清单表（含独立Δclip/Δlpips/ΔWFI）、组合实验表（含协同/拮抗判断）、推荐配置、已知天花板说明
2. **Given** 文档产出, **Then** 用户能直接回答"每个机制有什么效果"和"组合起来有什么效果"

---

### Edge Cases

- 某个消融导致训练崩溃/NaN怎么处理？→ 记录为"不稳定/不可用"，跳过该点
- 推理侧参数组合与Phase4已有数据重叠时？→ 优先复用已有数据，不重复实验
- 训练1 epoch不够暴露效果怎么办？→ 对关键训练消融扩展到3 epoch

## Requirements

### Functional Requirements

- **FR-001**: 系统必须对推理侧10+个参数执行单因素消融，每个参数至少测试2个水平（启用/禁用或2个代表性值）
- **FR-002**: 系统必须对训练侧8+个参数执行单因素消融，每个参数训练3 epoch并eval（历史数据显示3 epoch足以判断效果方向：T5/N11+N16在ep3已显现趋势，N1_lvl2 ep3即peak，T4 ep1已退化）
- **FR-003**: 系统必须对推理侧+训练侧各top-3有效机制执行2因素组合实验
- **FR-004**: 每次实验必须记录：all_pairs_clip_style, content_lpips, (WFI如可获取), 与基线的Δ值
- **FR-005**: 复用Phase4已有的25+组推理消融数据，避免重复实验
- **FR-006**: 所有实验结果必须汇总为结构化文档，含机制排名表、组合协同/拮抗表、推荐配置
- **FR-007**: 训练侧消融必须基于T5训练配置，在远程RTX 3060上执行
- **FR-008**: 推理侧消融基于T5 ep7 checkpoint，在远程RTX 3060上执行
- **FR-009**: 实验批量运行脚本必须支持远程WSL执行（SCP上传+SSH执行模式）
- **FR-010**: 每组实验后自动提取summary.json中关键指标，汇总到统一CSV

### Key Entities

- **消融机制 (AblationMechanism)**: 一个可独立开关的模型参数或训练策略，有名称、作用域(推理/训练/两者)、参数名、默认值、测试值
- **消融实验 (AblationExperiment)**: 一个具体的实验配置，对应一个config JSON，产出clip/lpips/WFI指标
- **消融结果 (AblationResult)**: 实验产出数据，含机制名、参数值、clip_style、lpips、WFI、Δvs基线
- **组合实验 (ComboExperiment)**: 2个机制同时改变的实验，产出实际Δ和预期加和Δ的对比

## Success Criteria

### Measurable Outcomes

- **SC-001**: 产出包含15+个单因素消融结果的完整表格，每个结果有明确的Δclip/Δlpips数值
- **SC-002**: 产出包含6+组2因素组合实验的协同/拮抗分析表
- **SC-003**: 最终文档能确定性回答"每个机制有什么效果"和"组合起来有什么效果"这两个问题
- **SC-004**: 推荐配置的指标不低于当前T5 ep7基线（clip≥0.7307, lpips≤0.3403）
- **SC-005**: 全部实验在12小时内完成（远程RTX 3060单卡）

## Assumptions

- T5 ep7 checkpoint在远程服务器完整可用（已验证：`exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt`）
- T5训练配置的config.json可复用为训练侧消融的base config
- 远程RTX 3060 (12GB VRAM) 可支持batch_size=16的训练和eval
- Phase4已有的25+组推理消融数据可直接复用，无需重跑
- 3 epoch训练（~45min/实验）足够判断效果方向：T5 ep3 clip=0.7282（vs ep7=0.7307，趋势+0.0025），N11+N16 ep3=0.7282（vs ep7=0.7315，趋势+0.0033），N1_lvl2 ep3=0.7243即peak。3 epoch能可靠区分"改善/恶化/无变化"三个方向
- WFI指标在当前eval流程中可能无法自动获取，但clip_style和content_lpips是核心指标
- SSH→WSL嵌套引号问题可通过SCP上传脚本文件方式绕过

## Open Questions

- 3-axis fix（gate_warmup/RMSNorm/anti-whitening）的训练侧消融是否也应包含？T5配置中这些参数当前为默认值（NOT_SET=未启用），是否意味着T5不含3-axis修复？如果是，训练侧消融需要独立测试这3个机制。
- 训练侧消融扩展epoch数：3 epoch已足够判断效果方向（见历史收敛数据），对关键top-3有效机制可扩展到7 epoch确认最终效果
