# AAAI 2027 论文深度清洗重写规范

## Why
当前论文草稿包含大量内部术语、非正式命名和工具特定词汇，不符合顶会投稿标准。需要全面清洗重写，确保：
1. 所有术语使用规范学术语言
2. 图表根据核心叙事重新设计
3. 消除所有内部草稿痕迹

## 当前状态（2026-06-27）
### 已完成
- ✅ 阶段一：术语规范化与文本清洗（Tasks 1-6）
- ✅ Figure 1 重绘及 caption 修复（Task 7）
- ✅ 表格列名规范化（Task 10）
- ✅ 核心术语 "real style transfer" 全文统一
- ✅ 定理数量修正（3→4）及解释段落更新
- ✅ 移除所有内部代号（I7, U4, V6, V3 等）

### 待完成（关键问题）
1. **Bootstrap 过程描述缺失**（Line 260）- 影响可重复性
2. **CycleGAN-256 提及但无结果**（Line 188, 201）- 需移除或补充
3. **Line 315 自引用问题** - Section 指向自身

### 待完成（图表与验证）
4. **Figure 2 重绘**（Task 8）- 三阶段流程图
5. **Figure 3 重绘**（Task 9）- 定性对比图
6. **最终验证** - 编译、术语一致性、叙事连贯性

## What Changes

### BREAKING: 术语规范化
- **移除所有内部代号**：I7, U4, V6, V3, LBM-K, LBM-Knee, LBM-PS, LBM-PS-v2
- **正式定义所有缩写**：OMF, SA-SWD, FC-SB, tw-ArtFID, EdgePurity, NonCLIPAcc
- **移除工具特定词汇**：live-dashboard, HTML payload, pairing cache, successor family
- **重命名方法变体**：使用描述性名称而非代号

### 文字清洗
- 重写 Abstract：聚焦核心贡献，移除变体细节
- 重写 Introduction：清晰阐述问题-方法-结果逻辑链
- 重写 Method：正式定义所有技术概念
- 重写 Experiments：使用规范评估术语
- 重写 Discussion/Conclusion：提炼核心洞察

### 图表重绘
- Figure 1 (Page 1 summary)：重新设计，突出 IDT 校准和效率对比
- Figure 2 (Framework)：简化为三阶段流程，移除内部细节
- Figure 3 (Qualitative)：选择最具代表性的案例
- Table 1-3：规范化列名和单位
- Table 4 (FC-SB)：移除内部代号，使用描述性名称

## Impact
- **Affected specs**: 无（这是独立的论文清洗任务）
- **Affected code**: 
  - `aaai2027/paper_aaai2027.tex` - 主论文文件
  - `aaai2027/figures/` - 图表生成脚本和输出
  - `aaai2027/*.py` - 图表生成脚本

## ADDED Requirements

### Requirement: 术语规范化系统
系统 SHALL 提供完整的术语映射表，将所有内部术语映射到规范学术表达。

#### Scenario: 方法变体命名
- **WHEN** 论文中提及不同方法配置
- **THEN** 使用描述性名称：
  - "LBM-K" → "LBM with kinetic regularization only"
  - "LBM-Knee" → "LBM with balanced regularization"
  - "LBM-PS" → "LBM with enhanced style pressure"
  - "LBM-PS-v2" → "LBM with maximum style pressure"

#### Scenario: 内部检查点命名
- **WHEN** 论文中提及特定检查点或实验变体
- **THEN** 完全移除内部代号（I7, U4, V6, V3），使用参数描述：
  - "I7 checkpoint" → "the base LBM checkpoint"
  - "U4 (α=0.1)" → "style extrapolation with α=0.1"
  - "V6 (k=32)" → "patchwise AdaIN with kernel size 32"
  - "V3 (k=16)" → "patchwise AdaIN with kernel size 16"

### Requirement: 图表重绘标准
系统 SHALL 根据核心叙事重新设计所有图表，确保视觉质量和信息清晰度。

#### Scenario: Figure 1 重绘
- **WHEN** 重新设计 Figure 1
- **THEN** 包含两个面板：
  - (a) IDT 校准散点图：x轴为内容保持(1-LPIPS)，y轴为风格强度(CLIP-S)，标注 IDT 基线，突出 LBM 在 IDT 之上的操作点
  - (b) 效率对比柱状图：x轴为训练时间（分钟/小时），y轴为方法名称，突出 LBM 的分钟级训练优势

#### Scenario: Figure 2 重绘
- **WHEN** 重新设计 Figure 2
- **THEN** 简化为三阶段流程：
  - Stage 1: Style-ID Encoding（风格标识编码）
  - Stage 2: Latent Transport（潜在空间传输）
  - Stage 3: Training Objectives（训练目标）
  - 移除所有内部实现细节

#### Scenario: 表格规范化
- **WHEN** 重新设计所有表格
- **THEN** 使用规范列名：
  - "CLIP-S_tr" → "Style Score (CLIP-S)"
  - "1-LPIPS_tr" → "Content Preservation (1-LPIPS)"
  - "Δ_idt,tr" → "Style Gain over IDT"
  - "tw-ArtFID_all" → "ArtFID (target-pooled)"
  - "EdgePurity" → "Edge Purity"
  - "NonCLIPAcc" → "Non-CLIP Style Accuracy"

### Requirement: 核心叙事一致性
系统 SHALL 确保每个章节紧扣核心故事线：Real style transfer, ultra efficient。

#### Scenario: Introduction 叙事
- **WHEN** 撰写 Introduction
- **THEN** 按以下逻辑组织：
  1. 风格迁移的目标是什么（style identity, not exemplar）
  2. 之前方法做的是什么（exemplar-guided or large-prior）
  3. IDT 揭示了什么问题（no-op failure mode）
  4. 我们如何做好（LBM: minimum-energy endpoint transport）
  5. 为什么这样好（三个定理支撑）
  6. 对未来的启发（execution-side gap）

#### Scenario: Method 叙事
- **WHEN** 撰写 Method
- **THEN** 按以下逻辑组织：
  1. 问题形式化（style-ID-only inference）
  2. 核心原理（minimum-energy endpoint move）
  3. 三个理论支撑（Theorem 1-3）
  4. 训练目标（endpoint objective + SA-SWD）
  5. 推理流程（Euler integration）

#### Scenario: Experiments 叙事
- **WHEN** 撰写 Experiments
- **THEN** 按以下逻辑组织：
  1. 评估协议（IDT calibration）
  2. 主要结果（LBM 超越 IDT 基线）
  3. 效率对比（分钟级 vs 小时级）
  4. 消融实验（不同正则化配置的影响）
  5. 推理时改进（fiber controls 的效果）

## MODIFIED Requirements

### Requirement: 学术写作规范
论文 SHALL 使用顶会标准的学术写作规范，避免所有非正式表达。

#### Scenario: 避免自我表扬
- **WHEN** 描述方法优势
- **THEN** 使用客观陈述：
  - ❌ "The main lesson is not that one row wins every metric"
  - ✅ "LBM achieves a balanced operating point"
  - ❌ "The main claim of the paper is therefore precise"
  - ✅ "These results demonstrate that..."

#### Scenario: 避免防御性表达
- **WHEN** 讨论局限性
- **THEN** 直接陈述：
  - ❌ "Distinct5-WikiArt should also be read in the correct way"
  - ✅ "Distinct5-WikiArt is a deliberately hard stress benchmark"
  - ❌ "The limitations are direct"
  - ✅ "Several limitations should be noted"

#### Scenario: 避免内部术语
- **WHEN** 提及技术细节
- **THEN** 使用正式定义：
  - ❌ "pairing cache exposes a selector-restricted candidate set"
  - ✅ "a precomputed set of candidate endpoints"
  - ❌ "successor family"
  - ✅ "enhanced model variants"
  - ❌ "Stokes coefficient"
  - ✅ "style pressure coefficient"

## REMOVED Requirements

### Requirement: 内部工具引用
**Reason**: 论文不应提及内部开发工具或调试界面
**Migration**: 移除所有提及 live-dashboard, HTML payload, internal scripts 的内容

### Requirement: 实验变体代号
**Reason**: 内部代号不符合学术规范
**Migration**: 使用参数描述或描述性名称替代所有代号（I7, U4, V6, V3, LBM-K, LBM-Knee 等）
