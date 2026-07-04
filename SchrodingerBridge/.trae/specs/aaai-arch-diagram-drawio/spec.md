# AAAI 主架构图 — draw.io MCP 绘制 Spec

## Why
现有 `gen_fig2_framework.py` 是 matplotlib 绘制的抽象三阶段流程图，缺乏模型内部细节（DWT 分解、谱域 ODE、速度头、Endpoint AdaIN 等核心创新）。需要一张精确反映 SpectralODEBridge620 真实架构的专业图，用于 AAAI 论文主图（Figure 2 替换或补充）。

## What Changes
- 先深入阅读理论文档，理解 Spectral ODE Bridge 的设计动机、为什么 work、核心创新点
- 调研 AAAI 论文主架构图的视觉风格（从 `F:\papers\latent_style_cited_20260605\AAAI` 中的几篇开始）
- 基于理论洞察（而非代码逐行）提炼出 1-2 张高层次的、故事性强的主架构图
- 使用 draw.io MCP (`open_drawio_xml`) 生成图，经子 agent 视觉评审迭代至满意

## Impact
- Affected specs: 无（新增）
- Affected code: draw.io MCP 工具调用 → 输出 .drawio XML → 用户可导出 PDF/PNG

## Theoretical Foundation (必须优先理解)

### 核心问题：为什么需要 Spectral ODE Bridge？
- 传统欧氏空间 Flow Matching 把内容 latent 直接推向风格 latent，导致低频结构破坏（LPIPS 升高）
- 风格信息主要分布在高频子带（LH/HL/HH），内容信息主要在 LL
- Haar DWT 提供正交分解，让模型可以在频域原生求解 ODE，而不是欧氏空间事后投影

### 核心洞察（why it works）
1. **频域解耦**：LL 保内容（w_ll 很小，推理时可锁死），LH/HL 传风格
2. **共享 Backbone 学习跨子带交互**：4 子带堆叠输入，但 3 个独立速度头分别输出
3. **Endpoint AdaIN 作为显式风格注入**：在轨迹终点用统计匹配把风格协方差/均值注入高频纤维
4. **ReLU² Attention 提供稀疏、幅值保持的 cross-attention**，避免 softmax 的平滑化
5. **信息瓶颈 + DWT-Route**：可选的 LL bypass 让 cross-attention 只处理高频，保护结构

### 想传达的故事（Storyline）
图不应该画成“模块连接图”，而应该讲一个故事：
> 我们把 latent 拆成不同频率的子带，在频域里让模型只学习如何改变高频，
> 而低频结构被显式保护；最后再用 AdaIN 把风格统计量精准注入。

## AAAI 论文风格调研
- 目标路径：`F:\papers\latent_style_cited_20260605\AAAI`
- 调研维度：
  - 主架构图是流程图式、模块图式、还是公式+模块混合式？
  - 如何表达 multi-scale / frequency decomposition？
  - 如何表达 training vs inference 两条路径？
  - 配色、字体、箭头风格、图注位置
- 输出：一份风格调研笔记，作为后续绘图的参考

## Architecture Diagram Direction (基于理论而非代码)

### 第一稿方向：Spectral Decomposition + ODE Story
- 视觉隐喻：把 latent 画成一个可分解为“层”的物体（LL 层 + LH/HL/HH 层）
- 内容路径（蓝色）：从 z₀ → DWT → LL 被保护 → 经过 backbone 后 LL 速度≈0
- 风格路径（橙色）：style memory → 高频子带 → velocity heads
- ODE 积分（绿色）：在谱域中一步步更新 LH/HL/HL，LL 保持不变
- 终点注入（紫色）：Endpoint AdaIN 把风格统计量注入纤维

### 第二稿方向：Training vs Inference Dual View
- 左半：Training —— 从 x_t 预测 v_LL, v_LH, v_HL，与目标频带残差对齐
- 右半：Inference —— 从 z₀ 出发，用训练好的速度场积分到 z_T
- 中间共享：同一个 Backbone + Style Conditioner

## MODIFIED Requirements
### Requirement: 理论驱动的架构图
架构图必须首先回答“为什么这样设计”和“模型为什么 work”，其次才展示模块连接。每个主要模块旁边应有简短的理论说明。

#### Scenario: 论文读者视角
- **WHEN** 读者看图
- **THEN** 能在 30 秒内理解：频域解耦、低频保护、高频风格传输、Endpoint AdaIN 这四个核心思想

### Requirement: AAAI 风格一致
最终图的布局、配色、箭头、字体、图注风格应与 AAAI 已发表论文（尤其是同一领域的风格迁移论文）保持一致。

#### Scenario: 风格对比
- **WHEN** 把新图与 AAAI 论文图并排放置
- **THEN** 不应有突兀的视觉差异

## REMOVED Requirements
### Requirement: 代码逐行映射的精确性
**Reason**: 过度精确会变成“类图”，失去论文主架构图应有的故事性
**Migration**: 在图的补充材料或文档中提供代码映射表，主图保留理论层面的精确性
