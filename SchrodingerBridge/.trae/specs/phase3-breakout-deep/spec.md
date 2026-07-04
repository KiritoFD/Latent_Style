# Phase 3: 深度突破平凡解 — 理论驱动的根因修复 Spec

## Why

Phase 1（12轮）+ Phase 2（4轮）共 16 轮实验取得了 **~60% 雾化改善**（9/10 → ~3.5/10 with AdaIN），但：
- clip_style 停留在 ~0.68-0.70，**未达目标 >0.72**
- LPIPS 在 0.47-0.50，**未达目标 <0.40**
- AdaIN 是后处理 patch，**不是根本解决**
- 根据《平凡解突破的统一数学理论》，我们只打破了 **1 个条件**（velocity magnitude），还需打破第 2 个才能跳出平凡解吸引盆

### 理论核心洞见（尚未充分验证的方向）

| 已尝试 | 效果 | 对应打破的条件 |
|--------|------|---------------|
| 去 GN | 微效 | L3 (Norm) 部分打破 |
| Fixed One Gate | 有效 (×19 delta) | L2 (Gate) 打破 |
| Velocity Mag Loss | **高效** (16%→88%) | C1 (FM主导) 部分打破 |
| AdaIN 后处理 | **显著** (饱和度+40-70%) | 运行时补偿 |
| **降低 FM 权重** | ❌ **未尝试！** | C1 (FM主导) 直接打破 |
| **Style-Specific Target** | ❌ **未尝试！** | 2.3 (条件期望坍缩) |
| **Contrastive Loss** | ❌ **未尝试！** | 让不同 style 输出分开 |
| **FiLM 大初始化** | ❌ **未尝试！** | L4 (初始化) 打破 |

## What Changes

### 核心：从"修症状"转向"改 loss landscape"

Phase 1-2 的改动都在模型架构和后处理层面。Phase 3 要改的是**训练目标的本质**：

1. **降低 FM 权重 / 提高 Style 权重** — 直接改变 loss landscape 的形状
2. **Style-Aware Velocity Target** — 避免多风格平均化
3. **Style Contrastive Loss** — 强制不同 style 的输出彼此远离
4. **FiLM 大初始化** — 从非保守起点开始训练
5. **组合最优方案 + AdaIN + 长 epoch early stopping**

## Impact
- Affected code: `src/losses620.py`（新增 loss 项）、`src/config_schema.py`（新参数）、训练 config
- 继承: `deep-dehaze-optize` 的所有发现（VelMag + AdaIN）
- 目标指标: **clip_style > 0.72**, **LPIPS < 0.40**, 雾化 < 3/10

## ADDED Requirements

### Requirement: FM Weight Reduction / Style Loss Enhancement

系统 SHALL 支持降低 Flow Matching loss 权重的配置，改变 loss landscape 中 FM vs SWD 的相对强度。

#### Scenario: FM 权重降低生效
- **WHEN** `w_flow_scale < 1.0`
- **THEN** FM loss 被乘以该系数，降低其在总 loss 中的主导地位
- **THEN** SWD / style 相关 loss 相对权重提升
- **THEN** 模型更倾向于学习 style-specific velocity 而非通用保守解
- **默认**: `w_flow_scale = 1.0`（向后兼容）

### Requirement: Style-Aware Velocity Target（避免条件期望坍缩）

系统 SHALL 支持按 style 分组计算 velocity target，而非 batch 平均。

#### Scenario: Style-Aware Target 生效
- **WHEN** `style_aware_target = true`
- **THEN** 每个 sample 的 v_target 不再是简单的 y_proj - x_t
- **THEN** v_target 包含 style-specific 的方向偏置（基于 style embedding 的线性投影）
- **或者更简单**: 按 style group 做 separate batch norm on v_target
- **目的**: 避免 $\bar{v}^* = \mathbb{E}_s[v^*_s]$ 的平均化效应

### Requirement: Style Contrastive Loss（风格对比损失）

系统 SHALL 提供可选的风格对比损失，强制同一 source + 不同 target style 的输出彼此区分。

#### Scenario: Style Contrastive Loss 生效
- **WHEN** `w_style_contrastive > 0`
- **THEN** 同一 batch 中，相同 source 但不同 target style 的 z_1_hat 输出被计算对比损失
- **THEN** 使用 InfoNCE 或 simple cosine distance 惩罚：`loss = max(0, cos_sim(z_s1, z_s2) - margin)`
- **目的**: 打破"不同 style 输出几乎相同"的平凡解特征

### Requirement: FiLM Large Initialization

系统 SHALL 支持 FiLM 参数的大初始化选项，让训练从非保守起点开始。

#### Scenario: FiLM 大初始化生效
- **WHEN** `film_init_std > 0`（如 0.05 或 0.1）
- **THEN** 所有 FiLM 层的 gamma/beta 参数初始化为 N(1, film_init_std) 和 N(0, film_init_std)
- **而非当前的 N(1, 0) 和 N(0, 0)
- **目的**: 初始时就有 style 调制能力，不从 identity 开始

## MODIFIED Requirements

### 最优基线配置更新
- 基于 Phase 2 最终方案：R4-D1 (3ep, w_vel_mag=0.5) + AdaIN ON
- 新增上述 loss 项到此基线上组合验证

## Open Questions
- [ ] 降低 FM 权重到多少合适？（0.5? 0.3? 0.1?）
- [ ] Style Contrastive Loss 的 margin 设多少？
- [ ] FiLM 大初始化是否会导致训练不稳定？
- [ ] 这些改动与 AdaIN 是否兼容（AdaIN 是否仍需要）？
