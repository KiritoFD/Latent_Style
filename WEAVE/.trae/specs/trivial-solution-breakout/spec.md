# 平凡解突破计划 - Product Requirement Document

## Overview
- **Summary**: 基于6个月实验数据与5种平凡解形成机制的数学分析，先在本地 GPU 快速验证三层干预方案（架构去安全阀 + 训练目标重构 + 训练策略调整），使 SpatialBridge620 模型跳出 Endpoint Shrinkage / Gate Collapse / 条件期望坍缩等保守策略。优先验证低成本高回报的方案（Endpoint Head 去 GroupNorm、FiLM-only 注入），再验证范式级改变（Endpoint-supervised 训练），最后组合优化。
- **Purpose**: 解决当前模型系统性选择保守策略的根因——在现有训练目标下，保守（低gate、小位移、均匀attention）确实是 loss 最优解。通过本地快速迭代验证各方案的有效性，为后续大规模远程实验筛选有效路径。
- **Target Users**: 风格迁移模型研究者与开发者，实验驱动的架构迭代团队。

## Goals
- **G1: 本地快速验证（Phase 1）** — 在本地 GPU 上用 1-epoch smoke test 快速验证各方案有效性，筛选出有前景的方向
- **G2: 架构去安全阀验证** — 验证 Endpoint Head 去 GroupNorm 和 FiLM-only 注入的独立贡献与组合效果
- **G3: 训练目标重构验证** — 验证 endpoint-supervised 训练能否消除 Training-Output Mismatch
- **G4: 训练策略优化验证** — 验证两阶段训练和 style strength 正则化的效果
- **G5: 组合优化（Phase 2）** — 在有效方案基础上做组合优化，找到局部最优配置

## Non-Goals (Out of Scope)
- 不改变 backbone 主体结构（仍用 SpatialBridge620 + DINO style encoder）
- 不引入新的大规模预训练或新数据集
- 不做扩散模型架构的全面迁移（保持 Flow Matching / SB 框架）
- 不做 text-to-image 或开放域生成，专注于图像到图像的风格迁移
- 不追求 SOTA 刷榜，目标是验证机制并突破当前天花板

## Background & Context

### 历史问题总结
基于 [05_lessons_learned.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/622/history/05_lessons_learned.md) 和 [trivial_solution.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/620/fog/theory/trivial_solution.md)，6个月的实验反复验证：

1. **保守策略反复出现**：每次解决一个保守问题（均匀attention、IN白化、heuristic loss膨胀），模型就在另一个维度重新选择保守（gate collapse、endpoint shrinkage、WFI恶化）
2. **这不是偶然**：在当前 training objective 下，保守策略确实是 loss 最优的
3. **五种机制共同作用**：Attention-driven × Endpoint-driven × Norm-driven × Loss-driven × Solver-driven = 总 shrinkage 系数 alpha ≈ 0.16

### 当前基线数据
| 指标 | 当前最优 | 目标值 | 差距 |
|------|---------|--------|------|
| clip_style | 0.676 (lowmix05, 1ep) | > 0.72 | +0.044 |
| LPIPS | 0.278 (lowswd, 2ep) | < 0.35 | 已达标 |
| WFI | 0.391 (film_v5_hd512, 1ep) | < 0.30 | -0.09 |
| endpoint_alpha | 0.163 (targetlinear, 8ep) | > 0.50 | +0.337 |
| style_gate | 0.048 (几乎所有实验) | > 0.30 | +0.25 |
| cross_attn_entropy | 6.24 (近均匀) | < 4.0 (更尖锐) | -2.24 |

### 关键理论洞察
来自 [07_theory_corrections.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/622/history/07_theory_corrections.md)：

- **Training-Output Mismatch**：Fiber-SDE 不训练就达 0.711，训练后反而更差（0.701），说明训练过程在"走偏"
- **SWD梯度与FM梯度正交**：cos ≈ -0.024，两个 loss 拉不同方向，模型选择优先满足 FM
- **有效style维度极低**：可能在10-50之间，远小于 4×64×64=16384
- **Per-style >> 通用模型**：差距 0.05-0.09，说明多风格泛化的平均化效应显著

## Functional Requirements

### FR-1: 数学理论框架文档
- 建立平凡解形成的统一数学模型（5机制乘积效应）
- 给出每种机制的可量化指标与阈值判定
- 提出至少 8 个可证伪预测及对应验证实验设计
- 文档化代码中各机制的对应位置

### FR-2: Endpoint-supervised 训练目标
- 实现 endpoint 直接监督模式：L_endpoint = w_content·L_content + w_style·L_style
- Content 项：低频结构保持（L1 / LPIPS 可选）
- Style 项：SWD 分布匹配（保留但调整权重）
- Velocity 项：保留为可选正则化（权重可配置）
- 与现有 velocity-supervised 模式可切换，通过 config 控制

### FR-3: Style Strength 正则化
- 实现 endpoint_alpha 奖励项：鼓励 endpoint 向 target 方向移动
- 实现 style 方向位移奖励：只奖励 style 子空间分量，不奖励 content 扰动
- 权重可配置，默认关闭，通过 config 启用

### FR-4: 架构去安全阀 - Gate 改造
- 实现 FiLM-only 注入模式：移除 style_gate，改用 FiLM 直接调制特征
- 验证 pre-cross-attn FiLM + post-cross-attn FiLM 的组合效果
- 保留 gated 模式作为可选，通过 config 切换

### FR-5: 架构去安全阀 - Endpoint Head 改造
- 移除 FiLMEndpointHead 中的 GroupNorm(1)
- 验证无 GN 版本对 WFI 和 alpha 的影响
- 保留有 GN 版本作为 baseline，通过 config 控制

### FR-6: 两阶段训练策略
- Stage 1（风格注入期）：高 SWD 权重、低 FM 权重，强制打破保守平衡
- Stage 2（平衡微调期）：恢复正常权重，微调内容保持
- 通过 config 配置两阶段的 epoch 数和权重比

### FR-7: 实验配置生成与评估
- 生成系统性 ablation 配置矩阵（~20-30组）
- 覆盖各独立方案和关键组合
- 自动评估并生成结果对比表

## Non-Functional Requirements

### NFR-1: 向后兼容性
- 所有改动通过 config 开关控制，默认配置与当前 baseline 行为一致
- 不破坏现有训练 pipeline 和评估流程

### NFR-2: 显存约束
- 所有实验必须在 12GB VRAM（RTX 3060）下安全运行
- batch_size ≥ 24（base_dim=64时）或 ≥ 16（base_dim=128时）

### NFR-3: 训练稳定性
- 新方案不得导致训练崩溃（NaN / Inf / black-dot）
- loss 曲线应平滑下降，无剧烈震荡

### NFR-4: 可验证性
- 每个方案都有明确的量化评估指标
- 结果可通过现有 dashboard 和评估脚本观测

## Constraints

### 技术约束
- 基于现有 SpatialBridge620 架构，不做大规模重写
- 保持 PyTorch + Flow Matching 技术栈
- 12GB VRAM 上限

### 业务约束
- 实验周期：Phase 1（本地快速验证）1-2天，Phase 2（组合优化）3-5天
- 仅使用本地 GPU，不动远程训练
- 人力：单开发者 + AI 辅助

### 依赖
- 现有训练/评估基础设施（trainer.py、run_evaluation.py、dashboard）
- WikiArt distinct5 512×512 数据集
- DINOv2 预训练模型（style encoder）

## Assumptions

- **A1**: 平凡解是多机制共同作用的结果，单一修复不足以突破，需要组合干预
- **A2**: Endpoint-supervised 训练比 velocity-supervised 更直接有效，因为消除了 Training-Output Mismatch
- **A3**: 移除 style gate 和 endpoint GN 不会导致训练不稳定，因为 FiLM 和其他正则化可以提供稳定性
- **A4**: 两阶段训练能有效打破保守平衡——先强制注入风格，再微调内容
- **A5**: 当前模型容量足够（64×4 或 64×6），瓶颈不在容量而在注入策略

## Acceptance Criteria

### AC-1: 数学理论框架完备
- **Given**: 已有的实验数据和理论文档
- **When**: 完成理论框架文档
- **Then**: 文档包含5种机制的数学定义、乘积模型、至少8个可证伪预测、以及代码对应索引
- **Verification**: `human-judgment`
- **Notes**: 由研究者评审理论自洽性和实验可验证性

### AC-2: Endpoint-supervised 训练可运行
- **Given**: 配置 `training_objective_mode = "endpoint_supervised"`
- **When**: 运行 1 epoch 训练
- **Then**: 训练正常完成，无崩溃，loss 下降，endpoint_alpha > baseline
- **Verification**: `programmatic`
- **Notes**: 对比 baseline（velocity mode）的 endpoint_alpha 和 WFI

### AC-3: Style Strength 正则化有效
- **Given**: 启用 style_strength_reg，权重设为 0.5
- **When**: 训练 1 epoch
- **Then**: endpoint_alpha 显著高于未启用版本（提升 > 20%），且 LPIPS 恶化 < 10%
- **Verification**: `programmatic`
- **Notes**: 确保"大胆移动"主要在 style 方向而非 content 方向

### AC-4: FiLM-only 模式增强注入
- **Given**: 配置 `style_film_enabled = true` 且 `style_gate_init = 0`（gate 移除或设为1）
- **When**: 训练 1 epoch 并评估
- **Then**: style_gate 值不再是瓶颈，cross_attn_entropy 下降，clip_style 提升 > 0.01
- **Verification**: `programmatic`
- **Notes**: 关注训练稳定性，若 loss 震荡则需要 warmup

### AC-5: Endpoint Head 去 GN 改善白化
- **Given**: FiLM endpoint head 移除 GroupNorm
- **When**: 训练 1 epoch 并评估 WFI
- **Then**: WFI 下降 > 0.03，endpoint_alpha 上升 > 0.05
- **Verification**: `programmatic`
- **Notes**: WFI 改善应伴随动态范围扩大，而非简单的亮度变化

### AC-6: 两阶段训练突破保守平衡
- **Given**: Stage1（高SWD/低FM，1-2epoch）+ Stage2（正常权重，1-2epoch）
- **When**: 完成完整两阶段训练
- **Then**: clip_style > 0.70 且 LPIPS < 0.35，优于单阶段训练
- **Verification**: `programmatic`
- **Notes**: 两阶段应优于"用同样总epoch数的单阶段训练"

### AC-7: 最优组合达到目标指标
- **Given**: 所有有效方案的最优组合
- **When**: 训练 3 epoch 并完整评估
- **Then**: clip_style > 0.72，LPIPS < 0.35，WFI < 0.35，endpoint_alpha > 0.4
- **Verification**: `programmatic`
- **Notes**: 若部分指标未达标，分析原因并迭代方案

### AC-8: 向后兼容性验证
- **Given**: 使用默认配置（所有新功能关闭）
- **When**: 运行 1 epoch 训练
- **Then**: 结果与 baseline 一致（差异 < 1%），无回归
- **Verification**: `programmatic`
- **Notes**: 确保旧配置文件仍可正常运行

## Open Questions

- [ ] **Q1**: Endpoint-supervised 训练是否会导致训练不稳定？需要多大的 velocity 正则化权重来稳定？
- [ ] **Q2**: 移除 style gate 后，模型是否会过度注入风格导致内容崩溃？content loss 需要多强？
- [ ] **Q3**: 两阶段训练的最优阶段长度和权重比是多少？是否需要更多阶段？
- [ ] **Q4**: Style strength 正则化如何区分"style方向"和"content方向"的位移？需要显式的子空间分解吗？
- [ ] **Q5**: 当前 WFI 计算公式是否能准确反映白化程度？是否需要补充其他指标（如 SIFT 特征匹配度）？
