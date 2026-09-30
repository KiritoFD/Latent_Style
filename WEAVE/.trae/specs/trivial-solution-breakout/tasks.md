# 平凡解突破计划 - The Implementation Plan (Local Quick Validation Phase)

## Phase 1: 本地快速验证（按改动成本从低到高排序）

---

## [ ] Task 1: Endpoint Head 去 GroupNorm（最低成本，最先验证）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 修改 `FiLMEndpointHead` 类，新增 `use_norm` 配置选项
  - `use_norm=False` 时移除开头的 GroupNorm(1)
  - 新增 config 参数：`endpoint_film_use_norm`（默认 true，向后兼容）
  - 本地跑 1 epoch smoke test，对比 baseline 的 WFI 和 endpoint_alpha
  - 这是成本最低的改动（1-2行代码），理论预期收益：WFI↓ > 0.03
- **Acceptance Criteria Addressed**: AC-5, AC-8
- **Test Requirements**:
  - `programmatic` TR-1.1: 默认配置（use_norm=true）结果与 baseline 一致
  - `programmatic` TR-1.2: 无 GN 版本训练正常，无崩溃
  - `programmatic` TR-1.3: 无 GN 版本 WFI 下降 > 0.03
  - `programmatic` TR-1.4: 无 GN 版本 endpoint_alpha 上升 > 0.05
- **Notes**: 优先做，因为改动最小、理论最明确、验证最快

---

## [ ] Task 2: FiLM-only 注入模式验证（已有代码，主要是验证）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 验证代码中已有的 `style_film_enabled = true` 模式
  - 新增 `style_gate_mode` 配置：tanh_gate（默认） | fixed_one | film_only
  - `film_only` 模式：移除 gate，cross-attn 输出 + FiLM 直接调制
  - 本地跑 1 epoch smoke test，对比 baseline 的 clip_style 和 cross_attn_entropy
  - 理论预期收益：clip_style↑ > 0.01，attention 更尖锐
- **Acceptance Criteria Addressed**: AC-4, AC-8
- **Test Requirements**:
  - `programmatic` TR-2.1: 默认配置（tanh_gate）结果与 baseline 一致
  - `programmatic` TR-2.2: film_only 模式训练正常，无崩溃
  - `programmatic` TR-2.3: film_only 模式 clip_style 提升 > 0.01
  - `programmatic` TR-2.4: film_only 模式 cross_attn_entropy 下降
- **Notes**: 代码中已有 FiLM 实现，主要是配置开关和验证效果

---

## [ ] Task 3: 组合验证：去GN + FiLM-only
- **Priority**: P0
- **Depends On**: Tasks 1, 2
- **Description**:
  - 组合 Task 1 + Task 2 的有效配置
  - 本地跑 1 epoch smoke test
  - 验证组合效应是否是叠加（1+1=2）还是协同（1+1>2）
  - 如果组合效果好，继续后续方案；如果效果差，分析原因
- **Acceptance Criteria Addressed**: AC-7
- **Test Requirements**:
  - `programmatic` TR-3.1: 组合配置训练正常，无崩溃
  - `programmatic` TR-3.2: 组合效果优于单独使用任一方案
  - `programmatic` TR-3.3: WFI < 0.36 且 clip_style > 0.68
- **Notes**: 先验证最容易的两个方案的组合，再决定是否继续更复杂的方案

---

## [ ] Task 4: Style Strength 正则化实现
- **Priority**: P1
- **Depends On**: Task 3（如果组合效果好才继续）
- **Description**:
  - 在 `losses620.py` 中新增 style_strength_reg 项
  - 基础版：endpoint_alpha 奖励 -lambda * ||endpoint - source|| / ||target - source||
  - 新增 config 参数：`w_style_strength_reg`（默认 0.0）
  - 本地跑 1 epoch smoke test，权重从 0.1 开始试
  - 理论预期：endpoint_alpha↑ > 20%，LPIPS 恶化 < 10%
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-4.1: 关闭时（w=0）结果与 baseline 一致
  - `programmatic` TR-4.2: 开启时 endpoint_alpha 提升 > 20%
  - `programmatic` TR-4.3: LPIPS 恶化 < 10%
  - `programmatic` TR-4.4: 训练稳定，无 NaN/Inf
- **Notes**: 如果 Task 3 组合效果已经很好，可以跳过或降低优先级

---

## [ ] Task 5: Endpoint-supervised 训练目标实现（范式级改变）
- **Priority**: P1
- **Depends On**: Task 3
- **Description**:
  - 在 `losses620.py` 中新增 endpoint-supervised 模式
  - 新增 config 参数：`training_objective_mode` (velocity | endpoint)
  - Endpoint loss 组成：L_content（低频L1） + L_style（SWD） + L_velocity_reg（可选）
  - 本地跑 1 epoch smoke test
  - 这是范式级改变，预期影响最大，但实现和调试成本最高
- **Acceptance Criteria Addressed**: AC-2, AC-8
- **Test Requirements**:
  - `programmatic` TR-5.1: 默认配置（velocity mode）结果与 baseline 一致
  - `programmatic` TR-5.2: endpoint mode 训练正常，无崩溃
  - `programmatic` TR-5.3: endpoint mode endpoint_alpha 高于 velocity mode
  - `programmatic` TR-5.4: loss 曲线平滑下降
- **Notes**: 这是最大的改动，放在后面验证，确保前面的简单方案先排除

---

## [ ] Task 6: 两阶段训练策略验证
- **Priority**: P2
- **Depends On**: Task 5
- **Description**:
  - 在 trainer 中支持两阶段训练配置
  - Stage 1：高 SWD 权重、低 FM 权重，1 epoch
  - Stage 2：正常权重，1 epoch
  - 本地跑 2 epoch 训练
  - 对比"2 epoch 单阶段训练"的效果
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `programmatic` TR-6.1: 两阶段训练正常完成
  - `programmatic` TR-6.2: 两阶段优于相同总 epoch 数的单阶段
  - `programmatic` TR-6.3: clip_style > 0.70 且 LPIPS < 0.35
- **Notes**: 只有在 endpoint-supervised 验证有效的情况下才做

---

## [ ] Task 7: 最优组合验证（3 epoch）
- **Priority**: P1
- **Depends On**: Tasks 1-6 中有效的方案
- **Description**:
  - 汇总 Phase 1 中验证有效的所有方案
  - 组合最优配置，跑 3 epoch 训练
  - 完整评估：clip_style, LPIPS, WFI, endpoint_alpha, 以及可视化
  - 对比当前最优 baseline
- **Acceptance Criteria Addressed**: AC-7
- **Test Requirements**:
  - `programmatic` TR-7.1: 3 epoch 训练稳定完成
  - `programmatic` TR-7.2: clip_style > 0.72 且 LPIPS < 0.35
  - `programmatic` TR-7.3: WFI < 0.35 且 endpoint_alpha > 0.4
  - `programmatic` TR-7.4: 全面优于当前最优 baseline
- **Notes**: Phase 1 的最终验收点

---

## [ ] Task 8: 向后兼容性回归测试
- **Priority**: P0
- **Depends On**: 所有改动完成后
- **Description**:
  - 用默认配置（所有新功能关闭）跑 1 epoch 训练
  - 对比 baseline 的 loss 曲线和关键指标
  - 确保旧配置文件仍可正常运行
- **Acceptance Criteria Addressed**: AC-8
- **Test Requirements**:
  - `programmatic` TR-8.1: 默认配置下 loss 曲线与 baseline 差异 < 1%
  - `programmatic` TR-8.2: 关键指标差异 < 1%
  - `programmatic` TR-8.3: 旧配置文件可无缝运行
- **Notes**: 必须通过，确保不破坏现有工作流
