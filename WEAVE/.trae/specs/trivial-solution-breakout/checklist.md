# 平凡解突破计划 - Verification Checklist (Local Quick Validation Phase)

## Phase 1: 本地快速验证（按优先级排序）

---

### Task 1: Endpoint Head 去 GroupNorm
- [ ] `endpoint_film_use_norm` 配置参数存在，默认值为 true（向后兼容）
- [ ] 无 GN 版本（use_norm=false）训练正常运行，无崩溃
- [ ] 无 GN 版本 WFI 下降 > 0.03（白化改善）
- [ ] 无 GN 版本 endpoint_alpha 上升 > 0.05（位移更大）
- [ ] 输出动态范围合理，无特征爆炸（数值在合理区间）
- [ ] 默认配置（use_norm=true）结果与 baseline 一致（差异 < 1%）

---

### Task 2: FiLM-only 注入模式
- [ ] `style_gate_mode` 配置参数存在，默认值为 "tanh_gate"（向后兼容）
- [ ] "film_only" 模式下训练正常运行
- [ ] Pre-cross-attn FiLM 和 post-cross-attn FiLM 都正确工作
- [ ] FiLM-only 模式下 clip_style 提升 > 0.01（相比 gated baseline）
- [ ] FiLM-only 模式下 cross_attn_entropy 下降（attention 更尖锐）
- [ ] 训练稳定，无 loss 震荡或发散
- [ ] 默认配置（tanh_gate）结果与 baseline 一致

---

### Task 3: 组合验证：去GN + FiLM-only
- [ ] 组合配置训练正常完成，无崩溃
- [ ] 组合效果优于单独使用任一方案（协同效应）
- [ ] WFI < 0.36 且 clip_style > 0.68
- [ ] LPIPS 仍在可接受范围（< 0.35）
- [ ] 训练稳定，loss 曲线平滑

---

### Task 4: Style Strength 正则化
- [ ] `w_style_strength_reg` 配置参数存在，默认值为 0.0
- [ ] 关闭时（w=0）结果与 baseline 完全一致
- [ ] 开启时 endpoint_alpha 提升 > 20%
- [ ] LPIPS 恶化 < 10%（内容保持尚可接受）
- [ ] 训练稳定，loss 曲线平滑
- [ ] Style strength 奖励值正确输出到 debug metrics

---

### Task 5: Endpoint-supervised 训练目标
- [ ] `training_objective_mode` 配置参数存在，默认值为 "velocity"（向后兼容）
- [ ] "endpoint" 模式下训练正常运行，无崩溃、无 NaN/Inf
- [ ] Endpoint loss 包含 content 项和 style 项，权重可配置
- [ ] Velocity 正则项可配置，默认关闭或权重为 0
- [ ] Endpoint mode 下 endpoint_alpha 高于 velocity mode baseline（提升 > 10%）
- [ ] 两种模式可通过 config 无缝切换，训练流程一致
- [ ] Debug metrics 正确输出各 loss 分量

---

### Task 6: 两阶段训练策略
- [ ] `two_stage_config` 配置参数存在，默认关闭（向后兼容）
- [ ] Stage 1 和 Stage 2 的权重配置可分别设置
- [ ] 两阶段训练正常完成，checkpoint 正确保存和加载
- [ ] 两阶段结果优于相同总 epoch 数的单阶段训练
- [ ] Stage 1 结束时 style 注入显著（endpoint_alpha > baseline）
- [ ] Stage 2 结束时内容保持恢复（LPIPS 回到可接受范围）
- [ ] clip_style > 0.70 且 LPIPS < 0.35

---

### Task 7: 最优组合验证（3 epoch）
- [ ] 3 epoch 训练稳定完成，无崩溃
- [ ] clip_style > 0.72
- [ ] LPIPS < 0.35
- [ ] WFI < 0.35
- [ ] endpoint_alpha > 0.4
- [ ] 全面优于当前最优 baseline
- [ ] 可视化结果质量提升（人工检查）

---

### Task 8: 向后兼容性
- [ ] 默认配置（所有新功能关闭）下训练 loss 曲线与 baseline 差异 < 1%
- [ ] 默认配置下关键指标（clip_style, LPIPS, WFI）差异 < 1%
- [ ] 旧配置文件（如 620_spatial_bridge_base.json）可无缝运行，无参数缺失错误
- [ ] 所有现有 debug metrics 正常输出
- [ ] 评估 pipeline 无变化，结果格式与之前兼容

---

## 代码质量
- [ ] 所有新增代码有清晰的注释和文档字符串
- [ ] 配置参数在 config_schema.py 中正确定义
- [ ] 代码风格与现有代码库一致
- [ ] 无硬编码的魔法数字（通过 config 控制）
- [ ] 错误处理完善，异常情况有明确提示
