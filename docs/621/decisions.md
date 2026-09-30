# 621 决策台账：哪些有用/哪些该删

> 建立日期: 2026-06-21  
> 基于全分支实验考古和当前620状态

---

## 1. 架构级决策

### 1.1 保留的核心架构

| 组件 | 状态 | 理由 |
|------|------|------|
| 620 SpatialBridge (transformer blocks) | ✅ 保留 | 比legacy LANCET更简洁，性能更好 |
| DINO cross-attention style injection | ✅ 保留 | 突破0.67天花板的关键 |
| 单步SWD loss | ✅ 保留 | 消除ODE展开的梯度问题 |
| Endpoint prediction (vs velocity直接) | ✅ 保留 | 更稳定的训练目标 |
| FiLM endpoint head | ✅ 保留 | 白化修复的核心 |
| target_linear training target | ✅ 保留 | 低频路径的正确方式 |

### 1.2 可以删除的组件

| 组件 | 状态 | 理由 |
|------|------|------|
| Legacy LANCET U-Net backbone | ⚠️ 可归档 | 620已完全替代，保留代码但不主动维护 |
| Terminal SWD (多步ODE展开) | ❌ 删除 | 被单步SWD替代，梯度问题 |
| OT online coupling (Sinkhorn) | ❌ 删除 | 被DINO离线预配对替代 |
| Cycle consistency loss | ❌ 删除 | 无显著收益，增加复杂度 |
| Target teacher (EMA) | ❌ 删除 | 620不需要 |
| Structure descriptors (GW, TopoGate) | ❌ 删除 | 结构loss无用(Classify分支验证) |
| Diff-Gram loss | ❌ 删除 | 极差(sdxl-fp32验证) |
| Gram-Moment matching | ❌ 删除 | 结果差(Gram-Moment分支验证) |

### 1.3 待评估的组件

| 组件 | 状态 | 评估条件 |
|------|------|----------|
| DINO多尺度 [4,8,11] | ⏳ 待测 | Phase4 Block A2 |
| Per-region SWD | ⏳ 待测 | Phase4 Block B |
| Skip α per-layer | ⏳ 待测 | Phase4 Block C |
| Cross-attention Q来源 | ⏳ 待测 | Phase4 Block D |
| Attention稀疏化 | ⏳ 待测 | Phase4 Block E |
| OT配对优化 | ⏳ 待测 | Phase4 Block F |
| Text conditioning | ⏳ 待测 | Phase4 Block C3 |

---

## 2. 实验级决策

### 2.1 有效实验 (保留/扩展)

| 实验 | 结果 | 决策 |
|------|------|------|
| SWD weight scan (12/16/20) | SWD=16最优 | 保留SWD=16 |
| velocity length scan (1.0/0.2/0.04) | vl=0.04最优 | 保留vl=0.04 |
| FiLM endpoint hd512 | WFI=0.3906过门 | **当前最优** |
| gate=0.3 | velocity_abs +16% | 保留gate=0.3 |
| StyleFiLM (block内) | film_gamma_abs增长 | 保留 |
| target_linear training | 早期有效 | 保留 |
| SWD noise σ=0.02 | 打破排序稳定性 | 保留 |

### 2.2 无效实验 (删除/归档)

| 实验 | 结果 | 决策 |
|------|------|------|
| lowfreqfix | velocity从0.15降到0.016 | ❌ 删除配置 |
| endpointaux | endpoint坍回source | ❌ 删除配置 |
| tlow (低t采样偏好) | 同上 | ❌ 删除配置 |
| endpoint_lowhigh (无FiLM) | style_sensitivity=0.003 | ❌ 删除配置 |
| endpoint_stylehead | style_sensitivity恢复但alpha仍负 | ❌ 删除配置 |
| direction loss | alpha=-0.007完全坍缩 | ❌ 删除配置 |
| gated_raw attention | WFI=0.64最差 | ❌ 删除配置 |
| relu2 attention | WFI=0.53仍白化 | ❌ 删除配置 |
| style_select attention | WFI=0.50无改善 | ❌ 删除配置 |
| Structure loss | "完全没用" | ❌ 删除loss项 |
| Diff-Gram | 极差 | ❌ 删除整个方向 |
| Gram-Moment | 差 | ❌ 删除整个方向 |

### 2.3 需要重新评估的实验

| 实验 | 原始结果 | 重新评估条件 |
|------|----------|-------------|
| HF residual | WFI=0.4746无效 | 组合FiLM hd512测试 |
| DINO adapter | 待定 | Phase4 Block A3 |
| intrinsic style (无DINO) | clip_style=0.6717 | 白化修复后重测 |
| sparsemax attention | 未充分测试 | Phase4 Block E |

---

## 3. 分支级决策

### 3.1 保留的分支

| 分支 | 理由 |
|------|------|
| codex/620-spatial-bridge | 当前开发分支 |
| main | 基线/tokenizer |
| SWD | 早期SWD实验记录 |
| attn | 3060适配记录 |

### 3.2 可归档的分支

| 分支 | 理由 |
|------|------|
| Classify | 分类器信号太强，结构loss无用 |
| Cycle-upscale | structure loss无用 |
| Diff-Gram | 极差 |
| Gram-Moment | 差 |
| Style8_Moment+SWD | 无显著优势 |
| sdxl-fp16 | 差 |
| re-SWD | 无显著优势 |
| multistep-texture | 有参考价值但非主线 |

### 3.3 需要特殊处理的分支

| 分支 | 处理 |
|------|------|
| Thermal | 风格好但质量差 → 提取LoRA思路 |
| exp/style-injection-priority-proto-sep | 参考style注入优先级 |

---

## 4. 开销分析

### 4.1 训练开销 (RTX 3060 12GB)

| 配置 | 1 epoch时间 | VRAM | 性价比 |
|------|------------|------|--------|
| 620 base (dim=128, 4 blocks) | ~45min | ~8GB | ⭐⭐⭐ |
| 620 + FiLM endpoint | ~50min | ~9GB | ⭐⭐⭐ |
| 620 + MoE (4 experts) | ~60min | ~10GB | ⭐⭐ |
| 620 + text conditioning | ~55min | ~9GB | ⭐⭐⭐ |

### 4.2 推理开销

| 配置 | 单图时间 | NFE | 备注 |
|------|---------|-----|------|
| 620 velocity (8步) | ~0.3s | 8 | 基线 |
| 620 endpoint (1步) | ~0.05s | 1 | 快速推理 |
| 620 CFG (3方向) | ~0.9s | 8×3 | 高质量 |

### 4.3 参数量

| 组件 | 参数量 | 占比 |
|------|--------|------|
| SpatialBridgeBlock ×4 | ~15M | 60% |
| StyleConditioner | ~5M | 20% |
| FiLM endpoint head | ~2M | 8% |
| 其他 | ~3M | 12% |
| **总计** | **~25M** | 100% |

---

## 5. 优先级排序

### P0: 白化修复 (必须)
1. FiLM endpoint hd512 ✅ 已完成
2. 无GN endpoint head → 待实现
3. Velocity scale loss → 待实现

### P1: 架构优化 (重要)
4. DINO多尺度 → Phase4 A2
5. Per-region SWD → Phase4 B
6. Skip α per-layer → Phase4 C

### P2: 性能提升 (可选)
7. Text conditioning → Phase4 C3
8. Attention稀疏化 → Phase4 E
9. OT配对优化 → Phase4 F

### P3: 清理 (低优先级)
10. 归档legacy LANCET
11. 清理无效实验配置
12. 压缩实验考古数据
