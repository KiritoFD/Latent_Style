# 620 Experiment Progress Summary

> 最后更新: 2026-06-21

## 当前最优

**swd16, vl=0.04, epoch=5**: clip_style=**0.7051**, content_lpips=0.2935
超过 IDT (+0.065), 超过 SaMAM (+0.090), 突破历史 0.67 天花板.

## 白化/雾化问题修复状态

### 问题诊断

- Endpoint shrinkage: α=0.16 at t=0（模型只走了16%目标方向）
- 根因: gate=0.05 → style信号太弱 → 条件期望坍缩 → v_model ≈ E[v_target|x,t] → 不同style互相抵消
- SWD梯度平坦性: ∇SWD=0.044 但 cos(∇SWD, v_target)=-0.024（几乎正交）
- 理论文档: `docs/620/fog/theory/swd_flatness.md`, `trivial_solution.md`, `sb_dynamics.md`, `noisy_swd.md`

### 修复方案

| 优先级 | 修复 | 原理 | 状态 |
|--------|------|------|------|
| P0 | gate=0.3 | 增强style信号，打破条件期望坍缩 | ✅ 已实现 |
| P0 | 大容量endpoint head (3层, 无GroupNorm) | 表达能力+无动态范围压缩 | ✅ 已实现 |
| P1 | NSWD (σ=0.02) | 打破SWD梯度正交性 | ✅ 已实现 |
| P2 | StyleFiLM | 额外style注入路径 | 待实现 |
| P3 | Velocity scale loss | 直接约束shrinkage | 待实现 |

### Smoke Test结果 (gate=0.3, epoch=1)

| 指标 | gate=0.05 (旧) | gate=0.3 (新) | 变化 |
|------|---------------|---------------|------|
| gate_value | 0.064 | **0.297** | **+365%** |
| velocity_abs | 0.186 | **0.216** | **+16%** |
| all_pairs clip_style | 0.700 | 0.696 | -1% |
| clip_s_delta_idt | 0.060 | 0.056 | -7% |
| content_lpips | 0.272 | 0.293 | +8% |
| identity clip_style | 0.841 | 0.836 | -1% |

**结论**: gate修复成功，velocity magnitude提升（shrinkage减少），但1 epoch不够完成style-specific学习。5-epoch训练进行中。

## 突破原因

620 架构解决了 619 诊断的 3 个致命缺陷:

| 619 缺陷 | 620 解决 | 机制 |
|---------|---------|------|
| OT 在线不稳定→均值坍缩 | DINO 离线预配对 (每个content固定K个候选) | target固定, 跨epoch不跳变 |
| 伪CrossAttention→1D瓶颈 | True CrossAttn: DINOv2 256×384空间特征→K,V | 信息量 KB→400KB |
| ODE展开→梯度被clamp截断 | 单步SWD: `SWD(ẑ₁,z_s)` 替代integrate() | 消除梯度爆炸 |

## 已完成实验

### Phase 1: SWD Weight Scan

| SWD | vlen | best_epoch | clip_style | lpips |
|-----|------|-----------|-----------|-------|
| 12 | 1.0 | e8 | 0.6725 | 0.2968 |
| 16 | 1.0 | e1 | 0.7053 | 0.2901 |
| 16 | 0.2 | e9 | 0.7038 | 0.3064 |
| **16** | **0.04** | **e5** | **0.7051** | **0.2935** |
| 20 | 0.04 | e1 | 0.7006 | 0.2750 |

### Phase 3: Hyperparameter Tuning

| 实验 | 状态 |
|------|------|
| lr=1e-4 | ✅ 完成 |

## 下一阶段: Phase 4

详见 `phase4_plan.md` — 23个架构实验, 7个block, ~14h.
