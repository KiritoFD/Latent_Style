# 620 Experiment Progress Summary

> 最后更新: 2026-06-20

## 当前最优

**swd16, vl=0.04, epoch=5**: clip_style=**0.7051**, content_lpips=0.2935
超过 IDT (+0.065), 超过 SaMAM (+0.090), 突破历史 0.67 天花板.

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
