# 620 总结 — 当前状态与下一步

> 最后更新: 2026-06-20

## 当前最优

**swd16, vl=0.04, epoch=5**: clip_style=0.7051, content_lpips=0.2935

| 对比 | clip_style | delta |
|------|:---:|:---:|
| IDT baseline | 0.6399 | — |
| 619 全部 7 组最优 | 0.670 | +0.030 |
| **620 swd16** | **0.705** | **+0.065** |
| 目标 | 0.720 | 差 0.015 |

## 为什么 620 突破了 0.67 天花板

三个致命缺陷同时被修复 (不是渐进改善, 是结构性修复):

| 619 缺陷 | 旧代码 | 620 新方案 | 效果 |
|---------|--------|-----------|------|
| OT 在线→均值坍缩 | minibatch Sinkhorn, target 每 batch 变 | DINO CLS 离线 top-K 固定配对 | 目标稳定, $v^*$ 非条件期望 |
| 伪 CrossAttn→1D 瓶颈 | learned tokens + 1D bias (KB 级信息量) | DINOv2 256×384 空间特征→ K,V (400KB) | 信息量 ×100 |
| ODE 展开→梯度截断 | `integrate()` N 步, clamp/nan_to_num | 单步 SWD: `SWD(ẑ₁, z_s)` | 梯度链长 1, 稳定 |

## Phase 4 方向

不调参. 改架构. 三个信息流杠杆 + 一个表征设计:

| 杠杆 | 核心问题 | 实验 |
|------|---------|------|
| Skip 信号比 | content直通路径太强→淹没了风格信号 | per-layer衰减, 可学习门控, skip上CrossAttn |
| CrossAttn Q来源 | Q决定attention的"视角"—content vs style驱动 | Q=bottleneck / Q=DINO / Q=concat |
| 多分辨率CrossAttn | 粗尺度需要全局(1D), 细尺度需要空间(256 tokens) | 分尺度CrossAttn + local encoder |
| 风格Encoder设计 | DINO单层 vs 多尺度 vs DINO+可训练local | 3 条路线 |
| Per-region SWD | 全局SWD混在一起→模型看不清"区域匹配" | 多尺度SWD, attention-weighted SWD |
| Attention 稀疏化 | softmax太软→精确笔触匹配不精准 | top-k, entropy正则 |
| OT 配对优化 | 固定配对 vs 轮转 vs 结构复杂度匹配 | 3 条路线 |

详见: `info_flow_analysis.md` (理论), `phase4_plan.md` (23实验, 7 blocks, 14h)

## 文档索引

| 文件 | 内容 |
|------|------|
| `math.md` | 数学基础: 均值坍缩定理, DPI, Pareto前沿 |
| `OT.md` | OT配对设计: DINO离线预配对 |
| `bridge.md` | 桥动力学: 单步SWD, SDE推理 |
| `tokenizer.md` | 风格表征: True Cross-Attention |
| `info_flow_analysis.md` | 信息流4问: OT/Attn/SWD/Encoder |
| `phase4_plan.md` | Phase4实验计划 |
| `experiment_progress.md` | 实验进度 |
| `convergence_diagnosis.md` | 收敛诊断protocol |
