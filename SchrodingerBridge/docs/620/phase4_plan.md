# 620 Phase 4 — 信息流驱动的架构实验

> 基于 `info_flow_analysis.md` 的 4 个理论问题.
> 当前: SWD16, style=0.705. 目标: 0.72. 16h.

---

## 实验矩阵

每个实验的"理论依据"列来自 info_flow_analysis.md.

### Block A: Style Encoder 多尺度 (3 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| A1 | DINO 单层 layer8 (基线) | — | — |
| A2 | DINO layers [4,8,11] concat → K,V | Q3 路线A: 多尺度特征, 浅层纹理+深层语义 | style +0.005-0.01 |
| A3 | DINO layer8 + Trainable LocalCNN | Q3 路线B: DINO全局+可训练局部高频 | 笔触锐利度提升 |

### Block B: Per-Region SWD (3 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| B1 | 全局 SWD (基线) | — | — |
| B2 | 2-scale SWD (64×64 + 32×32) | Q2 路线B: 多尺度分布匹配 | style +0.003-0.008 |
| B3 | 3-scale SWD (64+32+16) | Q2 路线B 扩展 | 最全面分布匹配 |
| B4 | Attention-weighted SWD | Q1 路线C + Q2 路线C: 高attention区域重加权 | LPIPS可能保持更好 |

### Block C: Skip 连接信号比 (4 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| C1 | α=1.0 (基线) | — | — |
| C2 | α=[1.0, 0.7, 0.5, 0.3] per-layer | 杠杆1: 粗尺度保结构, 细尺度放风格 | style↑ 0.01, LPIPS↑ 0.02 |
| C3 | α=0.5 all layers | 杠杆1: 激进 | style↑↑ 0.02, LPIPS↑↑ |
| C4 | 可学习门控 α=sigmoid(w) per-layer | 杠杆1: 模型自主决定 | 可能最优balance |

### Block D: Cross-Attention 架构 (4 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| D1 | Q=concat(skip,bottleneck) (基线) | — | — |
| D2 | Q=bottleneck only | 杠杆2: 风格驱动的视角 | style↑ |
| D3 | Q=proj(DINO features) | 杠杆2激进: 完全风格驱动 | style↑↑, risk LPIPS↑↑ |
| D4 | 粗尺度 CrossAttn 不做, 仅细尺度做 | 杠杆3: 省算力, 目标明确 | 可能不降 style 省 VRAM |

### Block E: Attention 稀疏化 (3 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| E1 | Softmax attention (基线) | — | — |
| E2 | Top-k attention (k=16 per Q) | Q4 路线B: 硬稀疏化, 每Q只关注top风格token | 匹配更精准 |
| E3 | Attention entropy 正则 (λ=0.01) | Q4 路线B: 鼓励低熵, 高confidence匹配 | 渐进式精准化 |

### Block F: OT 配对优化 (3 experiments, ~2h)

| ID | 实验 | 理论 | 预期 |
|----|------|------|------|
| F1 | DINO top-1 固定 (基线) | — | — |
| F2 | DINO top-5 轮转 | Q1 讨论: K=5 减少过拟合 | style多样性↑ |
| F3 | attention complexity matching | Q1 路线B: 复杂图配复杂画 | OT 配对质量↑ |

### Block G: 组合 (2 experiments, ~2h)

取 A-G 各最优 → G1. 取 G1 + 微调 → G2.

---

## 决策树

```
A done: A2 style > A1 +0.005? → YES=保留多尺度DINO
B done: B2 or B4 best? → 保留
C done: C2 or C4 best? → 保留
D done: D2 best? → 保留 Q=bottleneck. D3 LPIPS ok?→可能保留
E done: E2 best? → 保留 top-k. E3 too slow?→跳过
F done: F2 style > F1? → 保留轮转配对
G: G1 style ≥ 0.72? → 成功. 否则→提高lr/batch
```

---

## 时间预算

| Block | 实验 | 时间 |
|-------|:---:|------|
| A: Encoder | 3 | ~2h |
| B: Per-Region SWD | 4 | ~2h |
| C: Skip 信号比 | 4 | ~2h |
| D: CrossAttn | 4 | ~2h |
| E: Attention 稀疏 | 3 | ~2h |
| F: OT 配对 | 3 | ~2h |
| G: 组合 | 2 | ~2h |
| **合计** | **23** | **~14h** |
