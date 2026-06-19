# 620 深度探索计划 — 16h 架构级实验

> 当前: swd16, vl=0.04, e5 → style 0.705, LPIPS 0.294
> 目标: style ≥ 0.72
> 策略: 不调参, 改架构. 16h ≈ 24-32 experiments (30 min各, 含收敛诊断)

---

## 原则

1. **每个实验测试一条独立的结构假说** — 不是参数扫描
2. **虚拟长度 vl=0.04 固定** — 已证明是最优收敛粒度
3. **每次改一个变量** — 可归因
4. **每 4-6 个实验后做一次判断** — 决定下个方向

---

## Phase A: Style Encoder 架构 (6 experiments, ~3h)

**核心假说**: DINOv2 提供了强的风格表征基础, 但冻结 vs 微调 vs 替换, 以及特征层选择, 会显著影响最终 style 上限.

### A1: DINOv2 层选择

DINOv2 有 12 层 transformer. 浅层=纹理, 深层=语义. 当前用第 8 层 (中间). 对比:

| 实验 | 层 | 假说 |
|------|:---:|------|
| A1a | 第 4 层 | 浅层纹理信息更丰富 → style 更高 |
| A1b | 第 11 层 | 深层语义信息更强 → 全局风格一致性好 |
| A1c | 第 4+8+11 拼接 | 多尺度风格表征 → 最佳 |

### A2: DINOv2 微调

当前完全冻结. 最后 2 层微调→针对 WikiArt 风格域适配.

| 实验 | 微调范围 | 假说 |
|------|---------|------|
| A2a | 冻结全部 (基线) | — |
| A2b | 最后 2 个 transformer block 可训练 | 适配 WikiArt 风格分布 |
| A2c | 最后 4 个 block 可训练 + low lr (5e-5) | 更大的适配能力 |

**判据**: 如果 A2b/A2c 在 e3 时 style > A2a 超过 0.01 → 微调有收益, 进入 Phase B 时沿用.

---

## Phase B: Cross-Attention 设计 (8 experiments, ~4h)

**核心假说**: True Cross-Attention 是风格信息的唯一入口. 它的架构 (头数, 层数, 注入位置, 残差方式) 直接决定模型能"看见"多少风格细节.

### B1: 注入位置

当前可能只在 bottleneck 做 CrossAttn:

| 实验 | 注入位置 | 假说 |
|------|---------|------|
| B1a | bottleneck only (基线) | — |
| B1b | 所有 decoder blocks | 多尺度风格注入, 粗尺度色调+细尺度笔触 |
| B1c | encoder + decoder | 编码时就融合风格 |
| B1d | 仅 decoder 的最后 3 层 | 轻量, 减少内容泄漏 |

### B2: Multi-Head 数量

| 实验 | heads | 假说 |
|------|:---:|------|
| B2a | 4 (基线) | — |
| B2b | 8 | 更多注意力头→更精细的风格匹配 |
| B2c | 12 | 最大匹配灵活性 |

### B3: 残差连接方式

CrossAttn 的输出如何与 content feature 融合:

| 实验 | 融合方式 | 假说 |
|------|---------|------|
| B3a | `h + attn_output` (基线) | — |
| B3b | `h + gate × attn_output`, gate 可学习 | 自适应风格强度 |
| B3c | `(1-α)×h + α×attn_output`, α=0.3 | 固定偏置风格强度 |

**判据**: B1b+B2b+B3b 组合可能最优. 但需要实验验证每项独立贡献.

---

## Phase C: 预配对策略 (6 experiments, ~3h)

**核心假说**: 均值坍缩定理揭示配对稳定性是关键. DINO top-k 当前固定配对 → 但 K 值大小、轮转策略、是否混合随机配对, 都会影响模型的"风格多样性 vs 目标稳定性"tradeoff.

### C1: DINO Top-K 大小

| 实验 | K | 每 content 候选 target 数 |
|------|:---:|------|
| C1a | 1 (当前固定) | 完全确定, 风险: 过拟合特定 target |
| C1b | 5 | 轮转 5 个候选, 增加多样性 |
| C1c | 10 | 更多候选, 接近"软配对" |
| C1d | 20 | 接近 Independent Coupling 的上限 |

### C2: 混合策略

| 实验 | 策略 | 假说 |
|------|------|------|
| C2a | 100% DINO top-1 | 基线 |
| C2b | 80% DINO top-1 + 20% random | 最优探索-利用平衡 |
| C2c | 50% DINO top-1 + 50% random | 更强的泛化 |

**判据**: 如果 C1b (K=5) 在 style 和 LPIPS 上都优于 C1a (K=1) → 固定配对确实限制多样性. 如果 style 提升 < 0.005, 则配对策略非当前瓶颈.

---

## Phase D: 单步 Supervisory Signals (6 experiments, ~3h)

**核心假说**: SWD=16 提供了强风格信号. 但还有其他"不展开 ODE"的监督信号可以加入, 进一步强化风格学习.

### D1: 多尺度 SWD

当前 SWD 在整体 latent 上算. 改为不同 scale 分别算:

```python
L_swd_multi = SWD(z_hat, z_s) + SWD(down(z_hat, 2), down(z_s, 2)) + SWD(down(z_hat, 4), down(z_s, 4))
```

| 实验 | SWD scales | 假说 |
|------|-----------|------|
| D1a | [1] (基线, 64×64) | — |
| D1b | [1, 0.5] (64+32) | 多尺度分布匹配 |
| D1c | [1, 0.5, 0.25] (64+32+16) | 最全面 |

### D2: Content-Structure Edge Loss

Content 的边缘结构应该在风格化后保持. 用单步预测的 ẑ₁ 和 z_c 比较边缘:

```python
L_edge = L1(Sobel(ẑ₁), Sobel(z_c))
```

| 实验 | λ_edge | 假说 |
|------|:---:|------|
| D2a | 0 (基线) | — |
| D2b | 0.5 | 轻量边缘保持 |
| D2c | 1.0 | 强边缘保持 |

### D3: Style Feature Consistency

用 DINOv2 比较 ẑ₁ 和 z_s 的特征一致性 (不是 latent 级别, 是特征级别):

```python
L_feat = L2(DINO(ẑ₁_decoded), DINO(z_s_decoded))
```

| 实验 | λ_feat | 假说 |
|------|:---:|------|
| D3a | 0 (基线) | — |
| D3b | 0.1 | 特征级风格对齐 |

**判据**: D1c+D2b+D3b 组合可能把 style 推到 0.72. 但每个都要独立验证.

---

## Phase E: 组合验证 (4 experiments, ~2h)

取 Phase A-D 中各自最优配置, 组合测试:

| 实验 | 组合 | 预期 style |
|------|------|:---:|
| E1 | 基线 (当前最优) | 0.705 |
| E2 | A最优 + B最优 | 0.710-0.715 |
| E3 | A+B+C最优 | 0.715-0.720 |
| E4 | A+B+C+D最优 (全组合) | 0.720+ |

---

## 决策树

```
Phase A 完成 (6 exps):
  A2b style > A2a +0.01? 
    → YES: 后续都冻结 DINO, 用微调版本继续
    → NO:  冻结 DINO, 进入 Phase B

Phase B 完成 (8 exps):
  B1b (all decoder) 最优?
    → YES: 后续都注入所有 decoder 层
    → NO:  用各自最优注入位置

Phase C 完成 (6 exps):
  C1b (K=5) style > C1a (K=1) +0.005?
    → YES: 轮转 5 个候选
    → NO:  保持固定配对, 非瓶颈

Phase D 完成 (6 exps):
  D1c best? 进入 E 时用多尺度 SWD
  D2/D3 如果单个 LPIPS 不变而 style 升 >0.003 → 保留

Phase E 完成 (4 exps):
  E4 style ≥ 0.72? → 目标达成
  E4 style 0.715-0.72? → 进入 Phase F (备选: 增大 lr / NFE / batch)
```

---

## 时间预算

| Phase | 实验数 | 预估时间 |
|-------|:---:|------|
| A: Style Encoder | 6 | ~3h |
| B: Cross-Attention | 8 | ~4h |
| C: 预配对 | 6 | ~3h |
| D: Supervisory | 6 | ~3h |
| E: 组合 | 4 | ~2h |
| **总计** | **30** | **~15h** |

每个实验 30min (vlen=0.04 × 6epochs × 2.5min/epoch + eval).
