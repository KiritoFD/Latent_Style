# Schrödinger Bridge 动力学：条件期望坍缩与Style信号强度

Date: 2026-06-21

## 1. 核心问题：模型学到了什么？

### 1.1 问题的重新表述

梯度探针实验发现了一个关键矛盾：

- **Loss landscape** 的全局极小值在 $s=1.0$（正确解），不在 $s=0.16$
- **但模型** 的实际输出在 $s=0.16$（endpoint shrinkage）

这意味着：**训练的收敛点不是loss landscape的极小值**。这个矛盾迫使我们必须重新审视模型的学习动态。

### 1.2 条件期望坍缩假说

**核心假说**：模型学到的是**条件期望** $E[v_{\mathrm{target}} \mid x, t]$ 而非 style-specific 的 $v_{\mathrm{target}}(x, t, \mathrm{style})$。

$$
v_{\mathrm{model}}(x, t, \mathrm{style}) \approx E_{y \sim p(y|x,t)}\left[\frac{y - x}{1-t}\right]
$$

当 style 信号不足以区分不同的 $y$ 时，模型退化为对 $y$ 的边际分布做平均。

---

## 2. 数学框架

### 2.1 Flow Matching 的目标

Schrödinger Bridge 的 Flow Matching loss 为：

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t, (x,y), \mathrm{style}}\left[\left\|v_\theta(x_t, t, \mathrm{style}) - v_{\mathrm{target}}(x, y, t)\right\|^2\right]
$$

其中：
- $x_t = \mathrm{interpolate}(x, y, t)$ 是中间表示
- $v_{\mathrm{target}} = \frac{y - x_t}{1-t}$ 是目标 velocity
- $v_\theta$ 是模型预测的 velocity

### 2.2 最优解的条件期望形式

给定足够大的模型容量，FM loss 的最优解为：

$$
v^*(x_t, t, \mathrm{style}) = \mathbb{E}_{y \mid x_t, t, \mathrm{style}}\left[\frac{y - x_t}{1-t}\right]
$$

**关键**：这个期望是在 $(x_t, t, \mathrm{style})$ 条件下的。如果 style 信号太弱，条件退化为：

$$
v^*(x_t, t, \mathrm{style}) \approx \mathbb{E}_{y \mid x_t, t}\left[\frac{y - x_t}{1-t}\right]
$$

即 style 条件被边缘化掉。

### 2.3 条件边缘化的后果

设 $v_{\mathrm{specific}}(x_t, t, s) = \frac{y_s - x_t}{1-t}$ 为 style-specific 的目标 velocity，$v_{\mathrm{marginal}}(x_t, t) = E_{s}[v_{\mathrm{specific}}(x_t, t, s)]$ 为边际期望。

则：

$$
\|v_{\mathrm{marginal}}\| \leq \mathbb{E}_s[\|v_{\mathrm{specific}}\|]
$$

且对于互相"抵消"的 style 方向：

$$
\|v_{\mathrm{marginal}}\| \ll \mathbb{E}_s[\|v_{\mathrm{specific}}\|]
$$

**直观理解**：不同 style 的目标 velocity 指向不同方向，平均后互相抵消，导致边际 velocity 的幅度远小于 style-specific velocity。

---

## 3. Shrinkage 的动力学来源

### 3.1 两阶段动力学

模型训练可以分解为两个阶段：

**阶段 1（快速收敛）**：模型学习 $E[v_{\mathrm{target}} \mid x_t, t]$（边际期望）
- 收敛速度：由 FM loss 主导，$\tau_1 \approx 1/(2\alpha)$
- 收敛目标：$v_{\mathrm{marginal}}$

**阶段 2（缓慢微调）**：模型学习 $E[v_{\mathrm{target}} \mid x_t, t, \mathrm{style}]$（条件期望）
- 收敛速度：由 style 信号强度决定，$\tau_2 \gg \tau_1$
- 收敛目标：$v_{\mathrm{specific}}$

### 3.2 观测到的 Shrinkage 的解释

| 时间步 $t$ | $v_{\mathrm{marginal}}$ 幅度 | $v_{\mathrm{specific}}$ 幅度 | 观测 $\alpha$ |
|-----------|---------------------------|---------------------------|------------|
| 0.0 | 小（不同 style 的 $y$ 差异大） | 大 | 0.16 |
| 0.5 | 中等（$x_t$ 已向 target 靠拢） | 中等 | 0.56 |
| 0.875 | 大（$x_t$ 接近 target，差异小） | 接近 marginal | 0.90 |

**解释**：
- $t=0$：$x_t$ 完全由 source 决定，不同 target 的 $y$ 差异巨大，边际 velocity 被严重平均化，$\alpha \approx 0.16$
- $t=0.875$：$x_t$ 已经接近 target，$y$ 的分布很窄，边际 velocity 接近 style-specific velocity，$\alpha \approx 0.90$

### 3.3 为什么不是普通的 FM 训练问题？

标准的 FM 训练（如 Stable Diffusion 3）中，条件（text）通常足够强，模型可以区分不同的条件。但本模型中的 style 条件通过 cross-attention gate 注入，初始 gate=0.05。

**gate=0.05 意味着**：
- 在训练的早期，style 信号只占 5% 的 cross-attention 输出
- 95% 的 cross-attention 输出是 style-agnostic 的 residual
- 模型在阶段 1 快速学到了 $v_{\mathrm{marginal}}$
- 阶段 2 的 style 微调几乎无法进行（因为 gate 太小，style 梯度太弱）

---

## 4. Style 信号强度的定量分析

### 4.1 Gate 机制

Cross-attention block 的输出：

$$
h_{\mathrm{out}} = h_{\mathrm{in}} + \mathrm{gate} \cdot \mathrm{CrossAttn}(h_{\mathrm{in}}, \mathrm{style\_tokens})
$$

其中 $\mathrm{gate}$ 是可学习的标量参数，初始化为 $g_0$。

### 4.2 Style 信号对 Velocity 的影响

通过链式法则：

$$
\frac{\partial v}{\partial \mathrm{style}} = \frac{\partial v}{\partial h_{\mathrm{out}}} \cdot \frac{\partial h_{\mathrm{out}}}{\partial \mathrm{CrossAttn}} \cdot \frac{\partial \mathrm{CrossAttn}}{\partial \mathrm{style}}
$$

$$
\frac{\partial h_{\mathrm{out}}}{\partial \mathrm{CrossAttn}} = \mathrm{gate} \cdot I
$$

因此：

$$
\frac{\partial v}{\partial \mathrm{style}} \propto \mathrm{gate}
$$

**gate=0.05 时，style 梯度被衰减 20 倍**。

### 4.3 Gate 的动力学

Gate 在训练中会增长（因为 style-specific 的 velocity 可以降低 FM loss），但增长速度受限于：

1. **初始值**：$g_0 = 0.05$，需要从很小的值开始增长
2. **梯度竞争**：FM loss 的梯度主要推动 $v$ 向 $v_{\mathrm{marginal}}$ 收敛，style 信号的梯度是二阶的
3. **局部最优**：$v_{\mathrm{marginal}}$ 已经是一个合理的解（FM loss 不大），模型可能不会主动增大 gate

**预测**：gate=0.3 初始化的模型，style 梯度信号强 6 倍，模型在阶段 1 就能学到显著的 style-specific 信息。

---

## 5. 系统动力学：Gate、Head 容量、Skip 连接的相互作用

### 5.1 动力学系统

将模型看作一个动力学系统，状态变量为：
- $\theta$：模型参数
- $g$：gate 值
- $v$：预测 velocity

系统方程为：

$$
\begin{aligned}
\frac{d\theta}{dt} &= -\eta \cdot \nabla_\theta \mathcal{L} \\
\frac{dg}{dt} &= -\eta \cdot \nabla_g \mathcal{L} \\
\frac{d}{dt}\|v\| &= \frac{d}{dt}\|E[v_{\mathrm{target}} \mid \cdot]\|
\end{aligned}
$$

### 5.2 相图分析

系统有两个吸引子：

**吸引子 A（平凡解）**：
- $g \approx 0$：style 信号被抑制
- $v \approx v_{\mathrm{marginal}}$：边际 velocity
- $\|v\| \ll \|v_{\mathrm{target}}\|$：shrinkage 严重
- FM loss 中等，SWD loss 中等
- **稳定性**：高（gate 小 → style 梯度小 → gate 不增长 → 正反馈锁死）

**吸引子 B（正确解）**：
- $g \gg 0$：style 信号充分
- $v \approx v_{\mathrm{specific}}$：style-specific velocity
- $\|v\| \approx \|v_{\mathrm{target}}\|$：无 shrinkage
- FM loss 低，SWD loss 低
- **稳定性**：中（gate 大 → style 梯度大 → 微调生效）

### 5.3 分岔分析

从吸引子 A 到吸引子 B 的转变需要 gate 超过某个临界值 $g_{\mathrm{crit}}$：

$$
g_{\mathrm{crit}} \approx \frac{\|\nabla_\theta \mathcal{L}_{\mathrm{FM}}\|_{\mathrm{marginal}}}{\|\nabla_\theta \mathcal{L}_{\mathrm{FM}}\|_{\mathrm{style}}}
$$

当 $g < g_{\mathrm{crit}}$ 时，FM loss 的梯度主要来自边际 velocity 的误差，style 信号被淹没。

当 $g > g_{\mathrm{crit}}$ 时，style-specific 的梯度超过边际梯度，系统开始向吸引子 B 移动。

**预测**：$g_0 = 0.3$ 可能已经超过 $g_{\mathrm{crit}}$，允许模型从训练初期就学习 style-specific 信息。

### 5.4 Endpoint Head 容量的作用

大容量 endpoint head（3 层 Conv2d，无 GroupNorm）的作用：

1. **表达能力**：更大的容量允许模型表示更复杂的 style-specific 映射
2. **无 GroupNorm**：不压缩动态范围，允许更大的 endpoint 位移
3. **非零初始化**：$\mathrm{normal}(0, 0.02)$ 而非 $\mathrm{normal}(0, 10^{-3})$，初始输出不会坍缩到零

### 5.5 NSWD 的作用

NSWD 在系统动力学中扮演**扰动**角色：

- 打破 SWD 梯度的正交性，提供额外的方向信号
- 增大 loss landscape 的曲率，加速收敛
- 但**不是主要驱动力**：即使 NSWD 完美工作，如果 style 信号太弱，模型仍会落入条件期望坍缩

**关键结论**：NSWD 是辅助修复，gate=0.3 + 大容量 head 是核心修复。

---

## 6. 与现有理论的整合

### 6.1 修正后的理论框架

```
白化/雾化的根本原因链条：

gate=0.05 (style信号弱)
    ↓
条件期望坍缩: v_model ≈ E[v_target | x, t]
    ↓
不同style的velocity方向互相抵消
    ↓
||v_marginal|| << ||v_specific||
    ↓
Endpoint Shrinkage: α ≈ 0.16 at t=0
    ↓
图像白化/雾化
```

### 6.2 修复策略的优先级

| 优先级 | 修复 | 原理 | 预期效果 |
|--------|------|------|----------|
| **P0** | gate=0.3 | 增强 style 信号，打破条件期望坍缩 | 核心修复 |
| **P0** | 大容量 endpoint head | 表达能力 + 无 GroupNorm 压缩 | 核心修复 |
| **P1** | NSWD (σ=0.02) | 打破 SWD 梯度正交性 | 辅助修复 |
| **P2** | StyleFiLM | 额外的 style 注入路径 | 增强修复 |
| **P3** | Velocity scale loss | 直接约束 shrinkage | 安全网 |

### 6.3 失败的修复为什么失败

所有之前的修复失败，因为它们都试图绕过条件期望坍缩问题：

- **lowfreqfix**：惩罚低频动态，但模型仍输出 $v_{\mathrm{marginal}}$
- **target_linear**：改变路径，但没改变 style 信号强度
- **endpointaux**：source-endpoint loss，但 endpoint 仍是 $E[y|x,t]$ 而非 $y_s$
- **direction loss**：强制方向正确，但模型无法区分 style，只能输出零

---

## 7. 实验验证

### 7.1 当前实验（进行中）

| 配置 | Gate | NSWD σ | Endpoint Head | 预期 |
|------|------|--------|---------------|------|
| 620_nswd_gate03_smoke | 0.3 | 0.02 | 3层, 无GN | 核心修复 |
| 620_nswd_s01_smoke | 0.3 | 0.01 | 3层, 无GN | 弱NSWD |
| 620_nswd_s005_smoke | 0.3 | 0.05 | 3层, 无GN | 强NSWD |

### 7.2 验证指标

**条件期望坍缩的验证**：
- 对不同 style 的 $(x, t)$，模型输出的 velocity 方向是否不同？
- 计算 $\cos(v_{\mathrm{model}}(x, t, s_1), v_{\mathrm{model}}(x, t, s_2))$ 对于 $s_1 \neq s_2$
- 如果 gate 太小，cos 应该接近 1（所有 style 输出相似）

**Style sensitivity**：
- 测量 $\frac{\|v(x, t, s_1) - v(x, t, s_2)\|}{\|v_{\mathrm{target}}(x, s_1) - v_{\mathrm{target}}(x, s_2)\|}$
- gate=0.3 时预期 > 0.5（至少 50% 的 style-specific 响应）

### 7.3 验收标准

- Endpoint $\alpha(t=0) \geq 0.5$（从 0.16 提升）
- Style sensitivity $\geq 0.5$
- $\cos(v_{\mathrm{model}}(s_1), v_{\mathrm{model}}(s_2))$ 显著 < 1.0
- WFI score $\leq 0.20$（Seedream IDT 水平）

---

## 8. 附录：条件期望与Shrinkage的定量关系

### 8.1 假设

设有 $K$ 个 style，每个 style $s$ 的 target velocity $v_s$ 满足：

$$
v_s = v_{\mathrm{mean}} + \delta_s
$$

其中 $v_{\mathrm{mean}} = \frac{1}{K}\sum_s v_s$，$\delta_s$ 是 style-specific 的偏差，$\sum_s \delta_s = 0$。

### 8.2 边际 velocity

$$
v_{\mathrm{marginal}} = \mathbb{E}_s[v_s] = v_{\mathrm{mean}} = \frac{1}{K}\sum_s v_s
$$

### 8.3 Shrinkage factor

对于 style $s$：

$$
\alpha_s = \frac{\|v_{\mathrm{marginal}}\|}{\|v_s\|} = \frac{\|v_{\mathrm{mean}}\|}{\|v_{\mathrm{mean}} + \delta_s\|}
$$

**极端情况**：
- 若 $\delta_s$ 在各 style 间完全对称（$\|\delta_s\| = \|\delta_{s'}\|$ 且方向均匀分布），则 $v_{\mathrm{mean}} \approx 0$，$\alpha \approx 0$
- 若所有 style 的 $v_s$ 都指向同一方向，则 $v_{\mathrm{mean}} \approx v_s$，$\alpha \approx 1$

**620 的情况**：5 个风格（Early Renaissance, Impressionism, Minimalism, Rococo, Ukiyo-e），风格差异大，$\delta_s$ 的方向分散，$v_{\mathrm{mean}}$ 的幅度远小于 $v_s$，导致 $\alpha \approx 0.16$。