# Noisy Sliced Wasserstein Distance (NSWD): 打破排序稳定性的数学原理

Date: 2026-06-21

## 1. 问题回顾：SWD为什么梯度失效？

### 1.1 SWD的梯度结构

Sliced Wasserstein Distance定义为：

$$
\mathrm{SWD}(P, Q) = \frac{1}{K} \sum_{k=1}^{K} W_1(\mathrm{proj}_{\mathrm{dir}_k}(P),\; \mathrm{proj}_{\mathrm{dir}_k}(Q))
$$

其中 $W_1$ 是一维 Wasserstein-1 距离：

$$
W_1(P_1, Q_1) = \frac{1}{N} \sum_{i=1}^{N} \bigl| p_{(i)} - q_{(i)} \bigr|
$$

梯度：

$$
\frac{\partial \mathrm{SWD}}{\partial p_i} = \frac{1}{KN} \sum_{k} \mathrm{dir}_k \cdot \mathrm{sign}\!\left(p_{(\sigma(i))} - q_{(\sigma(i))}\right)
$$

其中 $\sigma$ 是排序排列。

### 1.2 排序稳定性导致梯度正交化

当排序排列 $\sigma$ 不变时，$\mathrm{sign}(\cdot)$ 恒定，$\nabla \mathrm{SWD}$ 为常数向量。这个常数向量与 $v_{\mathrm{target}}$ 的关系决定了梯度是否有用。

**关键实验发现**：
- $\|\nabla_v \mathrm{SWD}\| = 0.044$ （非零）
- $\cos(\nabla \mathrm{SWD}, v_{\mathrm{target}}) = -0.024$ （基本正交）
- 排序变化率 = 0% at all $\varepsilon \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$

**结论**：SWD梯度存在但方向与目标正交，且排序绝对稳定，梯度无法提供朝向正确方向的有效信号。

---

## 2. NSWD的数学定义

### 2.1 核心思想

在投影之前向样本添加高斯噪声，打破排序的确定性，使得排序排列随噪声采样而变化。

### 2.2 形式化定义

**标准 SWD**：

$$
\mathrm{SWD}(P, Q) = \mathbb{E}_{\mathrm{dir} \sim \mathcal{U}(S^{d-1})}\left[W_1(\mathrm{dir}^\top P,\; \mathrm{dir}^\top Q)\right]
$$

**Noisy SWD (NSWD)**：

$$
\boxed{\mathrm{NSWD}_\sigma(P, Q) = \mathbb{E}_{\mathrm{dir} \sim \mathcal{U}(S^{d-1}),\; \varepsilon \sim \mathcal{N}(0, \sigma^2 I)}\left[W_1(\mathrm{dir}^\top P + \varepsilon_P,\; \mathrm{dir}^\top Q + \varepsilon_Q)\right]}
$$

其中：
- $\varepsilon_P, \varepsilon_Q \sim \mathcal{N}(0, \sigma^2 I_N)$ 是独立的高斯噪声向量
- $\sigma$ 是噪声标准差（超参数）
- $N$ 是样本数量

### 2.3 实现

```python
def _sliced_wasserstein(a, b, dirs, noise_sigma=0.0):
    proj_a = a @ dirs.T  # [N, K]
    proj_b = b @ dirs.T  # [N, K]
    if noise_sigma > 0.0:
        proj_a = proj_a + noise_sigma * torch.randn_like(proj_a)
        proj_b = proj_b + noise_sigma * torch.randn_like(proj_b)
    # ... 排序 + W1 计算
```

噪声在每次前向传播时独立采样，期望值在SGD的mini-batch平均中自然实现。

---

## 3. 排序稳定性分析

### 3.1 无噪声情况

排序排列 $\sigma$ 在扰动 $\delta$ 下不变的条件（定理2）：

$$
\|\delta\|_\infty < \frac{1}{2} \min_{i \neq j} \bigl| p_{(i)} - p_{(j)} \bigr|
$$

对于 VAE 潜空间投影（$N=4096$，$\sigma_{\mathrm{proj}} \approx 0.2$）：

$$
\text{相邻间距} \approx \frac{\sigma_{\mathrm{proj}}}{\sqrt{N}} \approx \frac{0.2}{\sqrt{4096}} \approx 0.003
$$

实验发现排序稳定性远强于预期：即使 $\varepsilon = 0.01$（远大于估计的临界值 $0.003$），排序变化率仍为 0%。这意味着投影值分布存在聚类或重复值，导致相邻间距远大于独立同分布假设下的估计。

### 3.2 加噪声后

**定理 4（NSWD排序不稳定性）**：设 $p_{(1)} < p_{(2)} < \cdots < p_{(N)}$ 为排序后的投影值，相邻间距为 $\Delta_i = p_{(i+1)} - p_{(i)}$。添加噪声 $\xi_i \sim \mathcal{N}(0, \sigma^2)$ 后，$(i, i+1)$ 发生排序交换的概率为：

$$
P(\text{swap}_{i, i+1}) = \Phi\left(-\frac{\Delta_i}{\sigma\sqrt{2}}\right)
$$

其中 $\Phi$ 是标准正态分布的累积分布函数。

**证明**：

排序交换发生于 $p_i + \xi_i > p_{i+1} + \xi_{i+1}$，即 $\xi_i - \xi_{i+1} > p_{i+1} - p_i = \Delta_i$。

由于 $\xi_i - \xi_{i+1} \sim \mathcal{N}(0, 2\sigma^2)$：

$$
P(\text{swap}) = P\left(\mathcal{N}(0, 2\sigma^2) > \Delta_i\right) = \Phi\left(-\frac{\Delta_i}{\sigma\sqrt{2}}\right)
$$

$\square$

### 3.3 期望排序变化率

对所有相邻对求和：

$$
\mathbb{E}[\text{num\_swaps}] = \sum_{i=1}^{N-1} \Phi\left(-\frac{\Delta_i}{\sigma\sqrt{2}}\right)
$$

在 $\Delta_i$ 近似相等的情况下（$\Delta_i \approx \bar{\Delta}$）：

$$
\mathbb{E}[\text{num\_swaps}] \approx (N-1) \cdot \Phi\left(-\frac{\bar{\Delta}}{\sigma\sqrt{2}}\right)
$$

### 3.4 数值估计

| $\sigma$ | $\Phi(-\bar{\Delta}/(\sigma\sqrt{2}))$ | 期望交换数（$N=4096$） | 排序变化率 |
|----------|--------------------------------------|---------------------|-----------|
| 0.001 | $\Phi(-0.003/0.0014) \approx \Phi(-2.12) \approx 0.017$ | ~70 | ~1.7% |
| 0.005 | $\Phi(-0.003/0.0071) \approx \Phi(-0.42) \approx 0.337$ | ~1380 | ~34% |
| 0.010 | $\Phi(-0.003/0.014) \approx \Phi(-0.21) \approx 0.417$ | ~1700 | ~42% |
| 0.020 | $\Phi(-0.003/0.028) \approx \Phi(-0.11) \approx 0.456$ | ~1870 | ~46% |
| 0.050 | $\Phi(-0.003/0.071) \approx \Phi(-0.04) \approx 0.484$ | ~1980 | ~48% |

**关键观察**：
- $\sigma = 0.01$：约 42% 的排序变化率，已显著打破排序稳定性
- $\sigma = 0.02$：约 46% 的变化率
- $\sigma = 0.05$：接近 48%，接近完全随机排序（50%）

**但注意**：上述估计基于 $\bar{\Delta} \approx 0.003$。实验发现实际 $\Delta$ 可能更大（因为排序变化率在 $\varepsilon=0.01$ 时仍为 0%），因此实际需要的 $\sigma$ 可能比估计值更大。

---

## 4. NSWD的梯度性质

### 4.1 梯度期望

在NSWD中，$\nabla_p \mathrm{NSWD}(p, q)$ 是一个随机变量（因为 $\varepsilon$ 是随机的）。其期望：

$$
\mathbb{E}_\varepsilon[\nabla_p \mathrm{NSWD}(p, q)] = \frac{1}{KN} \sum_k \mathrm{dir}_k \cdot \mathbb{E}_\varepsilon\left[\mathrm{sign}\!\left(p_{(\sigma_\varepsilon(i))} - q_{(\sigma_\varepsilon(i))}\right)\right]
$$

其中 $\sigma_\varepsilon$ 是随机排序排列。

### 4.2 梯度方向分析

**无噪声情况**：$\nabla \mathrm{SWD}$ 是常数向量，方向由排序排列决定。

**有噪声情况**：排序排列随 $\varepsilon$ 变化，$\mathrm{sign}(\cdot)$ 在不同样本位置之间变化，导致：
- 梯度不再是常数向量
- 不同投影方向的梯度不会简单相加为固定方向
- 期望梯度方向可能更接近真实分布差异方向

### 4.3 方差-偏差权衡

$$
\mathrm{Var}_\varepsilon[\nabla_p \mathrm{NSWD}] = O\left(\frac{\sigma^2}{K}\right)
$$

- $\sigma$ 太小：排序变化不足，梯度仍接近常数，方差小但偏差大（梯度方向错误）
- $\sigma$ 太大：排序完全随机，梯度无偏但方差大，训练不稳定
- **最优 $\sigma$**：在排序变化率和梯度稳定性之间平衡

---

## 5. NSWD的Loss Landscape

### 5.1 平滑化效应

NSWD可以看作SWD的平滑版本：

$$
\mathrm{NSWD}_\sigma(P, Q) = \mathbb{E}_{\varepsilon}\left[\mathrm{SWD}(P + \varepsilon_P,\; Q + \varepsilon_Q)\right]
$$

这等价于对SWD做高斯核平滑，消除了排序稳定性带来的平坦区。

### 5.2 对Shrinkage的影响

在联合Loss $L(v) = \alpha\|v - v_{\mathrm{target}}\|^2 + \beta \cdot \mathrm{NSWD}(x + (1-t)v, y)$ 中：

- 无噪声SWD：$\nabla \mathrm{SWD} \approx C$（常数），与 $v_{\mathrm{target}}$ 正交
- NSWD：$\nabla \mathrm{NSWD}$ 方向随噪声变化，期望方向不再完全正交于 $v_{\mathrm{target}}$

**预测**：NSWD的SWD梯度方向与 $v_{\mathrm{target}}$ 的夹角余弦绝对值应大于无噪声情况。

---

## 6. 实验验证计划

### 6.1 Smoke Tests（进行中）

| 配置 | $\sigma$ | 预期 |
|------|---------|------|
| 620_nswd_s01_smoke | 0.01 | 弱打破排序稳定性 |
| 620_nswd_gate03_smoke | 0.02 | 中等打破排序稳定性 |
| 620_nswd_s005_smoke | 0.05 | 强打破排序稳定性 |

### 6.2 梯度探针对比

对每个 $\sigma$ checkpoint：
1. 测量 $\|\nabla \mathrm{NSWD}(v=0)\|$
2. 测量 $\cos(\nabla \mathrm{NSWD}, v_{\mathrm{target}})$
3. 测量排序变化率
4. 扫描1D loss landscape

### 6.3 验收标准

- $\cos(\nabla \mathrm{NSWD}, v_{\mathrm{target}})$ 显著大于 0.024（无噪声基线）
- Endpoint shrinkage $\alpha(t=0)$ 显著大于 0.16
- WFI score 不退化

---

## 附录：$\Phi$ 值表

| $z$ | $\Phi(-z)$ |
|-----|-----------|
| 0.00 | 0.500 |
| 0.10 | 0.460 |
| 0.20 | 0.421 |
| 0.50 | 0.309 |
| 1.00 | 0.159 |
| 1.50 | 0.067 |
| 2.00 | 0.023 |
| 2.50 | 0.006 |
| 3.00 | 0.001 |