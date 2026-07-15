# 620 整体动力学数学理论

> Round M 理论文档：建立从训练目标到推理轨迹的统一数学框架，明确白化/雾化的系统级可观测表现。

---

## 1. 状态变量定义

### 1.1 输入与目标

| 符号 | 维度 | 含义 | 来源 |
|------|------|------|------|
| $x = z_0$ | $\mathbb{R}^{B \times C \times H \times W}$ | source/content latent（VAE 编码后的源图潜变量） | 数据集 |
| $y = z_1$ | $\mathbb{R}^{B \times C \times H \times W}$ | target latent（目标风格图潜变量） | 数据集 |
| $s$ | — | 目标风格标识/条件 | 数据集/配置 |
| $t \in [0, 1]$ | $\mathbb{R}^B$ | 扩散/桥时间变量 | 训练采样 |

代码中 $C = 4$，$H = W = 64$（512 px 图像经 SD 1.5 VAE 下采样 8 倍）。

### 1.2 Style 条件表示

| 符号 | 维度 | 含义 | 代码对应 |
|------|------|------|----------|
| $S = \{s_i\}_{i=1}^{256}$ | $\mathbb{R}^{B \times 256 \times d}$ | DINO patch tokens（style tokens） | `style_tokens` |
| $s_{\text{global}}$ | $\mathbb{R}^{B \times d}$ | 全局 style 向量（DINO tokens 池化/投影） | `style_global` |
| $d$ | — | DINO 投影维度，默认等于 `base_dim`（64） | `StyleConditioner620` 输出 |

在 `model620.py` 中，`style_tokens, style_global = self.style_conditioner(...)`，随后被传入每个 `SpatialBridgeBlock620`。

### 1.3 中间状态与速度场

| 符号 | 维度 | 含义 | 代码对应 |
|------|------|------|----------|
| $x_t$ | $\mathbb{R}^{B \times C \times H \times W}$ | 训练时桥路径上的中间状态 | `_vertical_state` 输出 |
| $v_\theta(x_t, t, s)$ | $\mathbb{R}^{B \times C \times H \times W}$ | 神经网络预测的速度场 | `forward` 返回值 |
| $\hat{z}_1$ | $\mathbb{R}^{B \times C \times H \times W}$ | 单步预测的 endpoint | `z_hat1` |
| $\hat{z}_1^{\text{int}}$ | $\mathbb{R}^{B \times C \times H \times W}$ | 多步积分后的最终 latent | `integrate_transport` 输出 |

速度—端点关系（两种 head mode 统一）：

$$
\hat{z}_1 = x_t + (1 - t) \cdot v_\theta(x_t, t, s)
$$

在 `endpoint_head_mode == "velocity"` 时，网络直接输出 $v$，再构造 $\hat{z}_1$；在 `"endpoint_lowhigh"` 时，网络先预测 low/high endpoint delta，再反解 $v = (\hat{z}_1 - x_t) / (1 - t)$。

---

## 2. 训练时的演化路径

### 2.1 目标投影（Target Projection）

训练目标 $y$ 先经过 `_project_training_target` 投影为 $y_\text{proj}$，以控制 low-frequency 锚定程度：

$$
y_\text{proj} = \text{Project}(x, y; \lambda, \text{mode})
$$

其中 $\lambda \in [0, 1]$ 为 `training_target_projection_low_anchor`（代码中 `low_anchor`），mode 包括：

- `"source_low_target_high"`（默认）：低频锚定在 source，高频移向 target；
- `"target_linear"`：低频沿 $(1-t)c_\text{low} + t \cdot y_\text{low}$ 线性移动；
- `"pure_vertical_flow"` / wavelet 变体。

数学上，对任意 mode 都可以分解为低频与高频：

$$
c_\text{low} = L(x), \quad y_\text{low} = L(y), \quad c_\text{high} = x - c_\text{low}, \quad y_\text{high} = y - y_\text{low}
$$

其中 $L(\cdot)$ 为 `avg_pool2d` 低通滤波（kernel=5）。

### 2.2 桥状态 $x_t$

在 `"target_linear"` mode 下（当前最被证据支持的修复路径）：

$$
x_t = [(1-t)c_\text{low} + t \cdot y_\text{low}] + (1-t)c_\text{high} + t \cdot y_\text{high}
$$

等价地：

$$
x_t = x + t \cdot (y_\text{proj} - x)
$$

目标速度场：

$$
v_\text{target} = y_\text{proj} - x
$$

### 2.3 训练损失

总损失为四项加权和：

$$
\mathcal{L} = w_\text{FM} \cdot \mathcal{L}_\text{FM} + w_\text{SWD} \cdot \mathcal{L}_\text{SWD} + w_\text{edge} \cdot \mathcal{L}_\text{edge} + w_\text{low} \cdot \mathcal{L}_\text{low}
$$

其中：

1. **Flow Matching loss**：

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t, x, y, s}\left[ \| v_\theta(x_t, t, s) - v_\text{target} \|_2^2 \right]
$$

2. **Sliced Wasserstein Distance (SWD)**：

$$
\mathcal{L}_\text{SWD} = \mathbb{E}_{\{d_k\}}\left[ W_1\big( \{d_k^\top \hat{z}_1^{(i)}\}, \{d_k^\top y_\text{proj}^{(i)}\} \big) \right]
$$

其中 $d_k \sim \mathcal{S}^{D-1}$ 为随机投影方向，`num_projections=64`。可选 NSWD 时在投影值上加高斯噪声 $\sigma$：

$$
\tilde{p}^{(i)} = p^{(i)} + \sigma \cdot \varepsilon^{(i)}, \quad \varepsilon^{(i)} \sim \mathcal{N}(0, 1)
$$

3. **Edge loss**（高频 L1）：

$$
\mathcal{L}_\text{edge} = \| (\hat{z}_1 - L(\hat{z}_1)) - (y_\text{proj} - L(y_\text{proj})) \|_1
$$

4. **Low-frequency anchor loss**（可选）：

$$
\mathcal{L}_\text{low} = \| L(\hat{z}_1) - L(y_\text{proj}) \|_1
$$

### 2.4 训练时的关键可观测约束

训练目标在理想情况下要求：

$$
\hat{z}_1 \xrightarrow{\mathcal{L}} y_\text{proj}, \quad v_\theta \xrightarrow{\mathcal{L}} v_\text{target}
$$

但实际优化受以下结构约束：

- $v_\theta$ 的初始值接近 0（zero-init 或 small-init）；
- style 信号通过 `tanh(gate) \in [0, 1)` 缩放进入 trunk；
- 中间特征经过 GroupNorm / AdaLN，可能压缩动态范围；
- endpoint head 的 GroupNorm(1) 等价于 LayerNorm，会归一化特征均值/方差。

---

## 3. 推理时的演化路径

### 3.1 I2SB Solver

推理时调用 `integrate_transport`，执行 $N$ 步 I2SB（Iterative Schrödinger Bridge）更新：

对第 $i$ 步，$t_i = \frac{i}{N} \cdot \text{step\_size}$，$t_{i+1} = \frac{i+1}{N} \cdot \text{step\_size}$，有：

$$
\hat{z}_1^{(i)} = \text{predict\_endpoint}(h_i, t_i; s)
$$

$$
h_{i+1} = c_\text{curr} \cdot h_i + c_\text{tgt} \cdot \hat{z}_1^{(i)} + \sqrt{\text{var}} \cdot \varepsilon_i
$$

其中：

$$
c_\text{curr} = \frac{1 - t_{i+1}}{1 - t_i}, \quad c_\text{tgt} = \frac{t_{i+1} - t_i}{1 - t_i}
$$

$$
\text{var} = \sigma^2 \cdot \frac{(t_{i+1} - t_i)(1 - t_{i+1})}{1 - t_i}
$$

代码中默认 `bridge_sigma = 0.02`，`num_steps = 8`，`step_size = 1.0`。

### 3.2 推理与训练的关键差异

| 方面 | 训练 | 推理 |
|------|------|------|
| 输入 | $x_t$ 已沿 target 路径插值 | 始终从 source $x$ 出发（$t=0$） |
| 监督 | 有 $v_\text{target}$ 和 $y_\text{proj}$ 直接监督 | 无监督，仅依赖训练好的 $v_\theta$ |
| endpoint 使用 | 单步 $\hat{z}_1$ 用于算 loss | 多步 endpoint 用于构造桥更新 |
| 误差传播 | 单步梯度反传 | 多步累积，但无梯度 |
| 关键风险点 | $t$ 采样可能避开 $t \approx 0$ | 所有样本都经过 $t=0$ 端点预测 |

### 3.3 推理端点的显式形式

在 $t=0$ 时：

$$
\hat{z}_1^{(0)} = \text{predict\_endpoint}(x, t=0; s) = x + v_\theta(x, 0, s)
$$

后续积分每一步都重新调用 `predict_endpoint`，因此实际生成结果为：

$$
\hat{z}_1^{\text{int}} = \text{I2SB}_\theta(x, s; N, \sigma)
$$

---

## 4. 白化/雾化与平凡解的系统级可观测表现

### 4.1 白化的数学定义

设生成图像经 VAE decode 后为 $I_\text{gen}$，定义图像空间统计量：

| 指标 | 公式 | 健康参考（Seedream） | 620 白化信号 |
|------|------|---------------------|-------------|
| 对比度比 | $\kappa = \sigma_\text{luma} / \mu_\text{luma}$ | $\approx 0.42$ | $< 0.30$ |
| 动态范围 | $\rho = (p_{95} - p_5) / (p_{95} + p_5)$ | $\approx 0.62$ | $< 0.45$ |
| 饱和度均值 | $\varsigma = \text{mean}(\text{HSV}_S)$ | $\approx 0.36$ | $< 0.25$ |
| WFI | $1 - (0.4\kappa + 0.3\varsigma + 0.3\rho)$ | $\approx 0.16$ | $> 0.35$ |

本地最优基线 `620_film_v5_gated_local_smoke` WFI = 0.49，仍显著高于 Seedream 参考 0.16。

### 4.2 潜空间 shrinkage 指标

定义 source-target 位移 $\delta = y - x$，预测位移 $\Delta = \hat{z}_1 - x$，投影系数：

$$
\alpha = \frac{\langle \Delta, \delta \rangle}{\|\delta\|_2^2}
$$

| 指标 | 健康值 | 白化信号 | 本地证据 |
|------|--------|----------|----------|
| $\alpha(t=0)$ | $\geq 0.5$ | $< 0.3$ | targetlinear e8: 0.163 |
| $\alpha_\text{low}(t=0)$ | $\geq 0.3$ | 低但正 | targetlinear e8: 0.426 |
| $\alpha_\text{high}(t=0)$ | $\geq 0.3$ | $< 0.0$ | targetlinear e8: -0.050 |
| style sensitivity | $\geq 0.5$ | 接近 0 | endpoint_lowhigh: 0.003 |

### 4.3 平凡解（trivial solution）的可观测定义

称模型落入平凡解，当且仅当满足以下任一条件：

1. **端点位移不足**：$\alpha(t=0) < 0.3$；
2. **高频方向错误**：$\alpha_\text{high}(t=0) < 0$；
3. **风格不敏感**：对 $s_1 \neq s_2$，$\cos(v_\theta(x,0,s_1), v_\theta(x,0,s_2)) > 0.95$；
4. **图像动态范围压缩**：$\kappa_\text{gen} / \kappa_\text{src} < 0.9$ 且 $\rho_\text{gen} / \rho_\text{src} < 0.9$；
5. **训练—推理解耦**：训练 $\mathcal{L}$ 正常下降，但推理 $\hat{z}_1^{\text{int}}$ 出现上述指标异常。

---

## 5. 训练—推理耦合关系

### 5.1 训练目标并不直接约束推理轨迹

训练损失最小化的是单步 $v_\theta(x_t, t, s)$ 对 $v_\text{target}$ 的拟合误差。然而推理时：

- 起点固定为 $t=0$；
- 后续状态 $h_i$ 由模型自身的前一步输出决定；
- I2SB 积分引入随机噪声 $\varepsilon_i$。

因此，训练损失小 **不蕴含** 推理 endpoint 好。两者耦合的关键是 $v_\theta$ 在 $t=0$ 附近的正则性与连续性。

### 5.2 训练—推理耦合的数学条件

理想耦合要求：对任意 $t \in [0, 1]$ 和任意满足桥条件的状态 $h_t$，有：

$$
v_\theta(h_t, t, s) \approx \frac{y - h_t}{1 - t}
$$

该条件在 FM 最优解下成立。但若模型只学到了 $v_\theta(x_t, t, s) \approx v_\text{marginal}(x_t, t)$（条件期望坍缩），则对 $t=0$ 的 source 端：

$$
\hat{z}_1^{(0)} = x + v_\text{marginal}(x, 0)
$$

由于 $v_\text{marginal}$ 是多个 style 目标的平均，方向互相抵消，导致 $\|\hat{z}_1^{(0)} - x\| \ll \|y - x\|$，即白化。

### 5.3 关键验证命题

为证明/证伪训练—推理耦合是否破裂，应验证：

**命题 P1（轨迹一致性）**：对同一 $(x, y, s)$，沿训练插值路径 $x_t$ 和沿推理轨迹 $h_t$ 的 velocity 预测一致，即

$$
\| v_\theta(x_t, t, s) - v_\theta(h_t, t, s) \|_2 \ll \| v_\theta(x_t, t, s) \|_2
$$

**命题 P2（端点一致性）**：多步积分终点与单步 endpoint 接近：

$$
\| \hat{z}_1^{\text{int}} - \hat{z}_1^{(0)} \|_2 \ll \| \hat{z}_1^{(0)} - x \|_2
$$

本地实验证据：

- targetlinear formal e3 之前：$\hat{z}_1^{\text{int}}$ 与 $\hat{z}_1^{(0)}$ 几乎不变，solver 不补偿 endpoint 白化；
- targetlinear formal e6/e8：$\hat{z}_1^{(0)}$ 再次低对比度，但 solver 通过多步更新部分恢复动态范围，说明训练—推理耦合在晚期发生漂移。

---

## 6. 与代码实现的对应索引

| 数学对象 | 代码位置 |
|----------|----------|
| $x_t, v_\text{target}$ | `src/losses620.py::_vertical_state` |
| $y_\text{proj}$ | `src/losses620.py::_project_training_target` |
| $\mathcal{L}_\text{FM}, \mathcal{L}_\text{SWD}, \mathcal{L}_\text{edge}$ | `src/losses620.py::SpatialBridgeObjective620.compute` |
| $v_\theta$ 输出 | `src/model620.py::SpatialBridge620.forward` |
| endpoint 预测 | `src/model620.py::SpatialBridge620.predict_endpoint` |
| I2SB 积分 | `src/model620.py::SpatialBridge620.integrate_transport` |
| 推理封装 | `src/utils/inference.py::LGTInference.generation_with_target_latent` |
| cross-attention + FiLM | `src/blocks620.py::SpatialBridgeBlock620.forward` |

---

## 7. 假设清单

| 编号 | 假设 | 验证方式 |
|------|------|----------|
| A1 | 白化主要起源于 $t=0$ 端点预测 | 比较 `predict_endpoint(t=0)` 与 `integrate(nfe=N)` 的 WFI/alpha |
| A2 | 训练损失下降不能排除推理白化 | 检查训练 log 与 eval WFI 是否同步 |
| A3 | style 信号进入网络但未转化为端点位移 | 监控 `style_gate_value`, `film_gamma_abs`, `style_sensitivity_latent` |
| A4 | target_linear 路径改善低频迁移 | 对比 `"all"` 与 `"target_linear"` 的 `endpoint_low_to_source/target` |

---

## 8. 结论

620 的动力学可以概括为：

1. **训练阶段**：模型在 $(x_t, t, s)$ 上学习 $v_\theta$，损失由 FM + SWD + edge + lowfreq 组成；target projection mode 决定低频是否允许向 target 移动。
2. **推理阶段**：模型从 $x$ 出发，通过重复调用 `predict_endpoint` 并做 I2SB 更新得到最终结果；所有样本都必须经过 $t=0$ 端点预测。
3. **白化表现**：系统级可观测指标包括 WFI、潜空间 $\alpha$、高频方向、风格敏感度；当前证据表明问题集中在 $t=0$ 端点 shrinkage 与高频方向错误。
4. **训练—推理耦合**：训练损失正常不能保证推理质量；晚期训练可能出现 endpoint 动态范围再次坍缩，而 solver 部分补偿，形成一种新的 mismatch  regime。
