# 训练—推理 Mismatch 分析

> Round M 理论文档：分析 endpoint head 输出、velocity 预测、target projection 与 solver trace 的关系，推导训练 loss 正常但生成图雾化的条件，以及 late-stage mismatch 的判据。

---

## 1. 核心问题

620 项目中反复观察到一个现象：

> 某些 checkpoint 的训练 loss 正常甚至持续下降，但生成图像仍然雾化/白化。

这意味着训练目标与推理目标之间存在 **mismatch**。本文件从数学上分析 mismatch 的来源、判据和风险模式。

---

## 2. 训练 loss 的正常下降意味着什么？

### 2.1 训练目标回顾

训练时，对采样得到的 $(x, y, s, t)$，最小化：

$$
\mathcal{L} = w_\text{FM} \|v_\theta(x_t, t, s) - v_\text{target}\|_2^2 + w_\text{SWD} \cdot \text{SWD}(\hat{z}_1, y_\text{proj}) + \dots
$$

其中：

$$
x_t = x + t(y_\text{proj} - x), \quad v_\text{target} = y_\text{proj} - x, \quad \hat{z}_1 = x_t + (1-t)v_\theta(x_t, t, s)
$$

### 2.2 Loss 下降 ≠ 端点正确

训练 loss 下降只保证：

$$
v_\theta(x_t, t, s) \approx v_\text{target} \quad \text{在训练分布 } (x_t, t) \text{ 上}
$$

但推理时：

- 输入固定为 $x$（即 $t=0$ 的 source），而非训练分布中的 $x_t$；
- 模型需要在所有 $t \in [0, 1]$ 上连续、一致地预测 velocity；
- 多步积分引入累积误差和 self-consistency 要求。

因此，训练 loss 正常仅说明模型在插值路径上拟合良好，不保证在 source 端 $t=0$ 处的行为正确。

---

## 3. Endpoint Head 输出与 Velocity 预测的关系

### 3.1 两种 Head Mode 的数学等价性与差异

#### Mode A：`endpoint_head_mode = "velocity"`

网络直接输出 $v_\theta(h, t, s)$，然后：

$$
\hat{z}_1 = h + (1-t) v_\theta(h, t, s)
$$

此时 FM loss 直接监督 $v$，梯度路径短，但 $v$ 的幅度受网络输出尺度限制。

#### Mode B：`endpoint_head_mode = "endpoint_lowhigh"`

网络预测 low/high endpoint delta：

$$
\Delta_\text{low} = \text{head}_\text{low}(h) + M_\text{low}(s_\text{global})
$$

$$
\Delta_\text{high} = \big( \text{head}_\text{high}(h) + M_\text{high}(s_\text{global}) \big) \cdot \gamma
$$

$$
\hat{z}_1 = (h_\text{low} + \Delta_\text{low}) + (h_\text{high} + \Delta_\text{high})
$$

然后反解 velocity：

$$
v_\theta = \frac{\hat{z}_1 - h}{1 - t}
$$

#### 关键差异

| 方面 | velocity | endpoint_lowhigh |
|------|----------|------------------|
| 网络输出 | $v$ | $\Delta_\text{low}, \Delta_\text{high}$ |
| FM loss 监督对象 | 直接 $v$ | 通过 $\hat{z}_1$ 间接 $v$ |
| $t=0$ 行为 | $v$ 有限即 endpoint 有限 | 分母 $1-t$ 会放大 $\Delta$ 误差 |
| style 注入点 | 仅 trunk | trunk + endpoint head（若启用 FiLM） |
| 初始化风险 | 最后一层小 init → $v \approx 0$ | `endpoint_style_to_*` zero-init → $\hat{z}_1 \approx h$ |

### 3.2 训练时 $t$ 采样的影响

训练 $t$ 通过 `torch.empty(...).uniform_(0, 1).pow(power)` 采样。默认 `t_sampling_power=1.0` 即均匀采样，此时 $t=0$ 附近样本密度有限。

若模型在 $t > 0$ 时学得较好，但在 $t=0$ 处存在边界问题，训练 loss 仍可正常下降，因为：

$$
\mathbb{E}_t[\mathcal{L}(t)] = \int_0^1 \mathcal{L}(t) \, dt
$$

$t=0$ 的单个点对整体积分贡献小。

但推理时所有样本都经过 $t=0$，因此边界错误被放大。

---

## 4. Target Projection 与 Solver Trace 的关系

### 4.1 Target Projection 决定训练路径

`training_target_projection_low_mode` 决定低频部分是否允许移动：

- `"all"`：$x_\text{low}$ 锚定在 source，$v_\text{target,low} = 0$；
- `"target_linear"`：$x_\text{low} = (1-t)c_\text{low} + t \cdot y_\text{low}$，$v_\text{target,low} = y_\text{low} - c_\text{low}$。

### 4.2 Solver Trace 的数学形式

推理时 I2SB 轨迹 $\{h_i\}_{i=0}^N$ 满足：

$$
h_{i+1} = a_i h_i + b_i \hat{z}_1(h_i, t_i) + \xi_i
$$

其中 $a_i = \frac{1 - t_{i+1}}{1 - t_i}$，$b_i = \frac{t_{i+1} - t_i}{1 - t_i}$，$\xi_i \sim \mathcal{N}(0, \sigma_i^2)$。

定义误差 $e_i = h_i - y$：

$$
e_{i+1} = a_i e_i + b_i (\hat{z}_1(h_i, t_i) - y) + \xi_i
$$

若 $\hat{z}_1(h_i, t_i) = y$（理想 endpoint），则 $e_{i+1} = a_i e_i + \xi_i$，误差因 $a_i < 1$ 而指数衰减。

若 $\hat{z}_1(h_i, t_i) \approx h_i$（平凡 endpoint），则 $e_{i+1} \approx (a_i + b_i)e_i + \xi_i = e_i + \xi_i$，误差不收敛。

### 4.3 训练路径与推理轨迹的不一致

训练时 $x_t$ 是已知的、确定性的插值点；推理时 $h_i$ 由模型自己生成。若模型对训练分布外的 $h_i$ 泛化差，则：

$$
\hat{z}_1(h_i, t_i) \neq \hat{z}_1(x_{t_i}, t_i)
$$

即使训练 loss 很低，推理轨迹也会偏离训练路径。

---

## 5. 为什么训练 Loss 正常但生成图雾化？

### 5.1 充分条件推导

设训练损失低但生成图雾化，需要同时满足：

1. **平均拟合好**：$\mathbb{E}_t[\|v_\theta(x_t, t, s) - v_\text{target}\|^2] \ll 1$
2. **$t=0$ 边界差**：$\|v_\theta(x, 0, s) - v_\text{target}\| \gg 0$
3. **$t=0$ 对平均贡献小**：$t$ 采样使 $t=0$ 权重低
4. **solver 不补偿**：$\hat{z}_1^{\text{int}}$ 仍接近 $\hat{z}_1(x, 0, s)$

数学上，若 $t$ 采样密度为 $p(t)$，则：

$$
\mathcal{L} = \int_0^1 p(t) \|v_\theta(x_t, t, s) - v_\text{target}\|^2 dt
$$

即使 $p(0) > 0$，只要 $v_\theta(x_0, 0, s)$ 的误差被其他 $t$ 的低 loss 平均掉，总 loss 仍可低。

### 5.2 本地证据

`targetlinear_swd8_sigma002_nfe8_b64` formal run：

| epoch | loss | loss_fm | endpoint_low_to_source | endpoint_low_to_target | endpoint_low_target_ratio | velocity_abs |
|-------|------|---------|------------------------|------------------------|---------------------------|--------------|
| 1 | 2.6199 | 1.4483 | 0.3310 | 0.3355 | 1.0248 | 0.1524 |
| 3 | 2.5590 | 1.4280 | 0.3347 | 0.3262 | 0.9839 | 0.1747 |
| 6 | 2.3222 | 1.2427 | 0.4304 | 0.2790 | 0.6514 | 0.3603 |
| 8 | 2.2738 | 1.2099 | 0.4470 | 0.2694 | 0.6050 | 0.3958 |

观察：

- 训练 loss 持续下降；
- `endpoint_low_to_source` 上升，`endpoint_low_to_target` 下降；
- 按训练指标，模型在“更好地朝 target 移动”；
- 但 fog probe epoch 8 显示 `endpoint_img_std_vs_source_ratio = 0.6541`，endpoint 图像动态范围反而压缩。

这就是典型的 **training—inference mismatch**：训练指标与推理视觉质量脱节。

### 5.3 Mismatch 的解释

训练指标 `endpoint_low_to_source/target` 是在 latent 空间、低通滤波后、相对 source/target 的 L1 距离。它不能捕捉：

- 高频结构的正确性（`high_alpha` 在 epoch 8 仍为负）；
- 解码后图像的动态范围（VAE decode 对 latent 统计非线性）；
- 多步积分后的累积行为（solver 拉回 source）。

因此，模型可能在 latent 低通空间“更接近 target”，但在图像空间仍然雾化。

---

## 6. Late-Stage Mismatch 的理论判据

### 6.1 定义

称 checkpoint 进入 late-stage mismatch，当同时满足：

1. 训练 loss 仍在下降或稳定；
2. 训练指标 `endpoint_low_target_ratio` 改善（变小）；
3. 推理指标 WFI 恶化或不再改善；
4. `endpoint_img_std_vs_source_ratio` 下降或低于 0.9。

### 6.2 数学判据

定义训练—推理一致性指标：

$$
\mathcal{M} = \frac{\| \hat{z}_1^{\text{int}} - \hat{z}_1^{(0)} \|_2}{\| \hat{z}_1^{(0)} - x \|_2}
$$

- 若 $\mathcal{M} \ll 1$：solver 不改变 endpoint，mismatch 由 $t=0$ endpoint 决定；
- 若 $\mathcal{M} \approx 1$ 或 $> 1$：solver 显著改变结果，进入 late-stage mismatch regime。

进一步定义图像空间一致性：

$$
\mathcal{M}_\text{img} = \frac{\| I_\text{gen}^{\text{nfe}=N} - I_\text{gen}^{\text{nfe}=1} \|_2}{\| I_\text{gen}^{\text{nfe}=1} - I_\text{src} \|_2}
$$

若 $\mathcal{M}_\text{img} > 0.5$ 且 WFI(nfe=N) 显著不同于 WFI(endpoint)，则存在 late-stage mismatch。

### 6.3 本地 Late-Stage 证据

`targetlinear formal e8`：

| stage | img_std | to_source_img_delta_rms | to_target_img_delta_rms |
|-------|---------|-------------------------|-------------------------|
| source | 0.2022 | 0.0000 | 0.2950 |
| endpoint | 0.1323 | 0.2497 | 0.2359 |
| nfe16 | 0.1956 | 0.1559 | 0.2617 |

计算：

$$
\mathcal{M}_\text{img} = \frac{0.1956 - 0.1323}{0.2022 - 0.1323} \approx 0.91
$$

即 solver 对图像动态范围的修正幅度接近 endpoint 本身的偏差，late-stage mismatch 成立。

同时：

- endpoint `to_target_img_delta_rms = 0.2359`
- nfe16 `to_target_img_delta_rms = 0.2617`

说明 solver 的补偿部分通过拉回 source 实现，而非更接近 target。

---

## 7. Velocity vs Endpoint_LowHigh 两种 Head Mode 的 Mismatch 风险

### 7.1 Velocity Mode 的风险

| 风险 | 说明 |
|------|------|
| 幅度压缩 | 网络直接输出 $v$，容易被 weight decay / small init 压到接近 0 |
| 高频与低频耦合 | 单通道输出无法独立控制 low/high band |
| style 调制弱 | style 只通过 trunk 影响 $v$，endpoint head 本身无 style 条件 |

本地证据：`620_film_v5_gated_local_smoke` 使用 velocity head + block 内 FiLM，WFI=0.49，仍未解决白化。

### 7.2 Endpoint_LowHigh Mode 的风险

| 风险 | 说明 |
|------|------|
| 分母放大 | $v = (\hat{z}_1 - h)/(1-t)$，在 $t=0$ 处小误差被放大 |
| head 初始化 | `endpoint_style_to_low/high` zero-init 使初始 $\hat{z}_1 \approx h$ |
| GroupNorm 压缩 | head 中使用 `GroupNorm(1)` 限制动态范围 |
| style 注入不足 | 仅 additive style offset 不够，需要 feature-level FiLM |

本地证据：

- 无 FiLM 的 endpoint_lowhigh：`latent_alpha=-0.0404`，端点坍回 source；
- 有 style 注入的 endpoint_lowhigh：`style_sensitivity=0.2285`，但仍未恢复目标方向；
- FiLM endpoint head formal：`latent_alpha=0.1232`，仍未达健康阈值。

### 7.3 两种 Mode 的共同风险

无论哪种 mode，若 style 条件未充分进入 head，则：

$$
v_\theta(x, 0, s) \approx v_\text{marginal}(x, 0)
$$

导致平凡解。区别仅在于 shrinkage 的表现形式：

- velocity mode：$v$ 幅度小；
- endpoint_lowhigh mode：$\Delta$ 幅度小或方向错误。

---

## 8. 验证方法

### 8.1 必要探针

| 探针 | 目的 | 输出指标 |
|------|------|----------|
| `probe_620_hypothesis_metrics.py` | 测量 $\alpha, \alpha_\text{low}, \alpha_\text{high}$ | `latent_alpha_mean`, `high_alpha_mean`, `style_sensitivity_latent` |
| `probe_620_fog_path.py` | 比较 endpoint 与 integrate 的图像统计 | `img_std`, `img_grad_rms`, `to_source/target_img_delta_rms` |
| `probe_620_endpoint_time_sweep.py` | 检查 $t=0$ 边界是否异常 | `img_std(t=0)` vs `img_std(t=0.5)` |
| `probe_620_solver_trace.py` | 分析 solver 每步方向 | `step_low_to_source_cos`, `step_low_to_target_cos` |

### 8.2 判定流程

```
1. 测量 alpha(t=0)
   ├── alpha < 0.3 → endpoint 本身 shrinkage
   │   └── 进一步区分 attention / endpoint / norm 机制
   └── alpha >= 0.5
       ├── 比较 endpoint 与 nfe16 的 WFI
       │   ├── 差异小 → solver 不相关，问题在 decode 或 alpha 足够但方向错误
       │   └── 差异大 → late-stage mismatch
       └── 测量 solver trace cos → 判断 solver 是否 source-seeking
```

---

## 9. 结论

1. **训练 loss 正常不保证推理质量好**。训练目标只约束插值路径上的 $v_\theta(x_t, t, s)$，而推理要求 $t=0$ 端点正确且全路径连续。
2. **$t=0$ 边界问题是 mismatch 的核心**。即使平均训练 loss 低，$t=0$ 的小区域错误会被推理无限放大。
3. **Late-stage mismatch 有明确判据**：训练指标改善但推理 WFI 恶化，且 $\mathcal{M}_\text{img} > 0.5$。
4. **Velocity 与 endpoint_lowhigh 两种 mode 各有风险**：velocity mode 易受幅度压缩，endpoint_lowhigh mode 易受初始化、GN、分母放大影响。两者都需要 style 条件强进入 head。
5. **当前最需验证的修复**：确保 endpoint head（无论哪种 mode）内部有 style-FiLM 调制且无 GroupNorm，同时增大 gate 初始值以打破条件期望坍缩。
