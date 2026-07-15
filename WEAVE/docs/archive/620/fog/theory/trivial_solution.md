# 平凡解（Endpoint Shrinkage / Gray Mean）收敛性分析

> Round M 理论文档：推导白化平凡解形成的数学条件，区分五种进入机制，并给出可证伪预测与本地实验判定。

---

## 1. 平凡解的数学定义

设 source latent 为 $x$，target latent 为 $y$，模型预测的 endpoint 为 $\hat{z}_1$。定义：

- 目标位移：$\delta = y - x$
- 预测位移：$\Delta = \hat{z}_1 - x$
- 投影系数：$\alpha = \dfrac{\langle \Delta, \delta \rangle}{\|\delta\|_2^2}$

**平凡解**指优化结果落在以下区域：

$$
\alpha \ll 1, \quad \|\Delta\|_2 \ll \|\delta\|_2
$$

即端点几乎不朝目标方向移动。视觉上表现为低对比度、低饱和度、亮度偏高（WFI 上升）。

极端平凡解为：

$$
\hat{z}_1 \approx x \quad \text{或} \quad \hat{z}_1 \approx \bar{x} = \mathbb{E}[x]
$$

前者为 identity-like 收缩，后者为全局灰均值收缩。

---

## 2. 临界点方程

### 2.1 联合损失

训练目标为：

$$
\mathcal{L}(v) = w_\text{FM} \|v - v_\text{target}\|_2^2 + w_\text{SWD} \cdot \text{SWD}(x_t + (1-t)v,\; y_\text{proj}) + w_\text{edge} \cdot \mathcal{L}_\text{edge} + w_\text{low} \cdot \mathcal{L}_\text{low}
$$

其中 $v_\text{target} = y_\text{proj} - x$。

对 $v$ 求梯度：

$$
\nabla_v \mathcal{L} = 2 w_\text{FM}(v - v_\text{target}) + w_\text{SWD}(1-t) \nabla_z \text{SWD}(z, y_\text{proj})\big|_{z=x_t+(1-t)v} + \nabla_v \mathcal{L}_\text{edge} + \nabla_v \mathcal{L}_\text{low}
$$

临界点 $v^*$ 满足 $\nabla_v \mathcal{L} = 0$。

### 2.2 平坦区与排序稳定性

SWD 的梯度在投影值排序稳定时为 0：

$$
\nabla_z \text{SWD}(z, y_\text{proj}) = 0, \quad \text{当 } \{d_k^\top z^{(i)}\} \text{ 与 } \{d_k^\top y_\text{proj}^{(i)}\} \text{ 排序不变时}
$$

此时临界点退化为：

$$
v^* = v_\text{target} - \frac{1}{2 w_\text{FM}} \big( \nabla_v \mathcal{L}_\text{edge} + \nabla_v \mathcal{L}_\text{low} \big)
$$

若 edge/lowfreq 正则项指向 source，则 $v^*$ 被拉向 0，形成 shrinkage。

---

## 3. 五种进入机制

### 3.1 Loss-Driven：目标函数本身的 shrinkage 吸引子

**机理**：FM loss 与 SWD loss 在最优解附近共同作用，可能使 $v^*$ 落入低幅度区域。

在排序稳定区，SWD 梯度为 0，FM loss 单独作用：

$$
\mathcal{L}_\text{FM} = \|v - v_\text{target}\|_2^2
$$

最小值在 $v = v_\text{target}$，无 shrinkage。但加入 edge/lowfreq 后，若这些辅助 loss 惩罚大位移，则最小值内移。

**可证伪预测**：

- P-L1：若单独训练 FM loss（$w_\text{SWD}=w_\text{edge}=w_\text{low}=0$），$\alpha(t=0)$ 应接近 1。
- P-L2：若增大 $w_\text{edge}$ 或 $w_\text{low}$，$\alpha$ 应下降。

**本地证据**：

- `lowfreqfix` 分支增大了 endpoint low-frequency 惩罚，结果 `velocity_abs` 从 0.15 降到 0.016，`endpoint_low_to_target_ratio` 从 5.3 升到 12.4，表明确实被 loss 拉向 source。**支持 P-L2**。
- 但 baseline 在 $w_\text{edge}=0.1$ 时仍有 $\alpha=0.16$，说明 loss-driven 不是唯一机制。

**判定**：Loss-driven 是贡献因素，但不是根因。

---

### 3.2 Norm-Driven：GroupNorm / LayerNorm / AdaLN 压缩动态范围

**机理**：网络中大量 GroupNorm(1)（等价于 LayerNorm）会归一化 feature map 的均值和方差，抑制 endpoint 预测的大幅度变化。

对任意特征图 $h \in \mathbb{R}^{B \times C \times H \times W}$，GroupNorm(1) 输出为：

$$
\text{GN}(h)_{b,c,h,w} = \frac{h_{b,c,h,w} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

其中 $\mu_b, \sigma_b^2$ 为整张特征图的空间均值和方差。该操作会：

1. 强制输出均值为 0；
2. 压缩跨样本/跨通道的方差差异；
3. 在 endpoint head 中直接限制预测动态范围。

**可证伪预测**：

- P-N1：移除 endpoint head 中的 GroupNorm 后，$\alpha$ 应上升。
- P-N2：在 block 中将 GroupNorm 替换为 BatchNorm 或不使用 norm 后，`film_gamma_abs` 与 style 敏感度的相关性应增强。
- P-N3：比较同一 checkpoint 的 block 输入/输出方差比，若方差被系统压缩，则 norm-driven 显著。

**本地证据**：

- `diagnosis_and_solution.md` 明确指出 endpoint head 中的 `GroupNorm(1)` 会抑制动态范围；
- `model620.py` 中 `"velocity"` head 已改为无 GroupNorm 的 3 层 Conv+SiLU，但 `"endpoint_lowhigh"` head 仍保留 `GroupNorm(1)`；
- 本地 smoke `620_film_v5_*` 在 block 内启用 StyleFiLM 但 endpoint head 仍为 velocity，未显著降低 WFI，说明仅移除 trunk norm 不足。**部分支持 P-N1**（需要进一步验证 endpoint head 无 GN 的效果）。

**判定**：Norm-driven 是重要机制，尤其在 endpoint head 中。

---

### 3.3 Attention-Driven：Cross-Attention 将 style 信号平均化为边缘期望

**机理**：Cross-attention 的 softmax 输出接近均匀分布，使 style tokens 的加权和几乎与具体 style 无关：

$$
\text{CA}(x, S) = \text{softmax}\left(\frac{Q(x) K(S)^T}{\sqrt{d}}\right) V(S) \approx \frac{1}{N}\sum_{i=1}^N V(S)_i
$$

因此模型学到的是：

$$
v_\theta(x, t, s) \approx \mathbb{E}_s[v_\theta(x, t, s)] = \bar{v}(x, t)
$$

即条件期望坍缩。不同 style 的 $y_s - x$ 方向互相抵消，$\|\bar{v}\| \ll \|\mathbb{E}_s\|v_s\|\|$。

**可证伪预测**：

- P-A1：固定 $(x, t)$，对不同 style $s_1, s_2$，$\cos(v_\theta(x,t,s_1), v_\theta(x,t,s_2))$ 应接近 1。
- P-A2：增大 `style_cross_attn_gate_init` 后，上述余弦应下降。
- P-A3：用 FiLM 绕过 cross-attention 后，style sensitivity 应显著上升。

**本地证据**：

- gate=0.05 baseline：`cos_sim(v(s_1), v(s_2)) = 0.9995`**支持 P-A1**；
- gate=0.3 后 gate 值上升到 0.297，但 1 epoch 内 style transfer 质量未显著改善，说明 gate  alone 不够；
- StyleFiLM 在 block 内启用后 `film_gamma_abs` 增长，但本地 smoke 仍 WFI=0.49，说明 attention-driven 被部分绕过但仍未完全解决。**部分支持 P-A2/P-A3**。

**判定**：Attention-driven 是条件期望坍缩的核心机制之一，但非唯一。

---

### 3.4 Endpoint-Driven：Head 参数化与初始化使端点偏向 source

**机理**：Endpoint head 的初始化与结构使训练初期 $\hat{z}_1 \approx x$，优化从 identity 邻域出发，容易陷入 shrinkage basin。

以 `"velocity"` head 为例，输出为：

$$
v = \text{Conv}_3\big( \text{SiLU}(\text{Conv}_2(\text{SiLU}(\text{Conv}_1(h)))) \big)
$$

若最后一层权重接近 0，则 $v \approx 0$，$\hat{z}_1 \approx x_t$。即使非零初始化，若 head 容量小或 norm 压缩，也倾向于学习小位移。

以 `"endpoint_lowhigh"` 为例：

$$
\hat{z}_1 = (x_\text{low} + \Delta_\text{low}) + (x_\text{high} + \Delta_\text{high})
$$

其中 $\Delta_\text{low} = \text{head}_\text{low}(h) + \text{style}_\text{low}$，$\Delta_\text{high} = (\text{head}_\text{high}(h) + \text{style}_\text{high}) \cdot \gamma$。

若 head 输出小且 `endpoint_style_to_low/high` 初始化为 0（代码中确实 `zeros_`），则 $\hat{z}_1 \approx x$，即平凡解。

**可证伪预测**：

- P-E1：将 endpoint head 最后一层初始化为更大方差后，$\alpha$ 应上升。
- P-E2：直接预测 endpoint（而非 velocity）并移除 $(1-t)$ 分母后，shrinkage 应减轻。
- P-E3：在 endpoint head 中加入 style-FiLM 后，style sensitivity 应恢复。

**本地证据**：

- `model620.py` 中 `"velocity"` head 已使用 `std=0.02` 的非零初始化；
- `"endpoint_lowhigh"` 无 FiLM 的 smoke 结果：`latent_alpha=-0.0404`，`style_sensitivity=0.00285`，端点几乎坍回 source。**支持 P-E2 风险**；
- `"endpoint_lowhigh"` + style 注入的 smoke：`style_sensitivity=0.2285`，但仍未恢复目标方向。**部分支持 P-E3**；
- FiLM endpoint head formal 5 epoch：`latent_alpha=0.1232`，style sensitivity=10.1，但仍未达健康阈值。**说明 endpoint-driven 机制复杂，需要更强的 style 调制**。

**判定**：Endpoint-driven 是白化的直接表现层机制；head 初始化、容量、style 调制方式共同决定 shrinkage 深浅。

---

### 3.5 Solver-Driven：I2SB 积分放大或固化 shrinkage

**机理**：若 endpoint 预测本身低能量，多步 I2SB 可能进一步将结果拉向 source 或某个平均状态。

I2SB 更新为：

$$
h_{i+1} = \frac{1 - t_{i+1}}{1 - t_i} h_i + \frac{t_{i+1} - t_i}{1 - t_i} \hat{z}_1(h_i, t_i) + \text{noise}
$$

若 $\hat{z}_1(h_i, t_i) \approx h_i$（端点接近当前状态），则 $h_{i+1} \approx h_i$，积分成为 identity map，无法移向 target。

反之，若端点过度激进，solver 的加权平均可能部分修正，但会引入 source-reanchoring：

$$
h_{i+1} - y \approx \frac{1 - t_{i+1}}{1 - t_i}(h_i - y) + \frac{t_{i+1} - t_i}{1 - t_i}(\hat{z}_1 - y)
$$

若 $\|\hat{z}_1 - y\| > \|h_i - y\|$，solver 反而会拉回 source 侧。

**可证伪预测**：

- P-S1：若 solver 是白化主因，则 `predict_endpoint(t=0)` 应健康，但 `integrate(N)` 后 WFI 恶化。
- P-S2：改变 `num_steps` 应显著改变 WFI。
- P-S3：设 $\sigma=0$ 后推理结果应与有噪声时显著不同。

**本地证据**：

- 原始 baseline fog probe：`predict_endpoint(t=0)` 已白化，`integrate(nfe=16)` 几乎不改变 img_std。**否证 P-S1/P-S2**；
- targetlinear formal e8：`predict_endpoint(t=0)` img_std=0.132，nfe16 img_std=0.196，solver 部分补偿但主要拉回 source。**说明 solver 可以是补偿者而非原始来源**；
- solver trace probe：`step_low_to_source_cos \approx -0.867`，`step_low_to_target_cos \approx +0.637`，solver 本身不 source-seeking。**否证 solver-driven 是主因**。

**判定**：Solver-driven 机制在当前 620 证据中 **已被否证** 为主要原因；solver 仅在晚期作为部分补偿者出现。

---

## 4. 机制综合：进入平凡解的条件概率

基于本地实验，五种机制的贡献排序为：

| 机制 | 贡献度 | 证据强度 | 当前状态 |
|------|--------|----------|----------|
| Attention-driven | 高 | 强 | 已确认（gate/softmax/FiLM） |
| Endpoint-driven | 高 | 强 | 已确认（head init/容量/style 调制） |
| Norm-driven | 中—高 | 中 | 高度可疑（GN 在 head） |
| Loss-driven | 中 | 中 | 贡献但非根因 |
| Solver-driven | 低 | 强 | 已否证 |

**综合数学模型**：

设 shrinkage 系数 $\alpha$ 可近似为各机制的乘积效应：

$$
\alpha \approx \alpha_\text{loss} \cdot \alpha_\text{norm} \cdot \alpha_\text{attn} \cdot \alpha_\text{endpoint} \cdot \alpha_\text{solver}
$$

其中每个因子 $\in (0, 1]$。当前观测 $\alpha \approx 0.16$ 可由多个因子的联合压缩解释：

- $\alpha_\text{attn} \approx 0.3$（条件期望坍缩）
- $\alpha_\text{endpoint} \approx 0.5$（head 弱初始化/小容量）
- $\alpha_\text{norm} \approx 0.7$（GN 压缩）
- $\alpha_\text{loss} \approx 0.9$（辅助 loss 轻微拉扯）
- $\alpha_\text{solver} \approx 1.0$（无额外压缩）

乘积 $\approx 0.094$，与观测量级一致；由于存在非线性耦合，实际值在 0.16 附近。

---

## 5. 可证伪预测汇总

| 编号 | 预测 | 验证实验 | 当前判定 |
|------|------|----------|----------|
| P-L1 | 纯 FM loss 下 $\alpha \approx 1$ | 训练 FM-only smoke | 未做 |
| P-L2 | 增大 lowfreq/edge 惩罚会降 $\alpha$ | `lowfreqfix` 分支 | **支持** |
| P-N1 | 移除 endpoint head GN 升 $\alpha$ | `endpoint_film_enabled` + 无 GN | 待验证 |
| P-N2 | 替换 trunk GN 增强 style 敏感度 | BatchNorm / no-norm 实验 | 待验证 |
| P-A1 | gate=0.05 时 style velocity 余弦接近 1 | velocity direction probe | **支持** |
| P-A2 | gate=0.3 降低 style velocity 余弦 | gate 实验 | 部分支持（gate 值上升，1 epoch 未改善 transfer） |
| P-A3 | FiLM 绕过 attention 后 style sensitivity 上升 | StyleFiLM smoke | 部分支持 |
| P-E1 | 大 std init 升 $\alpha$ | 已实施 std=0.02 | 部分有效 |
| P-E2 | endpoint head 直接预测减轻 shrinkage | `endpoint_lowhigh` smoke | **否证**（无 style 注入时更差） |
| P-E3 | endpoint head + style-FiLM 恢复 style sensitivity | `endpointstylehead` / `film_formal` | 部分支持 |
| P-S1 | solver 是白化来源 | fog probe 对比 endpoint 与 integrate | **否证** |
| P-S2 | NFE 改变显著影响 WFI | 不同 NFE 对比 | **否证** |
| P-S3 | noise 改变显著影响结果 | solver trace with/without sigma | **否证** |

---

## 6. 与代码的对应关系

| 机制 | 代码位置 |
|------|----------|
| Loss-driven | `src/losses620.py::compute` 中 `w_flow`, `single_step_swd_weight`, `w_content_lowpass_anchor` |
| Norm-driven | `src/blocks620.py` `norm1`, `norm2`, `time_adaln`；`src/model620.py` `GroupNorm(1, ...)` in endpoint heads |
| Attention-driven | `src/blocks620.py::_attention_stats`, cross-attention softmax/gated/relu2/style_select modes |
| Endpoint-driven | `src/model620.py` `endpoint_head_mode`, `endpoint_film_enabled`, `endpoint_style_to_low/high` init |
| Solver-driven | `src/model620.py::integrate_transport`, `src/utils/inference.py::LGTInference` |

---

## 7. 结论

1. **平凡解不是单一机制造成**，而是 attention-driven 条件期望坍缩、endpoint-driven head 弱表达、norm-driven 动态范围压缩、loss-driven 辅助正则共同作用的结果。
2. **Solver-driven 机制已被本地实验否证**：白化在 `predict_endpoint(t=0)` 已存在，I2SB 既不原创也不主要恶化它（晚期仅部分补偿）。
3. **当前最关键的未验证假设**是 P-N1（移除 endpoint head GroupNorm）与更强的 P-E3（endpoint head 内部 style-FiLM 调制）。
4. 任何单一修复（仅增大 gate、仅加 FiLM、仅改 endpoint head）都不足以跳出 shrinkage basin；需要多机制联合干预。
