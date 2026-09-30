# 统计塌缩（Statistical Collapse）分析

> Round M 理论文档：分析 GroupNorm / LayerNorm / AdaLN 对振幅、通道方差、颜色统计的影响，以及 cross-attention 后 norm 如何洗掉 style 信号；给出层内 probe 应观测的统计量与数学预测。

---

## 1. 问题背景

620 模型中大量使用 GroupNorm(1)（等价于 LayerNorm）和 AdaLN(time)。这些归一化操作会改变特征图的一阶和二阶统计量，从而：

1. 压缩动态范围（与 WFI 中的低对比度、低动态范围直接相关）；
2. 洗掉 style 注入后的通道级调制信号；
3. 改变 latent 的均值/方差，经 VAE decode 后放大为图像空间白化。

本文件量化这些效应，并给出层内 probe 的数学预测。

---

## 2. GroupNorm / LayerNorm 的数学效应

### 2.1 GroupNorm(1) = LayerNorm

对特征图 $h \in \mathbb{R}^{B \times C \times H \times W}$，`nn.GroupNorm(1, C)` 在每个样本的全部空间维度上归一化：

$$
\text{GN}(h)_{b,c,i,j} = \gamma_c \cdot \frac{h_{b,c,i,j} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}} + \beta_c
$$

其中：

$$
\mu_b = \frac{1}{C H W} \sum_{c,i,j} h_{b,c,i,j}, \quad \sigma_b^2 = \frac{1}{C H W} \sum_{c,i,j} (h_{b,c,i,j} - \mu_b)^2
$$

注意：当 `affine=False` 时，$\gamma_c = 1, \beta_c = 0$。

### 2.2 对振幅的影响

归一化后，每个样本的 L2 范数被强制接近 $\sqrt{C H W}$：

$$
\| \text{GN}(h)_b \|_2 \approx \sqrt{C H W}
$$

若输入 $h_b$ 本身具有大动态范围（如 target latent 与 source latent 差异大），GN 会将其压缩到统一尺度。

**对 endpoint head 的影响**：

在 `endpoint_lowhigh` mode 中：

$$
\Delta_\text{low} = \text{Conv}\big( \text{SiLU}( \text{GN}(h) ) \big)
$$

由于 $\text{GN}(h)$ 的方差被归一化为 1，$\Delta$ 的幅度主要由 Conv 权重决定。若权重初始化小（`std=1e-3`），则 $\Delta$ 必然小，导致 endpoint 接近 source。

### 2.3 对通道方差的影响

GN(1) 在 **所有通道和空间位置** 上计算一个均值和方差，因此：

- 不同通道间的相对方差差异被消除；
- 颜色通道（latent 的 4 个通道）的独立统计被强制对齐。

在 VAE latent 空间中，4 个通道分别编码不同的视觉属性（如亮度、色度、纹理等）。GN(1) 抹平这些差异，相当于把颜色信息“漂白”。

---

## 3. AdaLN(time) 的数学效应

### 3.1 AdaLN 定义

在 `SpatialBridgeBlock620` 中：

$$
\text{scale}, \text{shift}, \text{gate} = \text{time\_adaln}(\text{time\_emb}).\text{chunk}(3, \text{dim}=1)
$$

$$
h_\text{time} = \text{GN}(h) \odot (1 + \text{scale}) + \text{shift}
$$

然后 self-attention 输出再乘以 $\sigma(\text{gate})$。

### 3.2 对时间条件的依赖

AdaLN 将时间信息以仿射变换形式注入。若 `time_adaln` 最后一层初始化为 0（代码中 `zeros_`），则初始时：

$$
\text{scale} = 0, \quad \text{shift} = 0, \quad \sigma(\text{gate}) = 0.5
$$

即初始行为接近 identity（带 0.5 缩放）。

但训练后，若 gate 学习为接近 0 或 1 的饱和值，self-attention 的贡献会被压制或放大；若 scale/shift 学习为接近 -1，特征会被归零。

### 3.3 AdaLN 与 style 的交互

style 信号通过 cross-attention 和 FiLM 进入 trunk 后，会立即经过 AdaLN 或 FFN 中的 GN。若 style 调制产生的通道差异被后续 GN 洗掉，则 style 信号无法有效传递。

---

## 4. Cross-Attention 后 Norm 洗掉 Style 信号

### 4.1 Style 注入路径

在 `SpatialBridgeBlock620` 中，style 信号通过三步进入 trunk：

1. **Cross-attention**：$\Delta_\text{CA} = \tanh(g) \cdot \text{softmax}(QK^T)V$
2. **Shortcut**：$x' = \alpha x + \Delta_\text{CA}$
3. **Post-CA FiLM**：$x'' = (1 + \gamma(s)) \odot x' + \beta(s)$
4. **FFN with GN**：$x''' = x'' + \text{FFN}(\text{GN}(x''))$

### 4.2 GN 洗掉 FiLM 调制的数学分析

FiLM 输出的期望和方差为：

$$
\mathbb{E}[x''] = \mathbb{E}[(1+\gamma)x' + \beta] = (1+\gamma)\mathbb{E}[x'] + \beta
$$

$$
\text{Var}[x''] = (1+\gamma)^2 \text{Var}[x']
$$

但随后的 GN 会重新归一化：

$$
\text{GN}(x'') = \frac{x'' - \mu_{x''}}{\sqrt{\sigma_{x''}^2 + \epsilon}}
$$

归一化后，由 $\gamma, \beta$ 引入的均值和方差变化被消除。**只有高阶统计量（如分布形状、高阶矩）保留**，而一阶、二阶 style 信息被洗掉。

### 4.3 量化：Style 信号保留率

定义 style 信号保留率为：

$$
R_\text{style} = \frac{\| \text{GN}(x''(s_1)) - \text{GN}(x''(s_2)) \|_2}{\| x''(s_1) - x''(s_2) \|_2}
$$

由于 GN 减去了均值并除以标准差，对于仅改变均值/方差的 style 差异，$R_\text{style} \approx 0$。

**预测**：在 FFN 前对两个不同 style 的输入测量 $R_\text{style}$，应观察到显著下降。

---

## 5. 层内 Probe 应观测的统计量

### 5.1 推荐观测列表

| 统计量 | 符号 | 测量位置 | 健康预测 | 白化信号 |
|--------|------|----------|----------|----------|
| 特征图均值 | $\mu_{b,c}$ | block 输入/输出、head 输入 | 保留 source/target 差异 | 所有样本趋同 |
| 特征图标准差 | $\sigma_{b,c}$ | 同上 | 通道间有差异 | 通道间被拉平 |
| 跨 style 差异范数 | $\|h(s_1) - h(s_2)\|_2$ | CA 后、FiLM 后、FFN 后 | 逐层保持或放大 | 被 norm 洗掉 |
| Style 信号保留率 | $R_\text{style}$ | FFN 前后 | $> 0.5$ | $< 0.2$ |
| 振幅比 | $\|h_\text{out}\|_2 / \|h_\text{in}\|_2$ | 每个 block | 接近 1 或合理增长 | 系统 $< 1$ |
| Endpoint head 输出范数 | $\|\Delta_\text{head}\|_2$ | endpoint head 输出 | 与 $\|y - x\|_2$ 可比 | 显著偏小 |

### 5.2 数学预测

#### 预测 1：GN 后振幅压缩

对任意输入 $h$，经过 `GN(1)` 后：

$$
\sigma_\text{out} = 1 \quad \text{（当 affine=False）}
$$

若期望 style 调制产生方差变化，GN 会将其归零。

#### 预测 2：AdaLN gate 饱和会压制 style

若 `time_adaln` 的 gate 输出 $\sigma(g) \to 0$，则 self-attention 分支被关闭：

$$
\text{sa\_delta} = \sigma(g) \cdot \text{SA}(h) \to 0
$$

style 只能通过 cross-attention 和 FiLM 传递，若这些也弱，则 style 完全失效。

#### 预测 3：FiLM 后 GN 洗掉通道调制

对 FiLM 输出 $x'' = (1+\gamma) x' + \beta$，GN 后：

$$
\text{GN}(x'')_{b,c,i,j} = \frac{(1+\gamma_c) x'_{b,c,i,j} + \beta_c - \mu_{x''}}{\sqrt{\sigma_{x''}^2 + \epsilon}}
$$

$\gamma_c, \beta_c$ 仅通过改变分布的高阶形状影响输出；若 $x'$ 近似高斯，则 style 信息大量丢失。

---

## 6. Attention 归一化对统计塌缩的影响

### 6.1 Softmax（默认）

$$
\text{attn}_i = \frac{\exp(z_i / \sqrt{d})}{\sum_j \exp(z_j / \sqrt{d})}
$$

**效应**：

- 强制 $\sum_i \text{attn}_i = 1$；
- 输出 $\text{CA} = \sum_i \text{attn}_i V_i$ 是 $V$ 的凸组合，幅度受 $V$ 控制；
- 当 attention 接近均匀时，$\text{CA} \approx \bar{V}$，style 信息被平均化。

本地证据：`cross_attn_entropy = 5.531 / \ln(256) = 5.545$（99.9% 均匀）。

### 6.2 Gated（Sigmoid + 重归一化）

$$
\text{attn}_i = \sigma(z_i / \sqrt{d}), \quad \text{CA} = \frac{\sum_i \text{attn}_i V_i}{\sum_i \text{attn}_i}
$$

**效应**：

- 输出仍是 $V$ 的凸组合，但权重不再被 softmax 的尖锐性限制；
- 本地实验 `620_film_v5_gated_local_smoke` WFI=0.49，为四个 attention 变体中最低；
- 但 content LPIPS 上升到 0.33，说明 gated attention 在保留内容的同时仍无法充分迁移 style。

### 6.3 Gated_Raw（Sigmoid，无归一化）

$$
\text{attn}_i = \sigma(z_i / \sqrt{d}), \quad \text{CA} = \sum_i \text{attn}_i V_i
$$

**效应**：

- 输出不是凸组合，幅度可随 gate 总和变化；
- 可能保留更多幅度信息，但输出尺度不稳定；
- 本地实验 WFI=0.64，白化最严重，说明无归一化导致 output 均值漂移，反而被后续 norm 拉回 source-like 状态。

### 6.4 ReLU²

$$
\text{attn}_i = \text{ReLU}(z_i)^2, \quad \text{CA} = \sum_i \text{attn}_i V_i
$$

**效应**：

- 稀疏且非负，无归一化；
- 可能放大大幅值方向；
- 本地实验 WFI=0.53，仍白化，且 content LPIPS=0.31。

### 6.5 Style_Select（Top-k + Softmax）

$$
\text{attn}_i = \text{softmax}(\text{top\_k}(z_i))
$$

**效应**：

- 只选择 top-16 tokens，减少平均化；
- 但本地实验 WFI=0.50，效果不显著，可能因为 top-k 选择仍由 content-dependent Q 决定，style 区分度不足。

### 6.6 综合比较

| 模式 | WFI（本地 smoke） | content LPIPS | 解释 |
|------|-------------------|---------------|------|
| gated | 0.49 | 0.330 | 相对最好，但仍白化 |
| gated_raw | 0.64 | 0.297 | 无归一化导致统计漂移，norm 后更白 |
| relu2 | 0.53 | 0.310 | 稀疏但风格区分不足 |
| style_select | 0.50 | 0.333 | top-k 未能解决 content-style 冲突 |

**结论**：改变 attention 归一化形式 alone 不能解决白化；核心问题在 style 信号是否足够强地进入 endpoint head，而非 attention 内部。

---

## 7. 与图像空间白化的联系

### 7.1 Latent 统计 → Decode 图像

VAE decode 可近似为非线性映射 $D: \mathbb{R}^{4 \times 64 \times 64} \to \mathbb{R}^{3 \times 512 \times 512}$。若 latent 的：

- 均值被拉高（接近 1），decode 后图像整体偏亮；
- 方差被压缩，decode 后对比度降低；
- 通道间统计被拉平，decode 后饱和度降低。

这些正是 WFI 中亮度高、对比度低、饱和度低的来源。

### 7.2 本地证据

`targetlinear formal e8` endpoint：

- `latent_std = 0.9249`（source=0.7916，target 未知但应更高）
- `img_std = 0.2194`（source=0.2022）
- `endpoint_img_std_vs_source_ratio = 1.0852$（e3 健康）→ e8 降至 0.6541

说明 latent 方差尚可，但解码后图像动态范围压缩，可能是因为 latent 的统计分布形状（高阶矩）不健康，或 endpoint 方向错误导致 decode 后亮度分布集中。

---

## 8. 验证实验设计

### 8.1 层内 Hook Probe

在每个 block 的以下位置注册 hook：

1. `input_proj` 后（进入 trunk 前）
2. self-attention 后
3. cross-attention 后
4. post-CA FiLM 后
5. FFN 后

测量每个位置的 $\mu, \sigma, R_\text{style}$。

### 8.2 Endpoint Head 统计

测量 endpoint head 输入/输出的：

- 均值、方差
- 与 source/target 的 L2 距离
- 经过 GN 前后的范数比

### 8.3 Attention 模式对比

在相同初始化下训练不同 `style_attn_mode`：

- 测量每个 mode 的 `cross_attn_delta_abs`、`film_gamma_abs`、`style_sensitivity_latent`；
- 关联 WFI 和 content LPIPS。

---

## 9. 结论

1. **GroupNorm(1) = LayerNorm 是统计塌缩的主要来源之一**：它强制每个样本的均值和方差归一化，抹平通道差异和 style 调制产生的一阶/二阶变化。
2. **AdaLN(time) 进一步叠加时间条件化**：若 gate 饱和或 scale/shift 不当，会压制 style 相关分支。
3. **Cross-attention 后的 GN/FFN 会洗掉 FiLM 的 style 调制**：即使 FiLM 能产生 style-specific 输出，后续 GN 也会将其均值/方差信息归零。
4. **Attention 归一化形式对统计塌缩有次要影响**：gated 相对最好，但所有变体均未能解决白化，说明 attention 不是唯一瓶颈。
5. **层内 probe 的关键指标**：$R_\text{style}$、振幅比、通道方差差异、endpoint head 输出范数。这些指标可定量验证 norm-driven 机制的贡献。
