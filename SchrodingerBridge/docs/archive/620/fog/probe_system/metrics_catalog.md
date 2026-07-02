# 620 白化/雾化指标目录（Metrics Catalog）

> 本文件定义 620 白化问题诊断所需的全部定量指标，覆盖图像空间、潜空间、后处理补偿需求，以及与 Seedream IDT 的对照口径。

## 1. 图像空间指标（Image-Space Metrics）

所有图像空间指标都在解码后的 RGB 图像上计算。输入图像统一转换为 `np.uint8 [H, W, 3]`，像素值范围 `[0, 255]`。亮度（luminance）采用 Rec. 709 系数：

\[
L = 0.2126 R + 0.7152 G + 0.0722 B
\]

### 1.1 亮度均值（Brightness Mean）

\[
\text{brightness\_mean} = \frac{1}{H W} \sum_{i,j} \frac{L_{ij}}{255} \in [0, 1]
\]

物理意义：图像整体明暗。白化/雾化的典型特征是亮度向中间—高值区域集中（通常 `> 0.45`）。

### 1.2 对比度（Contrast Ratio）

\[
\text{contrast\_ratio} = \frac{P_{95}(L)}{\max(P_{5}(L), 1.0)}
\]

其中 `P5/P95` 为亮度分布的 5% 与 95% 分位数。范围 `[1, 255]`。值越小，说明图像越“灰”，白化越严重。

### 1.3 动态范围（Dynamic Range）

\[
\text{dynamic\_range} = \mathrm{std}(L) \in [0, 127.5]
\]

直接衡量像素值分布的展宽。白化图像动态范围显著收窄。

### 1.4 饱和度均值（Saturation Mean）

将 RGB 转换到 HSV 后取 S 通道均值：

\[
\text{saturation\_mean} = \frac{1}{H W} \sum_{i,j} S_{ij} \in [0, 1]
\]

\[
S = \begin{cases}
0, & \max(R,G,B) = 0 \\
\frac{\max - \min}{\max}, & \text{otherwise}
\end{cases}
\]

白化图像通常表现为饱和度系统性下降。

### 1.5 颜色方差塌缩（Color Variance Collapse）

对 RGB 三个通道分别计算标准差，再取平均：

\[
\text{color\_std\_mean} = \frac{1}{3} \bigl( \mathrm{std}(R) + \mathrm{std}(G) + \mathrm{std}(B) \bigr)
\]

颜色方差塌缩指生成图的 `color_std_mean` 明显低于 source 或 target，说明颜色多样性被压缩。

### 1.6 综合 WFI 分数（Whitening Fog Index）

由 `src/utils/wfi.py` 计算，将上述单项归一化后加权：

\[
\begin{aligned}
c_{\text{norm}} &= 1 - \min\bigl(\text{contrast\_ratio}/5,\ 1\bigr) \\
r_{\text{norm}} &= 1 - \min\bigl(\text{dynamic\_range}/60,\ 1\bigr) \\
s_{\text{norm}} &= 1 - \text{saturation\_mean} \\
b_{\text{norm}} &= \max\bigl(0,\ (\text{brightness\_mean} - 0.3)/0.4\bigr) \\
e_{\text{norm}} &= 1 - \min\bigl(\text{hist\_entropy}/7,\ 1\bigr)
\end{aligned}
\]

\[
\text{wfi\_score} = 0.25 c_{\text{norm}} + 0.20 r_{\text{norm}} + 0.20 s_{\text{norm}} + 0.15 b_{\text{norm}} + 0.20 e_{\text{norm}} \in [0, 1]
\]

`wfi_score` 越高代表越白/雾。`hist_entropy` 为亮度直方图 Shannon 熵（256 bins，log2）：

\[
H = -\sum_{k} p_k \log_2 p_k,\quad p_k = \frac{\#\{L \in \text{bin}_k\}}{H W}
\]

## 2. 潜空间指标（Latent-Space Metrics）

潜空间指标在 VAE 编码后的 latent `x \in \mathbb{R}^{B \times C \times H \times W}` 上计算。所有统计均在 `torch.no_grad()` 下完成，使用 `.float().detach()`。

### 2.1 Endpoint Alpha

对于 velocity 训练，模型输出 `v(x, t)`，预测 endpoint：

\[
\hat{x}_1 = x + (1 - t) \cdot v(x, t)
\]

Endpoint alpha 定义为预测 endpoint 相对 source `x_0` 向 target `x_1` 的位移比例：

\[
\alpha_{\text{endpoint}} = \frac{\|\hat{x}_1 - x_0\|_2}{\|x_1 - x_0\|_2 + \epsilon}
\]

其中 `||.||_2` 为 RMS（per-sample per-channel flatten 后的 L2 范数除以元素数开根号），`epsilon = 1e-6`。

- `alpha ≈ 0`：模型预测几乎回到 source，风格未迁移。
- `alpha ≈ 1`：预测 endpoint 接近真实 target。
- `alpha > 1` 或 `< 0`：过冲或反向，可能伴随白化/失真。

### 2.2 High-Frequency Alpha

先对 latent 做低通—高通分解：

\[
\begin{aligned}
x^{\text{low}} &= \text{AvgPool2d}(k=5)(x) \\
x^{\text{high}} &= x - x^{\text{low}}
\end{aligned}
\]

High-frequency alpha 只衡量高频分量的 endpoint 位移比例：

\[
\alpha_{\text{high}} = \frac{\|\hat{x}_1^{\text{high}} - x_0^{\text{high}}\|_2}{\|x_1^{\text{high}} - x_0^{\text{high}}\|_2 + \epsilon}
\]

白化往往表现为高频分量塌缩（`alpha_high` 偏低或 RMS 偏小），即图像细节/纹理丢失。

### 2.3 Channel Std Ratio

\[
\text{channel\_std\_ratio} = \frac{\frac{1}{C} \sum_c \mathrm{std}(x_c)}{\mathrm{std}(x)}
\]

其中 `std(x_c)` 为第 `c` 个通道在空间维度上的标准差，`std(x)` 为所有元素的全局标准差。

- 若各通道独立变化充分，ratio 接近 1/√C 附近（取决于通道相关性）。
- 白化塌缩时，常出现通道间统计高度一致，ratio 异常接近 1 或异常小。

### 2.4 Effective Rank

对 latent 张量 reshape 为 `(B, C, H*W)` 后按 batch 拼接成矩阵 `M \in \mathbb{R}^{(B \cdot C) \times (H W)}`，计算其奇异值 `sigma_1 >= sigma_2 >= ...`。Effective rank（基于奇异值熵）：

\[
p_i = \frac{\sigma_i}{\sum_j \sigma_j},\quad
\text{effective\_rank} = \exp\Bigl(-\sum_i p_i \log p_i\Bigr)
\]

生成图白化/塌缩时，有效秩通常显著低于 source/target。

### 2.5 Covariance Trace

将 latent reshape 为 `(B, C, H*W)` 后，按 batch 计算空间维度上的协方差矩阵（合并 batch）：

\[
\Sigma = \mathrm{Cov}(X) \in \mathbb{R}^{C \times C}
\]

\[
\text{cov\_trace} = \mathrm{tr}(\Sigma) = \sum_c \mathrm{Var}(X_c)
\]

trace 下降对应整体方差塌缩，是白化的直接潜空间信号。

## 3. 后处理补偿需求指标（Post-Processing Compensation Metrics）

这些指标量化“模型原生输出距离目标统计有多远”，用于判断是否需要后处理，以及后处理能否修复。

对生成图 `I_gen`、源图 `I_src`、目标风格参考图 `I_tgt` 分别计算第 1 节中的图像统计量，记为 `M(I)`。补偿需求定义为：

\[
\Delta_{\text{target}} M = M(I_{\text{gen}}) - M(I_{\text{target}})
\]

\[
\Delta_{\text{source}} M = M(I_{\text{gen}}) - M(I_{\text{source}})
\]

需要记录的 `M` 包括：

- 亮度均值差异：`brightness_mean_delta`
- 对比度差异：`contrast_ratio_delta`
- 动态范围差异：`dynamic_range_delta`
- 饱和度差异：`saturation_mean_delta`
- 颜色标准差差异：`color_std_delta`

以及潜空间版本（在 VAE latent 上）：

- latent 均值差异：`latent_mean_delta`
- latent 标准差差异：`latent_std_delta`
- latent 动态范围差异：`latent_dynamic_range_delta`
- latent 通道方差差异：`latent_channel_var_delta`

当 `|Delta_target M|` 远大于 `|Delta_source M|` 时，说明生成图偏离了目标风格统计，后处理补偿可能有效；若两者同时偏离，说明模型本身已经塌缩。

## 4. 与 Seedream IDT 的对照口径

Seedream IDT（参考值 `wfi_score ≈ 0.158`）是白化问题的一个关键放行锚点。为了复现该参考值，评估方式必须严格对齐：

1. **图像来源**：使用同一测试集（当前为 `wikiart_distinct5_samam_512_classview/test`）中的 source 图像经过 Seedream 的 identity-transfer（IDT）管线生成。
2. **解码方式**：Seedream IDT 的输出直接由 Seedream VAE 解码，不做额外 latent 后处理、不做人工亮度/对比度拉升。
3. **WFI 计算**：使用本仓库 `src/utils/wfi.py::compute_wfi_for_directory` 对生成目录批量计算，参数：
   - `pattern="*.png"`
   - `recursive=False`（如 source 目录含子风格目录则设为 `True`）
4. **汇总口径**：取 `generated_wfi.wfi_score.mean` 作为单点参考值。
5. **对照操作**：对 620 模型跑 `tools/run_eval_with_wfi.py`，在相同测试集、相同 source 目录、保存 PNG 后，读取 `wfi_eval_report.json` 中的 `wfi_score`。若 620 的 `wfi_score` 接近 0.158，则认为白化指标压到 Seedream IDT 水平。

> 注意：当前 `wfi_score` 的归一化阈值（contrast_ratio=5、dynamic_range=60、saturation_mean=1、brightness_mean 以 0.3 为起点、entropy 以 7 为满熵）是基于常见自然/艺术图像经验设定的。若 Seedream IDT 的实际统计分布显著不同，应同步更新 `wfi.py` 中的归一化参数或单独记录 raw 分量，避免单一综合分数掩盖分项差异。

## 5. 指标计算入口

| 指标类型 | 入口 | 输出位置 |
|---|---|---|
| 图像空间 WFI | `python -m utils.wfi <eval_dir> [--source-dir <dir>]` | `<eval_dir>/wfi_benchmark.json` |
| 图像空间 + 综合报告 | `python tools/run_eval_with_wfi.py --checkpoint ...` | `<output>/wfi_eval_report.json` |
| 潜空间 endpoint alpha | 训练/推理时读取 `model.last_debug` | 训练日志 CSV、TensorBoard/WandB |
| 层内探针 | 训练/推理时读取 `model.last_debug` 与各 `block.last_debug` | 训练日志 CSV、numeric_debug.jsonl |
| 后处理补偿差异 | `src/utils/run_evaluation.py` 生成的 `summary.json` | `summary.json.appearance_deltas` |
