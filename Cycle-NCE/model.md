# 当前模型说明（与代码一致）

对应代码：
- `src/model.py`
- `src/losses.py`
- `src/config.json`

## 1. 模型总览

`LatentAdaCUT` 是一个在 latent 空间工作的轻量 U-Net（无 skip）。

- 输入：`x ∈ R^{B×4×32×32}`
- 输出：`y ∈ R^{B×4×32×32}`
- 本质：预测残差 `delta`，再做 `x + delta * step`

风格条件来源：
1. `style_id`（离散风格）
2. `style_ref`（参考风格 latent）

两者可由 `style_mix_alpha` 混合。

## 2. 主干结构（逐层）

### 2.1 风格编码分支

1. `style_enc`
- `Conv2d(4 -> lift_channels, 3x3)`
- `SiLU`
- `Conv2d(lift_channels -> base_dim*2, 4x4, stride=2)` 32->16
- `SiLU`
- `Conv2d(base_dim*2 -> base_dim*4, 4x4, stride=2)` 16->8
- `SiLU`
- `AdaptiveAvgPool2d(1)`

2. `style_proj`
- `Linear(base_dim*4 -> style_dim)`

3. `style_emb`
- `Embedding(num_styles, style_dim)`

4. 可学习空间先验
- `style_spatial_id_32`: `[num_styles, lift_channels, 32, 32]`
- `style_spatial_id_16`: `[num_styles, body_channels, 16, 16]`

### 2.2 生成主干

1. 输入抬升
- `enc_in: Conv2d(4 -> lift_channels, 3x3)`
- `enc_in_act: SiLU`

2. 32x32 高分辨率主体
- `hires_body`: `num_hires_blocks` 个 `ResBlock(lift_channels)`

3. 下采样
- `down: Conv2d(lift_channels -> body_channels, 4x4, stride=2)`

4. 16x16 主体
- `body`: `num_res_blocks` 个 `ResBlock(body_channels)`

5. 解码
- `dec_up: Upsample(scale=2, mode=upsample_mode)`
- `dec_conv: Conv2d(body_channels -> lift_channels, 3x3)`
- `dec_norm: AdaGN 或 GroupNorm`
- `dec_act: SiLU`
- `dec_out: Conv2d(lift_channels -> 4, 3x3)`

6. Texture 头（唯一风格细节头，Content-Aware）
- 输入默认为 `concat(h, style_spatial_dec)`（通道 `2*lift_channels`）
- `Conv2d(2*lift_channels -> lift_channels, 3x3)`
- `SiLU`
- `Conv2d(lift_channels -> 4, 1x1)`

7. NCE 投影头
- `Linear(4 -> projector_dim) -> ReLU -> Linear(projector_dim -> projector_dim)`

## 3. 风格注入机制（当前清理版）

当前保留三条核心路径：

### 3.1 AdaGN 向量调制

在 `ResBlock`（和可选 decoder norm）中通过 `style_code` 预测 `(scale, shift)`，调制归一化特征：
- `out = GN(x) * scale + shift`

### 3.2 空间注入（精简后）

当前仅保留两个注入点：
1. **16×16 pre 注入**：`h += style_spatial_pre_gain_16 * map16`
2. **decoder 单点注入**：`dec_conv` 后注入一次

职责划分：
- `map16`：中频结构引导（边缘、块面、形体组织）
- `map32`：解码端细节调制（纹理/笔触门控）

decoder 注入固定为 film 形式：
- `h <- h * (1 + gain * tanh(map))`

已移除：
- 32×32 注入
- block 后重复注入
- decoder 第二次 post-act 注入
- force 分支

### 3.3 delta 聚合

`_compute_delta` 现在是：
1. 基础残差：`dec_out(h) * latent_scale_factor * residual_gain`
2. 可选输出仿射：`use_output_style_affine`
3. 可选门控：`use_style_delta_gate`
4. texture 残差（默认 Content-Aware）：`style_texture_head(concat(h, style_spatial_dec))`，按 `style_texture_gain * style_strength` 叠加
5. 可选高通偏置（见第 4 节）

## 4. “最后一步轻微高通”机制

这是当前版本新增的重点策略：

相关参数（`model` 配置）：
- `use_delta_highpass_bias`
- `style_delta_lowfreq_gain`
- `highpass_last_step_only`
- `highpass_last_step_scale`

行为：
- 在 `integrate()` 多步时，前几步不加高通（稳结构）
- 仅最后一步按 `highpass_last_step_scale` 混合高通后的 `delta`（补细节）

高通公式（实现上是低高频重组）：
- `low = avgpool+upsample(delta)`
- `high = delta - low`
- `hp = high + low * style_delta_lowfreq_gain`
- `delta_final = (1-a)*delta + a*hp`，其中 `a = highpass_last_step_scale`

这与“先稳中频、再补中高频”目标一致。

## 5. 前向与多步

### 5.1 单步 `forward`
- 计算 `delta`
- 输出：`x + delta * step_size * step_scale`

### 5.2 多步 `integrate`
每步都重算 `delta`，并按 `step_schedule` 权重累积：
- 支持 `num_steps / step_size / style_strength / step_schedule`
- 权重归一化后求和
- 高通策略按第 4 节执行

## 6. 当前损失函数（`src/losses.py`）

当前版本已去掉稀疏调度，所有 loss 都是每步按权重直接计算。

- `distill`：student 对齐 teacher
- `code`：teacher 对齐 ref 编码，student 对齐 id 编码
- `cycle`：跨域回环一致
- `struct`：结构一致
- `edge`：Sobel 边缘一致
- `stroke_gram`：笔触统计（Gram）
- `color_moment`：颜色矩匹配
- `nce`：token 级 InfoNCE
- `semigroup`：一步/两步等价约束
- `push`：远离源域风格
- `delta_tv`：残差平滑
- `style_spatial_tv`：风格先验图平滑

## 7. 当前风格关键配置（建议重点关注）

`src/config.json -> model`：
- `style_spatial_pre_gain_16`
- `style_spatial_dec_gain_32`
- `style_texture_gain`
- `style_texture_mode`（`content_aware` 或 `style_only`）
- `use_style_delta_gate`
- `use_output_style_affine`
- `use_delta_highpass_bias`
- `highpass_last_step_only`
- `highpass_last_step_scale`
- `style_delta_lowfreq_gain`

`src/config.json -> inference`：
- `num_steps`
- `step_size`
- `style_strength`
- `step_schedule`

这就是当前“瘦身后 + 末步轻高通”的完整模型形态。

## 8. 配置健壮性（防幽灵参数）

`build_model_from_config()` 会对未知 `model` 字段发出 `UserWarning`。  
目的：避免“配置里写了参数但模型实际未读取”的训练/推理漂移问题。
