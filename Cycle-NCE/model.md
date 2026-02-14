# 当前模型说明（聚焦风格注入与损失）

对应代码：
- `src/model.py`
- `src/losses.py`
- `src/config.json`

## 1. 核心范式

`LatentAdaCUT` 在 latent 空间做残差迁移：

- 输入：`x ∈ R^{B×4×32×32}`
- 输出：`y = x + delta`
- `delta` 由风格条件控制，风格条件可来自 `style_id`、`style_ref`，并可用 `style_mix_alpha` 混合。

训练时主路径是 **student(style_id-only)**；teacher 路径只在启用 `w_distill` 或 `w_code` 时参与。

## 2. 风格注入路径（当前实现）

### 2.1 向量注入：AdaGN

`ResBlock` 内部使用 AdaGN：
- `style_code -> Linear -> (scale, shift)`
- `h = GN(x) * scale + shift`

注入位置：
- 32x32 的 `hires_body` 块
- 16x16 的 `body` 块
- decoder（`use_decoder_adagn=true` 时）

### 2.2 空间注入：style_spatial map

空间风格图来自两类源：
- `style_ref` 编码得到的 spatial feat（32/16）
- `style_id` 对应的可学习先验图：`style_spatial_id_32/16`

混合后可选：
- `use_style_spatial_highpass`
- `normalize_style_spatial_maps`
- `use_style_spatial_blur`

实际注入点：
- 16x16 pre 注入：`h += style_spatial_pre_gain_16 * style_strength * map16`
- decoder 注入（可开关）：`h = h * (1 + gain * tanh(map32))`，其中 `gain=style_spatial_dec_gain_32 * style_strength`

### 2.3 Delta 聚合路径

`_compute_delta()` 按如下顺序组合：
1. 基础残差：`dec_out(h) * latent_scale_factor * residual_gain`
2. 可选输出仿射：`use_output_style_affine`
3. 可选门控：`use_style_delta_gate` + `style_gate_floor`
4. 纹理头：`style_texture_head`，增益 `style_texture_gain * style_strength`
5. 可选高频偏置：`use_delta_highpass_bias`

高频偏置参数：
- `style_delta_lowfreq_gain`
- `highpass_last_step_only`
- `highpass_last_step_scale`

在 `integrate()` 多步下，默认只在最后一步施加高频偏置（稳结构后补细节）。

### 2.4 style_strength 与 step schedule

- `style_strength` 会同时影响注入增益和步进缩放。
- `style_strength_step_curve` 支持：`linear` / `smoothstep` / `sqrt`。
- 多步积分 `integrate()` 支持 `step_schedule`：`flat` / `late` / `early` / `cosine` 或显式权重列表。

## 3. 损失函数（当前实现）

`AdaCUTObjective.compute()` 每个 batch 会随机采样训练动态：
- `train_num_steps` from `[train_num_steps_min, train_num_steps_max]`
- `train_step_size` from `[train_step_size_min, train_step_size_max]`
- `train_style_strength` from `[train_style_strength_min, train_style_strength_max]`

### 3.1 风格相关项

- `w_distill`：student 对齐 teacher（可 `distill_low_only`、`distill_cross_domain_only`）
- `w_code`：style code 闭环（teacher->ref, student->style_id）
- `w_stroke_gram`：笔触统计（Gram）
- `w_color_moment`：颜色矩匹配
- `w_push`：输出风格码与源域原型拉开 margin
- `w_style_spatial_tv`：约束 style_id 空间先验图的 TV

### 3.2 内容/结构项

- `w_struct`：输出与内容结构对齐（`struct_loss_type`, `struct_lowpass_strength`）
- `w_edge`：Sobel 边缘对齐
- `w_cycle`：跨域 cycle 对齐（`cycle_loss_type`, `cycle_lowpass_strength`, `cycle_edge_strength`）
- `w_nce`：token InfoNCE 内容一致性
- `w_semigroup`：一步与两步的半群一致性
- `w_delta_tv`：输出残差 TV，抑制块状/棋盘伪影

### 3.3 关键现状：无 warmup/ramp 调度

当前 `src/losses.py` 中 `set_progress()` 是空实现，loss 不再按 epoch 做 warmup/ramp。

这意味着旧配置字段（例如 `*_warmup_epochs`, `*_ramp_epochs`）对当前训练不生效。
建议：
- 清理这些 legacy 字段，避免误判“已启用调度”。
- 通过权重本身与 train-time 随机范围（steps/strength/step_size）来控制短程训练节奏。

## 4. 消融重点参数

### 4.1 注入侧（model）

- `use_decoder_spatial_inject`
- `style_spatial_pre_gain_16`
- `style_spatial_dec_gain_32`
- `style_texture_gain`
- `use_style_delta_gate`
- `use_output_style_affine`
- `use_delta_highpass_bias`
- `style_delta_lowfreq_gain`
- `highpass_last_step_scale`
- `use_style_spatial_highpass`
- `normalize_style_spatial_maps`
- `use_style_spatial_blur`

### 4.2 损失侧（loss）

- 风格：`w_distill`, `w_code`, `w_stroke_gram`, `w_color_moment`, `w_push`, `w_style_spatial_tv`
- 内容：`w_struct`, `w_edge`, `w_cycle`, `w_nce`, `w_semigroup`, `w_delta_tv`
- 训练动态：`train_num_steps_*`, `train_step_size_*`, `train_style_strength_*`, `train_step_schedule`

## 5. 50-epoch 训练建议（与当前实现匹配）

如果做短程（约 50 epoch）全面消融：
- 不要沿用 300 epoch 的 warmup/ramp 叙事；当前代码不会执行这些调度。
- 直接设置较密的评测节奏（例如每 10 epoch full eval + 最后一轮评测）。
- 通过“单项关断 + bundle 组合 + 动态范围扫描”定位敏感项。

仓库中的 `scripts/style_ablation.py` 已按上述口径实现。
