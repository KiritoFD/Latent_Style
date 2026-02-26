# 当前模型设计总结（基于 `src/` 实现）

更新时间：2026-02-25  
覆盖范围：`src/model.py`、`src/losses.py`、`src/config.json`、`src/run.py`

## 1. 模型目标与 I/O

当前主模型是 `LatentAdaCUT`，工作在 VAE latent 空间做风格迁移残差预测。

- 输入：`x ∈ R[B, 4, 32, 32]`
- 条件：`style_id`（离散风格 ID）
- 输出：`y = x + delta`

当前实现是纯 `style_id` 条件路径，不依赖参考风格图像（`style_ref`）。

---

## 2. 网络结构

### 2.1 总体拓扑

`LatentAdaCUT` 可以看成「32x32 提升 + 16x16 主干 + 32x32 解码」的轻量 U-Net：

1. `enc_in`: `4 -> lift_channels`（默认 128）  
2. `hires_body`: 若干 `ResBlock` 在 `32x32` 上运行  
3. `down`: stride=2，降到 `16x16` 且通道到 `body_channels = 2*base_dim`（默认 256）  
4. `body`: 若干 `ResBlock` 在 `16x16` 上运行  
5. `dec_up`: 上采样回 `32x32`  
6. `skip_fusion`: 融合 decoder 特征与 32x32 skip  
7. `dec_conv + dec_norm + dec_act + dec_out` 输出 `delta`

### 2.2 风格注入机制

核心注入单元是 `CoordSPADE`：

- 先做 `GroupNorm(affine=False)`
- 拼接三类输入：`normalized feature`、`(x,y) 坐标网格`、`style_code` 广播图
- 通过卷积预测 `gamma/beta`，形成 `normalized * (1 + gamma) + beta`
- 由 `gate` 控制注入强度（残差式混合）

`ResBlock` 中两次归一化都用 `CoordSPADE`，因此在高分辨和主干 block 内都可注入风格。

### 2.3 风格表示

模型内部有两种 style 表示：

- 全局风格向量：`style_emb(style_id)`（`num_styles x style_dim`）
- 空间风格先验：`style_spatial_id_16`（`num_styles x body_channels x 16 x 16`）

训练时可对空间先验施加随机平移抖动（`style_id_spatial_jitter_px`），并做标准化。

### 2.4 注入位置与强度

注入强度由 `style_strength`（0~1）与三类 gate 共同决定：

- `inject_gate_hires`：32x32 `hires_body`
- `inject_gate_body`：16x16 `body`
- `inject_gate_decoder`：decoder 归一化层

此外，16x16 处有显式空间先验预注入：

- `h += style_spatial_pre_gain_16 * style_strength * tanh(style_map_16)`

### 2.5 输出与积分

- 单步 `forward`:  
  `x + delta * step_size * step_scale(style_strength)`
- 多步 `integrate`:  
  重复 `num_steps` 次，每次加 `delta * step_size * step_scale / num_steps`

`step_scale` 由 `style_strength_step_curve` 控制：

- `linear`
- `sqrt`
- `smoothstep`

---

## 3. 损失设计（`AdaCUTObjective`）

`compute()` 每个 batch 会随机采样训练动态参数：

- `train_num_steps`（整数区间）
- `train_step_size`（连续区间）
- `train_style_strength`（连续区间）

然后组合以下损失：

1. `loss_swd`：跨域样本（`source_style_id != target_style_id`）上的 SWD  
2. `loss_identity`：同域样本上的恒等约束  
3. `loss_delta_tv`：`pred-content` 的 TV  
4. `loss_delta_l1`：`pred-content` 的 L1  
5. `loss_output_tv`：`pred` 的 TV  
6. `loss_semigroup`：半群一致性（按 `semigroup_every_n_steps` 间隔触发）

总损失：

`total = w_swd*swd + w_identity*identity + w_delta_tv*delta_tv + w_delta_l1*delta_l1 + w_output_tv*output_tv + w_semigroup*semigroup`

---

## 4. 关键配置映射（当前 `src/config.json`）

### 4.1 模型侧

- `latent_channels=4`
- `base_dim=128`
- `lift_channels=128`
- `style_dim=256`
- `num_styles=5`（运行时会与数据目录数自动对齐）
- `num_hires_blocks=4`
- `num_res_blocks=6`
- `num_groups=4`
- `latent_scale_factor=0.18215`
- `residual_gain=1.0`
- `style_spatial_pre_gain_16=1.0`
- `use_decoder_adagn=true`
- `style_strength_default=1.0`
- `style_strength_step_curve=smoothstep`
- `upsample_mode=bilinear`
- `upsample_blur=true` / `upsample_blur_kernel=gaussian3`

### 4.2 损失侧（当前有效）

- `w_swd=20.0`
- `w_identity=4.0`
- `w_delta_tv=0.01`
- `w_delta_l1=0.0`
- `w_output_tv=0.0`
- `w_semigroup`（配置里默认未给出，按 0 处理）
- `swd_patch_sizes=[1,3,5]`
- `swd_num_projections=512`
- `train_num_steps_min/max=1/1`

---

## 5. 运行时边界与注意点

1. `run.py` 会严格校验 `config.loss` 键。老字段如 `w_distill`、`w_code` 会直接报错。  
2. `run.py` 的白名单里有部分 semigroup 扩展键，但 `src/losses.py` 当前并未使用这些扩展项。  
3. 模型输出是 latent 残差，不直接约束像素范围；视觉行为依赖下游 VAE decode。  
4. `integrate` 会复用同一 `style_code/style_map`，属于固定条件下的迭代积分。  
5. 当前模型设计目标是稳定、可部署的 `style_id -> latent edit` 路径，而非 teacher/student 蒸馏框架。

