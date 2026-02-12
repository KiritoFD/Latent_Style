# 模型说明（当前实现，中文详细版）

## 1. 目标与当前版本定位

当前 `Cycle-NCE` 的模型实现，目标是同时满足：

- 8GB 显存可训练（结合分时损失与 checkpoint）
- 推理多步可控（`num_steps / step_size / step_schedule / style_strength`）
- 风格强度统一控制（避免“旋钮分散”）
- 降低网格纹理风险（特别是解码器近输出注入）

核心代码文件：

- `src/model.py`
- `src/losses.py`

默认配置入口：

- `src/config.json`

## 2. 当前默认配置（关键项）

以 `src/config.json` 为准，当前默认关键模型设置：

- 通道与规模：
  - `latent_channels=4`
  - `base_dim=128`
  - `lift_channels=128`
  - `style_dim=256`
  - `num_hires_blocks=4`
  - `num_res_blocks=2`
- 风格注入相关：
  - `style_spatial_pre_gain_32=0.38`
  - `style_spatial_block_gain_32=0.18`
  - `style_spatial_pre_gain_16=0.32`
  - `style_spatial_block_gain_16=0.16`
  - `use_decoder_spatial_inject=true`
  - `style_spatial_dec_gain_32=0.08`
  - `style_spatial_dec_gain_out=0.03`
- 风格残差路径：
  - `use_style_texture_head=true`
  - `use_style_force_path=true`
  - `share_style_head_trunk=true`
  - `style_texture_gain=0.28`
  - `style_force_gain=0.35`
- 高频与抗混叠：
  - `use_delta_highpass_bias=true`
  - `style_delta_lowfreq_gain=0.1`
  - `use_style_spatial_blur=true`
  - `use_downsample_blur=true`
  - `upsample_mode="bilinear"`
  - `decoder_spatial_mode="film"`
- 统一强度控制：
  - `style_strength_default=1.0`
  - `style_strength_step_curve="smoothstep"`
  - `step_schedule_default="flat"`
  - `style_gate_floor=0.0`
  - `style_gate_floor_low_strength=0.85`

## 3. 网络结构与信息流

### 3.1 主干结构（无 skip 的微型 U-Net）

输入输出均为 latent：

- 输入：`[B, 4, 32, 32]`
- 输出：`[B, 4, 32, 32]`

主流程：

1. 编码抬升：`enc_in`（4 -> 128，32x32）
2. 高分辨率残差体：`hires_body`（4 个 ResBlock，128 通道）
3. 下采样：`down`（128 -> 256，32 -> 16）
4. 主体残差体：`body`（2 个 ResBlock，256 通道）
5. 解码：上采样 -> `dec_conv` -> `dec_norm` -> `dec_out`
6. 残差更新：`x + delta * step_size_scale`

### 3.2 风格编码与风格图来源

风格条件来自两条路径：

- `style_id` 路径：
  - `style_emb`
  - `style_spatial_id_32 / style_spatial_id_16`
- `style_ref` 路径：
  - `style_enc + style_proj`
  - 从中间特征提取空间图（32 与 16）

推理部署可只走 `style_id` 路径，不依赖参考图。

### 3.3 风格空间注入位置

当前实现注入点：

- 32 尺度预注入：`style_spatial_pre_gain_32`
- 32 尺度 block 后注入：`style_spatial_block_gain_32`
- 16 尺度预注入：`style_spatial_pre_gain_16`
- 16 尺度 block 后注入：`style_spatial_block_gain_16`
- 解码器注入两次（pre-norm 与 post-act）：
  - `style_spatial_dec_gain_32`
  - `style_spatial_dec_gain_out`

且所有注入增益统一乘以 `style_strength`。

## 4. 风格残差聚合（delta 形成）

`_compute_delta` 当前包含：

1. 基础残差：`dec_out(h) * latent_scale_factor * residual_gain`
2. `style_delta_gate` 门控（含 floor）
3. texture 分支
4. force 分支
5. 高频偏置（可对分支和最终 delta 生效）

### 4.1 共享风格头（Round 2）

开启 `share_style_head_trunk=true` 后：

- 共享 `style_head_trunk`：`conv3x3 + SiLU`
- 分别 `style_texture_out`、`style_force_out` 做 1x1 输出

与旧版双独立 3x3 相比，减少一个热点 3x3 卷积。

### 4.2 解码注入模式

`decoder_spatial_mode` 支持：

- `add`：直接加法注入
- `film`：乘性调制（`h * (1 + gain * tanh(map))`）

默认 `film`，用于降低近输出端直接叠加产生的周期纹理风险。

## 5. 统一强度与多步可控机制

### 5.1 `style_strength` 统一入口

`style_strength` 同时影响：

- 空间图注入增益
- `step_size` 的有效缩放（经 `style_strength_step_curve`）
- gate floor 插值（`style_gate_floor` 到 `style_gate_floor_low_strength`）
- texture/force 分支强度
- 高频偏置的低频保留比例

### 5.2 `forward` 与 `integrate`

- `forward`：单步
- `integrate`：多步 Euler 迭代

二者都支持：

- `style_strength`
- `step_schedule`

`step_schedule` 支持：

- 命名策略：`flat / late / early / cosine`
- 显式权重列表：`[w1, w2, ...]`

内部会归一化，保证不同 schedule 的总量可比较。

## 6. 与损失侧的训练-推理对齐

`src/losses.py` 当前做了三层对齐：

1. 训练随机采样：
  - `train_num_steps_min/max`
  - `train_step_size_min/max`
  - `train_style_strength_min/max`
  - `train_step_schedule`
2. Semigroup 约束：
  - `F(h1+h2)` 对齐 `F_h2(F_h1(x))`
3. 重损失分时：
  - interval 控制
  - 可选 round-robin

这样可以把“推理 5 步改 3 步”的漂移风险显著压低。

## 7. 当前实测规模（基于当前代码与默认配置）

以下统计为当前实现的单样本、单步、`style_id` 路径实测：

- 可训练参数总量：`8,856,333`
- Conv/Linear 总 MAC（B=1）：约 `2,410,742,016`

主要参数模块（按聚合模块）：

- `body`: `2,886,656`
- `style_enc`: `2,626,944`
- `hires_body`: `1,707,008`
- `down`: `524,544`
- `dec_conv`: `295,040`
- `style_spatial_id_32`: `262,144`
- `style_head_trunk`: `147,584`

主要 MAC 热点（Top）：

- `dec_conv`: `301,989,888`
- `hires_body.*.conv*`: 每层约 `150,994,944`
- `body.*.conv*`: 每层约 `150,994,944`
- `style_head_trunk.0`: `150,994,944`
- `down`: `134,217,728`

说明：

- 历史 `model_layer_macs_b1.csv` / `model_layer_params.csv` 仍可用于趋势参考。
- 但 Round 2 开启共享头后，热点名应关注 `style_head_trunk.0`，而非旧的 `style_texture_head.0` 与 `style_force_head.0` 双 3x3。

## 8. 质量与风险控制建议（对应当前代码）

为降低网格/周期纹理风险，建议保持以下组合：

- `upsample_mode="bilinear"`
- `use_downsample_blur=true`
- `use_style_spatial_blur=true`
- `decoder_spatial_mode="film"`
- `w_style_spatial_tv > 0`

若需要更强风格：

- 优先加 `style_strength`
- 再调 `step_schedule`（如 `late`）
- 最后再提高 texture/force gain

避免一次性同时拉高所有注入与高频项。
