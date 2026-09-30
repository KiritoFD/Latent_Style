# Latent AdaCUT 模型与训练设计详解（中文）

本文档对应当前 `Cycle-NCE/src` 代码实现，目标是把“模型如何注入风格、各损失项如何拉扯、每个配置参数实际影响什么”讲清楚，便于你做稳定调参与复现实验。

- 代码主入口：`src/run.py`
- 模型定义：`src/model.py`
- 损失定义：`src/losses.py`
- 训练循环：`src/trainer.py`
- 主实验配置：`src/config.json`
- overfit50 配置：`src/overfit50.json`

## 1. 总体设计

当前系统是 `latent -> latent` 的风格迁移：

1. 输入 `content latent`（通常 `[B,4,32,32]`）。
2. 模型输出残差 `delta`，最终输出 `pred = content + delta`。
3. 风格条件同时来自：
   - `style_id`（推理必须可用）
   - `style_ref`（训练时 teacher 路径可用）
4. 训练采用 teacher-student：
   - teacher：有 reference 风格图
   - student：只用 style_id（部署路径）

这个结构的重点是：
- 不让模型依赖推理时不存在的 reference；
- 用蒸馏把 teacher 的风格能力迁移到 student。

## 2. 模型结构细节（`src/model.py`）

### 2.1 主干流程

1. `enc_in`: `4 -> lift_channels`，先把低维 latent 展开。
2. 32x32 高分辨率段（`num_hires_blocks`）做风格注入与内容变换。
3. 下采样到 16x16 后进入 body 段（`num_res_blocks`）。
4. 解码回 32x32，输出 `delta`（再按多种门控/频率偏置处理）。

### 2.2 风格注入通道

风格注入不是单一路径，而是叠加：

- 全局风格码：`style_enc + style_proj`（reference）与 `style_emb`（style_id）混合。
- 空间风格图：32x32/16x16 的 spatial map，注入到 encoder/body/decoder。
- 纹理支路：`style_texture_head`（可单独控制增益）。
- 风格强制支路：`style_force_path`（用于强化风格变化）。
- 输出仿射：`use_output_style_affine`（输出阶段再做风格调制）。

### 2.3 频率相关机制

- `use_delta_highpass_bias`：对输出残差做高频偏置，抑制纯低频色偏捷径。
- `style_delta_lowfreq_gain`：控制低频残差保留比例（越小越偏向高频变化）。
- `use_style_spatial_highpass`：对空间风格图做高通。
- `use_style_spatial_blur` / `use_downsample_blur`：抗混叠，减轻格子/锯齿伪影。
- `upsample_mode`：当前主配置用 `bilinear`，比某些上采样方式更稳。

## 3. 损失函数细节（`src/losses.py`）

### 3.1 teacher-student 主线

- `w_distill`：student 对齐 teacher。
- `distill_low_only=true`：蒸馏仅在低频上对齐，避免高频被过度抹平。
- `distill_cross_domain_only=true`：只对跨域样本蒸馏，防止同域锁死。

### 3.2 风格约束

- `w_gram`：风格统计约束（增强风格纹理一致性）。
- `w_moment`：通道统计约束（均值/方差等）。
- `w_code`：style code 闭环约束，防止模型忽略风格条件。
- `w_push` + `push_margin`：把 student 结果从源域风格原型推离。

### 3.3 结构约束（已重构为可配置）

- `w_struct`：结构对齐。
- `struct_loss_type`：`l1` 或 `mse`。
- `struct_lowpass_strength`：低通强度（0~1，可做“全匹配 vs 低频匹配”的连续插值）。

### 3.4 cycle 约束（可配置低通强度）

- `w_cycle`
- `cycle_loss_type`：`l1` 或 `mse`
- `cycle_lowpass_strength`：0~1，数值越大越偏低频约束。
- `cycle_edge_strength`：在 cycle 中混入边缘一致性。

### 3.5 细节约束

- `w_edge`：边缘约束（Sobel/Laplacian 方向）。
- `w_delta_tv`：对 `delta` 的 TV 正则，用于抑制棋盘格/块状伪影。
- `w_nce`：局部 token 对齐，缓解“只改颜色、不改结构纹理”的问题。

### 3.6 调度逻辑

以下项都支持 warmup + ramp（前期先不生效，后期逐步接管）：

- `cycle_warmup_epochs` / `cycle_ramp_epochs`
- `nce_warmup_epochs` / `nce_ramp_epochs`
- `struct_warmup_epochs` / `struct_ramp_epochs`
- `edge_warmup_epochs` / `edge_ramp_epochs`

这套调度是当前避免“风格刚学起来就被结构项压回去”的关键。

## 4. 主配置 `src/config.json`（300 epoch）参数详解

以下是当前主实验配置的关键值与意义。

### 4.1 `model` 段

- `latent_channels=4`：latent 通道数（与 SD VAE 对应）。
- `base_dim=128`：主干宽度。
- `lift_channels=128`：输入展开宽度。
- `style_dim=256` / `projector_dim=256`：风格向量维度。
- `num_styles=2`：域数（photo/Hayao）。
- `num_hires_blocks=4`：32x32 段的高分辨率残差块数。
- `num_res_blocks=2`：16x16 主干残差块数。
- `num_groups=4`：归一化分组数。
- `latent_scale_factor=0.18215`：SD latent 约定缩放。
- `residual_gain=0.4`：输出残差总增益（太小会“改不动”，太大会崩）。
- `style_ref_gain=1.0`：reference 风格强度。
- `style_spatial_pre_gain_32=0.45`
- `style_spatial_block_gain_32=0.22`
- `style_spatial_pre_gain_16=0.4`
- `style_spatial_block_gain_16=0.2`
- `use_decoder_spatial_inject=true`
- `style_spatial_dec_gain_32=0.25`
- `style_spatial_dec_gain_out=0.12`
- `use_style_texture_head=true`
- `style_texture_gain=0.45`
- `use_style_delta_gate=true`
- `use_decoder_adagn=true`
- `use_delta_highpass_bias=true`
- `style_delta_lowfreq_gain=0.35`
- `use_style_spatial_highpass=false`
- `normalize_style_spatial_maps=true`
- `use_output_style_affine=true`
- `use_style_force_path=true`
- `style_force_gain=0.7`
- `style_gate_floor=0.7`
- `style_texture_ignore_residual_gain=false`
- `use_style_spatial_blur=true`
- `use_downsample_blur=true`
- `upsample_mode="bilinear"`

### 4.2 `loss` 段

- `w_distill=0.25`
- `distill_low_only=true`
- `distill_cross_domain_only=true`
- `w_gram=30.0`
- `w_moment=2.0`
- `w_code=6.0`
- `w_struct=0.08`
- `w_edge=0.08`
- `struct_loss_type="l1"`
- `struct_lowpass_strength=0.15`
- `w_push=1.5`
- `push_margin=0.2`
- `w_nce=0.35`
- `nce_warmup_epochs=60`
- `nce_ramp_epochs=120`
- `w_idt=0.0`
- `w_cycle=0.3`
- `cycle_loss_type="l1"`
- `cycle_lowpass_strength=0.05`
- `cycle_edge_strength=0.05`
- `w_delta_tv=0.004`
- `cycle_warmup_epochs=60`
- `cycle_ramp_epochs=120`
- `struct_warmup_epochs=45`
- `struct_ramp_epochs=90`
- `edge_warmup_epochs=45`
- `edge_ramp_epochs=90`
- `nce_temperature=0.07`
- `nce_spatial_size=16`
- `nce_max_tokens=512`

### 4.3 `training` 段

- `batch_size=128`（主实验）
- `num_workers=0`（WSL 热点控制策略）
- `cpu_threads=4`, `cpu_interop_threads=2`
- `learning_rate=1.5e-4`
- `min_learning_rate=5e-6`
- `scheduler="cosine"`
- `grad_clip_norm=1.0`
- `num_epochs=300`
- `save_interval=50`
- `full_eval_interval=50`
- `full_eval_on_last_epoch=true`
- `full_eval_batch_size=10`
- `use_amp=true`, `amp_dtype="bf16"`
- `allow_tf32=true`
- `channels_last=true`
- `use_compile=false`
- `fused_adamw=true`

### 4.4 `data`/`checkpoint`/`inference`

- `data_root="../../latent-256"`
- `style_subdirs=["photo","Hayao"]`
- `virtual_length_multiplier=3`
- `save_dir="../full_300_distill_low_only_v1"`
- 推理为单步 latent 迁移：`num_steps=1`, `use_cfg=false`

## 5. overfit50 配置说明（`src/overfit50.json`）

overfit50 与主实验的差异：

- 更小数据（`../../latents_overfit50`）
- 更短训练（`num_epochs=60`）
- 更快评测（`full_eval_interval=20`）
- 更激进采样覆盖（`virtual_length_multiplier=100`）
- 结构约束整体更弱（便于先观察风格注入是否生效）

用于结论：
- overfit50 主要验证“模型是否能学会风格方向”；
- 主实验配置才用于稳定性、泛化和画质平衡。

## 6. 常见问题与定位手册

### 6.1 现象：loss 先降后升

常见原因：
- warmup 结束后 `nce/cycle/struct/edge` 开始爬升，整体 loss 分量变多；
- 这不一定代表训练坏掉，要看子项是否符合预期（例如风格项继续改善、结构项接管）。

检查顺序：
1. 看 `w_cycle_eff / w_nce_eff / w_struct_eff / w_edge_eff`。
2. 看 `distill/code/gram/push` 是否仍在合理区间。
3. 看 full_eval 的 photo->Hayao 指标是否提升。

### 6.2 现象：雾化、细节发软

优先调整：
- 降 `struct_lowpass_strength` 和 `cycle_lowpass_strength`；
- 降 `w_distill`（避免 teacher 过度平滑传递给 student）；
- 适当提高 `w_edge`，保边缘。

### 6.3 现象：棋盘格/块状伪影

优先调整：
- 提高 `w_delta_tv`（小步加，如 0.004 -> 0.006）；
- 确保 `upsample_mode=bilinear` 且 `use_downsample_blur=true`；
- 降低过激风格注入增益（`style_force_gain`, `style_texture_gain`）。

### 6.4 现象：风格方向失效（两种 style 输出很像）

优先调整：
- 提高 `w_code`, `w_push`；
- 降低过早结构约束（延后 warmup/ramp）；
- 确认 `distill_cross_domain_only=true` 仍开启。

## 7. 推荐调参流程（建议固定模板）

1. 先跑 overfit50，确认 `photo->Hayao` 与 `Hayao->photo` 都明显可分。
2. 再跑主实验 300 epoch，按 50 epoch 间隔看趋势。
3. 每次只改一组参数（风格组/结构组/频率组），不要同时全动。
4. 保留 `summary_history.json` 做多轮统计，避免单次波动误判。

## 8. 一句话结论

当前设计不是“风格转不出来”，核心挑战是“风格强度、结构保持、纹理画质”三者平衡。
最有效策略是：
- teacher/student 维持风格能力迁移；
- 结构项延后接管；
- 低频约束不过强；
- 边缘与 TV 抑制伪影。

这也是当前 `config.json` 与 `overfit50.json` 的设计基线。
