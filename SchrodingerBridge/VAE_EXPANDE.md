# VAE 扩展实验备忘

更新时间：2026-05-25 23:00

## 当前目标

按顺序推进三阶段：先把 SDXL 4 通道后端榨干，再进入 FLUX1 16 通道，最后进入 FLUX2 32 通道。每一阶段必须保留配置、源码 snapshot、训练日志、eval CSV、最佳 ckpt 和失败原因记录。

## 当前状态

- 远程机器：`administrator@100.115.18.62 -p 2222`
- 远程仓库：`I:\Github\Latent_Style\SchrodingerBridge`
- 本地仓库：`G:\GitHub\Latent_Style\SchrodingerBridge`
- 三套 latent 已存在；其中原 `latent-256-sdxl` 已验证为坏编码，需要重建：
  - `latent-256-sdxl`：4ch，`scaling_factor=0.13025`，`shift_factor=0.0`，采样 finite ratio 为 0，弃用。
  - `latent-256-sdxl-fp32`：计划重建，使用 `sdxl-fp32`。
  - `latent-256-flux1`：16ch，`scaling_factor=0.3611`，`shift_factor=0.1159`
  - `latent-256-flux2`：32ch，`scaling_factor=0.18215`，`shift_factor=0.0`
- 首轮旧任务 `LANCET_VAE_Backend_256_Probe` 已结束：
  - SDXL：第 1 step 出现 `style_spatial_id_16` 非有限梯度。
  - FLUX1/FLUX2：旧配置仍按 4 通道建模，遇到 16/32 通道 mismatch。

## 硬盘清理

2026-05-25 21:31 已执行远程清理：

- manifest：`exp\_cleanup_manifests\vae_backend_cleanup_20260525_213120`
- 候选：225 项
- 释放候选规模：8.676GB
- `I:` 盘 free 从约 193.9GB 增至约 203.2GB
- 已验证保留：
  - `I:\Github\Latent_Style\latent-256-sdxl\manifest.json`
  - `I:\Github\Latent_Style\latent-256-flux1\manifest.json`
  - `I:\Github\Latent_Style\latent-256-flux2\manifest.json`
  - `exp\diffeomorphic_tangent_sweep\t01_ws0p03_g6_nl0p05\config.json`

清理规则：保护 `vae_backend_256`、`diffeomorphic_tangent_sweep`、`frontier_decision_tree_8h`、`orthogonal_budget36`、论文主点和视频对比目录；删除非保护旧实验的非代表 ckpt 和已由 summary/metrics 记录的 `full_eval/images`。

## 监控

本地定时任务：

- 任务名：`Codex_VAE_Backend_StatusWatch`
- 周期：每 10 分钟
- 脚本：`tools\experiments\check_vae_backend_remote_status.ps1`
- 输出：`exp\vae_backend_256_status\status.md`、`status_raw.txt`、`heartbeat.csv`
- 状态：2026-05-25 21:31 已创建并成功写出 status。

## SDXL 阶段

原则：SDXL 仍是 4ch latent，先解决数值尺度问题，不混入 FLUX。

2026-05-25 21:34 诊断更新：

- 原始 `sdxl_s0_stability` 在 terminal SWD 为 0 时仍 first-step NaN。
- numeric debug 显示首轮 `semantic_attn` 先出现 NaN；加入 attention logit clamp 后，`semantic_attn` 已 finite，但 `style_spatial_id_16` 反传仍 NaN。
- 因此下一步切换为 `sdxl_s0_minimal`：关闭 semantic body blocks 和 diffeomorphic head，使用真实 `terminal_swd_weight=1`，先验证 SDXL encode/train/eval 主链闭环。

2026-05-25 21:41 诊断更新：

- `sdxl_s0_minimal` 仍出现 non-finite loss。
- 对 `latent-256-sdxl` 采样检查后发现所有抽样 latent 的 finite ratio 为 0，说明旧 SDXL latent 编码本身损坏。
- 已将 SDXL 路线切到 `sdxl-fp32`，目标 latent root 改为 `latent-256-sdxl-fp32`，下一步重新 encode 后再 smoke。

2026-05-25 22:05 诊断更新：

- `latent-256-sdxl-fp32` 已完成全量重建，采样检查 finite ratio 为 1.0；各风格 latent std 约 0.73-0.83，最大绝对值约 2.96-3.81，尺度正常。
- `sdxl_s0_minimal` 1 epoch smoke 已稳定完成，无 non-finite：
  - 训练：323 step，约 99 秒，epoch loss 0.0999，terminal SWD 0.0987。
  - ckpt：`exp\vae_backend_256_sdxl_smoke\sdxl_s0_minimal\epoch_0001.pt`
  - eval：`clip_style=0.6677`，`content_lpips=0.3005`，`EC=0.4671`。
- 自动 eval 首次失败原因是远程 `run_evaluation.py` 尚未同步本地 import-path 修复；手动 rerun 已成功。已在 runner 中显式注入 `PYTHONPATH=<repo>\src`，避免后续子进程再依赖环境偶然性。

2026-05-25 22:08 执行更新：

- 已将本地源码/脚本同步到远程 3060。
- 已通过计划任务 `LANCET_VAE_Backend_256_SDXL` 启动完整 SDXL 阶段：
  - action：`cmd.exe /c "I:\Github\Latent_Style\SchrodingerBridge\start_remote_vae_sdxl_stages.bat"`
  - variants：`sdxl_s0_minimal,sdxl_s0_minimal_diffeo,sdxl_s0_stability,sdxl_s1_light_swd,sdxl_s2_balanced,sdxl_s3_style_push`
  - epochs：8
  - eval epochs：6,7,8
  - out root：`exp\vae_backend_256_sdxl`
- 启动后已看到 `src\run.py --config exp\vae_backend_256_sdxl\sdxl_s0_minimal\config.json` 进程。

2026-05-25 23:00 执行更新：

- 已中止低显存 SDXL 队列：`LANCET_VAE_Backend_256_SDXL` 停止后 GPU 回到空闲。
- 已完成并保留的低显存归档结果：
  - `sdxl_s0_minimal`：epoch 6/7/8 均成功；最佳约 `clip_style=0.6671`、`content_lpips=0.3017`、`EC=0.4657`。
  - `sdxl_s0_minimal_diffeo`：epoch 6/7/8 均成功；最佳约 `clip_style=0.6673`、`content_lpips=0.3203`、`EC=0.4534`。
- 结论：minimal 路线证明 SDXL fp32 latent 可稳定训练/eval，但显存仅约 4.2GB，且 style 明显不足；不作为冲 SaMST 的主线。
- 新约束：后续正式 SDXL 运行以 9.0-10.8GB peak VRAM 为目标；runner 已规划记录每次训练/eval 的 `peak_gpu_memory_mb`。
- 下一步：运行 `sdxl_mem_b96/b128/b160/b192` memory ladder，每档 1 epoch、30 batch，选定 10G 级默认 batch 后启动 `sdxl_t01_recover/style_push/content_guard/t01_fullish`。

2026-05-25 23:06 执行更新：

- 10G memory ladder 有效结果：
  - `batch=96`：`peak_gpu_memory_mb=7553`，稳定但低于目标。
  - `batch=128`：`peak_gpu_memory_mb=9795`，稳定，落在 9.0-10.8GB 目标区间。
  - `batch=160`：`peak_gpu_memory_mb=11808`，稳定但越过 10.8GB 上限。
  - `batch=192`：运行中瞬时约 11.78GB，因 160 已越界而主动停止，不作为正式配置。
- 决策：SDXL 10G 阶段默认训练 batch 固定为 128。
- 已启动 `LANCET_VAE_Backend_256_SDXL_10G_Candidates`：
  - variants：`sdxl_t01_recover,sdxl_style_push,sdxl_content_guard,sdxl_t01_fullish`
  - epochs：8
  - eval epochs：6,7,8
  - out root：`exp\vae_backend_256_sdxl_10g_candidates`
- 启动后 `sdxl_t01_recover` 已进入真实训练，观测显存约 9.65GB，符合 10G 目标。

2026-05-25 23:05 监控更新：

- `LANCET_VAE_Backend_256_SDXL_10G_Candidates` 正在运行。
- 当前子进程为 `sdxl_t01_recover`，训练到 epoch 6/8；GPU 约 `9697/12288MB`，util 约 `95%`，温度约 `65C`。
- 最新 log 无 OOM / non-finite，loss 主要由 terminal SWD 与 kinetic 项组成，训练过程稳定。
- 候选结果 CSV 尚未出现；原因是 eval 只在 ckpt 6/7/8 生成后启动。下一步继续等待 epoch 6 eval 写入，再判断 style 是否突破 minimal 路线。

2026-05-25 23:17 结果更新：

- `sdxl_t01_recover` 已完成 epoch 6/7/8 训练与 eval，训练 peak VRAM 记录为 `9906MB`，符合 10G 目标。
- 指标：
  - epoch 6：`clip_style=0.6484`，`content_lpips=0.6539`，`EC=0.2244`
  - epoch 7：`clip_style=0.6481`，`content_lpips=0.6454`，`EC=0.2298`
  - epoch 8：`clip_style=0.6541`，`content_lpips=0.6485`，`EC=0.2299`
- 结论：`sdxl_t01_recover` 虽然稳定且显存达标，但并没有恢复 SD15 t01 的双指标；style 低于 minimal，content_lpips 也明显恶化。暂判为 SDXL latent 尺度/解码可见风格与原 t01 结构不相容，不作为主线会师点。
- 队列已自动进入 `sdxl_style_push`，目前训练稳定；下一步重点看强 terminal SWD + `[3,5,7]` patch 是否能把 style 拉回 `0.70+`。

2026-05-25 23:26 路线修正：

- `sdxl_style_push` 已完成 epoch 6/7/8，训练 peak VRAM 约 `10050MB`，但结果仍差：
  - epoch 6：`clip_style=0.6621`，`content_lpips=0.7077`，`EC=0.1936`
  - epoch 7：`clip_style=0.6582`，`content_lpips=0.6986`，`EC=0.1984`
  - epoch 8：`clip_style=0.6614`，`content_lpips=0.7004`，`EC=0.1982`
- 已停止后续 `sdxl_content_guard/sdxl_t01_fullish`，避免继续在已证伪的强 t01 迁移路线上耗时。
- 新判断：SDXL 当前最可靠基座是 `sdxl_s0_minimal` 的 `~0.667 style / ~0.302 LPIPS` 内容保真点。后续应从这个点向外推，而不是直接复制 SD15 t01。
- 新实验组 `sdxl_minimal_scale_switches`：
  - 保持 minimal 架构：无 semantic body、无 diffeomorphic head、zero-init output head。
  - 只扫温和 SWD、patch、`model_latent_scale_factor`、`vae_decode_scale`、推理步数。
  - 训练 batch 固定 128；eval batch 提升到 16；sleep 监控节奏改为 120 秒。

2026-05-25 23:33 优先级修正：

- `sdxl_minimal_scale_switches` 已在首组 epoch 3 前停止；原因是当前更优先的问题不是 decode scale，而是 minimal 基座上的 loss 比重与架构开关。
- 新队列 `sdxl_minimal_loss_arch` 从 `sdxl_s0_minimal` 内容好点出发，先只测：
  - loss 比重：降低 `w_kinetic`、提高 terminal SWD、增加 `w_content_anchor`。
  - SWD 形式：`spectral_orthogonal`、micro/macro patch 权重。
  - 架构开关：`num_res_blocks=1`、极低 `style_spatial_pre_gain_16=0.05`、极弱 diffeomorphic head。
- 目标不是一次冲到 0.72，而是先找到哪类开关能在 `content_lpips ~0.30` 附近提高 style；找到有效方向后再做 scale / 推理参数第二层扫描。

2026-05-25 23:48 首组 loss 结果：

- `sdxl_min10g_loss_kin025_swd2` 已完成 epoch 6/7/8，peak VRAM `7755MB`；minimal 架构较轻，显存低于 10G 但数值稳定。
- 指标：
  - epoch 6：`clip_style=0.6716`，`content_lpips=0.3289`，`EC=0.4507`
  - epoch 7：`clip_style=0.6714`，`content_lpips=0.3304`，`EC=0.4496`
  - epoch 8：`clip_style=0.6714`，`content_lpips=0.3304`，`EC=0.4495`
- 相对 `sdxl_s0_minimal`：style 约 `+0.0044`，但 LPIPS 约 `+0.0285`，EC 下降。结论是降低 `w_kinetic` 能让风格略动，但不是高性价比方向；需要看 `content_anchor`、spectral SWD 或轻架构开关是否能更好地保内容。

2026-05-25 23:55 目标修正：

- 明确主目标：先冲 `clip_style > 0.72`，然后看 `content_lpips < 0.53`；不再把 `LPIPS ~0.30` 当作第一约束。
- 已停止 mild loss/architecture 队列；其中 `sdxl_min10g_loss_kin05_swd4` 的 `train_failed_2` 是人为中断，不是自然失败。
- 新队列 `sdxl_style_first`：
  - batch 固定 160，目标显存靠近 10G。
  - 从 minimal 内容好基座出发，但把 terminal SWD 提到 8/12/16，并使用 `[1,3,5,7]` patch。
  - 开关重点：低 `w_kinetic`、spectral high-frequency SWD、micro/macro 权重、tiny spatial prior、1 个 residual block、弱 diffeomorphic head。
  - 判定顺序：先看是否有点能接近或超过 `clip_style=0.72`；若有，再用 content/scale/推理参数把 LPIPS 压到 `<0.53`。

2026-05-26 00:06 style-first 首条结果：

- `sdxl_stylefirst_swd8_p1357_k025` epoch 6：peak VRAM `9582MB`，`clip_style=0.6763`，`content_lpips=0.4127`，`EC=0.3972`。
- 相对 minimal：style 从 `~0.667` 到 `0.676`，且 LPIPS 仍低于 SaMST 阈值 `0.53`；说明 style-first 的强 SWD/patch/低 kinetic 方向比 mild loss 更有效。
- 但离首要目标 `clip_style>0.72` 仍差 `~0.044`，需要继续看 swd12/swd16、spectral、micro/macro 和轻架构开关。

2026-05-26 00:10 历史文档回查后的修正：

- 回查 `2026-05-22 t01 主线高层备忘` 和 `EXPERIMENT_PLAN.md` 后确认：真正能把 style 顶到 `0.72+` 的不是 minimal 基座，而是完整 `t01` style actuator：
  - `num_res_blocks=4`
  - `style_spatial_pre_gain_16=0.35`
  - `use_diffeomorphic_stroke=true`
  - `diffeomorphic_color_strength=0.85`
  - `diffeomorphic_warp_strength=0.03`
  - `diffeomorphic_texture_gate_strength=6.0`
  - `diffeomorphic_normal_leak=0.05`
- `EXPERIMENT_PLAN.md` 的高 style 预测来自 `w_swd/w_kin` 比值，尤其是 `W30/K0.5` 与 `W40/K0.25`；之前只开到 `W8/K0.25`，明显不是历史强配方。
- 已停止半强 `sdxl_style_first` 队列，保留结果作为“minimal-style actuator 不够强”的证据。
- 新队列 `sdxl_t01_stylemax` 将直接测试完整 t01 + 历史强 style 开关：
  - `W30/K0.5`
  - `W40/K0.25`
  - all-style boosters：`w_style_energy_floor`、`w_style_contrastive`、`w_residual_style_direction`、`w_spectral_amplitude=0.05`、`w_phase_separation`、`w_fourier_phase_lock`、`latent_canvas_strength=0.05`
  - `terminal_num_steps=8`
  - `factorized_amp` diffeomorphic head
  - `output_moment_match`
- 目标判定顺序保持：先看 `clip_style > 0.72`，再筛 `content_lpips < 0.53`。

阶段队列：

0. `sdxl_s0_minimal`
   - batch 32，LR 1e-5，terminal SWD 1
   - `num_res_blocks=0`
   - `use_diffeomorphic_stroke=false`
   - `zero_init_output_head=true`
   - SWD patch `[1, 3]`，projection 16
1. `sdxl_s0_stability`
   - batch 48，LR 2e-5，terminal SWD 0
   - `style_spatial_pre_gain_16=0.10`
   - `diffeomorphic_warp_strength=0.01`
   - 关闭 AMP、channels_last、gradient checkpointing
2. `sdxl_s1_light_swd`
   - terminal SWD 2，其余保持保守
3. `sdxl_s2_balanced`
   - LR 3e-5，terminal SWD 5
   - `style_spatial_pre_gain_16=0.22`
   - `diffeomorphic_warp_strength=0.02`
4. `sdxl_s3_style_push`
   - batch 40，LR 4e-5，terminal SWD 8
   - `style_spatial_pre_gain_16=0.28`
   - `diffeomorphic_warp_strength=0.03`

执行顺序：

- 已完成 `sdxl_s0_minimal` 1 epoch smoke，eval epoch 1。
- 下一步跑 `sdxl_s0_minimal`、`sdxl_s0_minimal_diffeo`、`sdxl_s0_stability`、`sdxl_s1_light_swd`、`sdxl_s2_balanced`、`sdxl_s3_style_push` 的 8 epoch，eval 6/7/8。
- 若 S0 smoke 仍非有限，下一步降 LR 到 1e-5，并临时关闭 diffeomorphic stroke 或进一步压低 spatial gain。

## FLUX 后续

SDXL 阶段结束后再进入：

1. FLUX1 native 16ch smoke。
2. FLUX1 `16 -> 6 -> 16` adapter。
3. FLUX1 `16 -> 4 -> 16` adapter。
4. FLUX2 native 32ch smoke。
5. FLUX2 `32 -> 6 -> 32` adapter。
6. FLUX2 `32 -> 4 -> 32` adapter。

高通道默认加入 latent whitening/scale 校准、低 SWD、低 batch；只有 native 确认不行后再判断 adapter 是主路线。
## 2026-05-26 SDXL 通道/Patch 诊断更新

- 新增诊断脚本：`tools/experiments/diagnose_sdxl_channels_and_patches.py`。
- 远程完成全量通道和 patch SNR 诊断：`exp\vae_backend_256_status\sdxl_channel_patch_diagnostic_full_fg`。
- 本地同步结果：`G:\GitHub\Latent_Style\SchrodingerBridge\exp\vae_backend_256_status\sdxl_channel_patch_diagnostic_full_fg`。
- 诊断覆盖 `10361` 个 SD15/SDXL latent；photo→style 传输熵先用每风格 `128` 对估计。
- 关键结论：SDXL 4ch 不是 SD15 4ch 的同构坐标；SDXL channel 2 梯度能量最高，channel 3 有明显正均值偏置，旧的 `mean(channel)` texture guide 在 SDXL 上不应继续作为默认架构假设。
- 代码已加入 VAE-aware diffeomorphic guide：`diffeomorphic_guide_mode`、`diffeomorphic_guide_channel`、`diffeomorphic_guide_weights`。
- 支持 guide：`mean`、`whitened_mean`、`sdxl_channel2`、`sdxl_pca1`、`sdxl_grad`。
- 新增 SDXL channel-aware 候选：`sdxl_channel_pca1_diffeo`、`sdxl_channel_ch2_diffeo`、`sdxl_channel_factorized_pca1`。
- 下一步不再盲目复制 SD15 `[3,5,7,15]` + raw mean guide；优先测试 `[1,2,3,5]` patch 与 PCA1/channel2 guide 的 6ch/7ch 迁移。
- 2026-05-26 01:35 修正：PCA1/channel2 两条 channel-aware 6ch 结果均约 `clip_style=0.634`、`content_lpips=0.656`，明显劣于 minimal/style-first plain4。当前先暂停通道增强，回到朴素 4ch base；新增 `sdxl_plain4_base_swd16_p1235`、`sdxl_plain4_base_swd24_p1235`、`sdxl_plain4_base_swd32_p1235`、`sdxl_plain4_res1_swd24_p1235`，目标先把 plain4 推到 `clip_style≈0.70+`。
## 2026-05-26 KL-f4 / SD15-EMA route switch

- SDXL is paused as the main route: minimal, stronger SWD, OMF, and t01max variants stayed below the style target; the strongest style-heavy t01max point reached only about `clip_style=0.679` and pushed LPIPS to about `0.75`.
- New detailed note: `docs/experiments/2026-05-26-klf4-ema-vae-backend.md`.
- KL-f4 loader is now working with the legacy CompVis checkpoint. It produces finite `3x64x64` latents and uses `scaling_factor=1.0`.
- KL-f4 full latent set is encoded at remote `I:\Github\Latent_Style\latent-256-kl-f4`.
- KL-f4 memory ladder:
  - batch 48: OOM/killed, peak `12086MB`.
  - batch 40: train ok, peak `12049MB`.
  - batch 32: train ok, peak `11135MB`.
  - batch 28: train ok, peak `9676MB`, selected for formal runs.
  - batch 24: train ok, peak `8443MB`.
- Active formal KL-f4 run: `exp\vae_backend_256_klf4_ema_full\klf4_t01_w20_b28`, 8 epochs, eval `6/7/8`.
- Next after KL-f4 readout: run `sd-vae-ft-ema` as the f8 non-MSE control.

## 2026-05-26 EMA content guard update

- KL-f4 first formal run completed at the 10G target but underperformed on style: best region was only `clip_style~0.660`, `content_lpips~0.459`, `EC~0.356`. It is content-preserving but not a style route in its current actuator/latent convention.
- SD15 EMA first formal run restored high style: `clip_style~0.722-0.724`, but LPIPS was too high at `~0.576`.
- EMA content guard queue is active at remote `exp\vae_backend_256_ema_content_guard`.
- `ema_guard_w16` completed:
  - epoch 6: `clip_style=0.7183`, `content_lpips=0.5230`, `EC=0.3426`
  - epoch 7: `clip_style=0.7207`, `content_lpips=0.5256`, `EC=0.3419`
  - epoch 8: `clip_style=0.7192`, `content_lpips=0.5235`, `EC=0.3427`
- Current interpretation: lowering terminal SWD and visible actuator strength is the right EMA direction. It turns EMA from high-style/poor-content into a near-target balanced backend. It is close to original t01 but not yet clearly better.
- Running now: `ema_guard_w18_patch357`, followed by `ema_guard_w20_lowwarp`.

## 2026-05-26 EMA fragmentation attribution

- Visual review of `exp\vae_backend\vae_backend_256_status\ema_guard_grids_review\compare_summary_grid_first.png` shows that the recent high-style EMA outputs are locally fragmented, especially around tree branches, windows, and body edges.
- `ema_guard_w20_lowwarp` still fragments with `diffeomorphic_warp_strength=0.01`, so warp alone is not a sufficient explanation.
- Current hypothesis: high terminal SWD and patch pressure are injecting local style statistics through residual/color paths; warp amplifies the edge damage but is not the sole source.
- Added `ema_plain4_w20_anchor`, a no-diffeo W20 counterpart to `ema_guard_w20_lowwarp`, to isolate whether the same damage survives without warp channels.
- Remote queue launched at `I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_arch_attribution`:
  - `ema_hardcontent_w18_anchor`
  - `ema_hardcontent_w24_anchor`
  - `ema_amp_only_w24_anchor`
  - `ema_identity_w24_anchor`
  - `ema_plain4_w20_anchor`
  - `ema_plain4_spectral_iso_w32`
- Decision logic:
  - if no-diffeo W20 is still broken, focus on SWD patch/style pressure and residual/color isolation;
  - if amplitude-only keeps style and repairs geometry, continue with factorized amplitude and near-zero warp;
  - if barrier improves LPIPS but loses style, tune patch bands and style pressure around the barrier.
- `ema_hardcontent_w18_anchor` completed:
  - epoch 6: `clip_style=0.6671`, `content_lpips=0.4945`, `EC=0.3372`
  - epoch 7: `clip_style=0.6677`, `content_lpips=0.4944`, `EC=0.3376`
  - epoch 8: `clip_style=0.6672`, `content_lpips=0.4944`, `EC=0.3373`
- Interpretation: strong edge/structure barrier can reduce LPIPS, but it blocks style transport too aggressively. It is useful evidence, not a main route.
- Reprioritized the remote queue to run `ema_amp_only_w24_anchor`, then `ema_plain4_w20_anchor`, then `ema_plain4_spectral_iso_w32`, before returning to the remaining barrier-style checks.
- `ema_amp_only_w24_anchor` completed:
  - epoch 6: `clip_style=0.6682`, `content_lpips=0.5092`, `EC=0.3280`
  - epoch 7: `clip_style=0.6708`, `content_lpips=0.5113`, `EC=0.3278`
  - epoch 8: `clip_style=0.6695`, `content_lpips=0.5123`, `EC=0.3265`
- Interpretation: factorized amplitude with `warp_strength=0` is not sufficient under the current anchor/barrier design. It preserves content better than plain EMA t01, but it loses too much style.
- Running now: `ema_plain4_w20_anchor`, the direct no-diffeo W20 attribution probe.
- `ema_plain4_w20_anchor` completed:
  - epoch 6: `clip_style=0.7007`, `content_lpips=0.4215`, `EC=0.4053`
  - epoch 7: `clip_style=0.7001`, `content_lpips=0.4240`, `EC=0.4033`
  - epoch 8: `clip_style=0.7007`, `content_lpips=0.4248`, `EC=0.4030`
- Visual attribution: this is the cleanest recent EMA grid. It removes most of the fragmentation seen in `ema_guard_w18_patch357`, `ema_guard_w20_lowwarp`, and `ema_amp_only_w24_anchor`, while crossing the LPIPS target. It misses style by about `0.02`.
- Current conclusion: severe fragmentation is not caused by W20 SWD alone. It is mainly introduced by the stronger diffeomorphic/geometry style actuator near content boundaries. The no-diffeo 4ch route is the clean base for the next style push.
- Running now: `ema_plain4_spectral_iso_w32`, testing whether no-diffeo style pressure can reach `clip_style > 0.72` without losing the content win.
- `ema_plain4_spectral_iso_w32` completed:
  - epoch 6: `clip_style=0.7025`, `content_lpips=0.4342`, `EC=0.3974`
  - epoch 7: `clip_style=0.7024`, `content_lpips=0.4371`, `EC=0.3954`
  - epoch 8: `clip_style=0.7026`, `content_lpips=0.4379`, `EC=0.3949`
- Interpretation: W20 -> W32 pure 4ch only gains about `0.002` style while preserving content. Pure residual 4ch seems capped near `clip_style~0.70`.
- Stopped the remaining duplicate `hardcontent/identity` attribution tasks to save time.
- New remote queue launched at `I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_cleanbase_push`:
  - `ema_plain4_styleloss_w24`
  - `ema_plain4_canvas_w24`
  - `ema_microdiffeo_w20`
  - `ema_microdiffeo_styleloss_w20`
- Goal of this queue: keep the clean-base LPIPS band (`~0.42-0.44`) and lift style from `~0.70` to `>0.72` with the smallest possible style actuator.
- `ema_plain4_styleloss_w24` completed: best `clip_style=0.7045`, best content-compatible row `epoch 6: clip_style=0.7044, content_lpips=0.4473`.
- `ema_plain4_canvas_w24` completed: best `clip_style=0.7045`, best content-compatible row `epoch 6: clip_style=0.7045, content_lpips=0.4459`.
- `ema_microdiffeo_w20` completed: failed, all rows around `clip_style=0.6606-0.6609`, `content_lpips=0.464-0.471`.
- Stopped `ema_microdiffeo_styleloss_w20` early because the microdiffeo branch clearly moved in the wrong direction.
- Updated conclusion: clean 4ch EMA has a content-preserving ceiling around `clip_style=0.70-0.705`; style-side training losses and latent canvas do not close the remaining `~0.016-0.020` gap. Next cheap check is inference strength on `ema_plain4_w20_anchor` because its e6 LPIPS (`0.4215`) leaves margin for a small style-strength increase.
- Inference strength sweep on `ema_plain4_w20_anchor` epoch 6 completed:
  - `style_strength=1.05`: `clip_style=0.7006`, `content_lpips=0.4216`
  - `style_strength=1.10`: `clip_style=0.7005`, `content_lpips=0.4215`
  - `style_strength=1.15`: `clip_style=0.7005`, `content_lpips=0.4216`
  - `style_strength=1.20`: `clip_style=0.7005`, `content_lpips=0.4215`
- Interpretation: plain4 W20 style is saturated at inference; scaling does not release hidden style. The next route must be architectural, not just inference strength.
- New non-geometric style actuator queue launched at `I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_moment_dynamic`:
  - `ema_plain4_moment_w20`
  - `ema_plain4_premoment_w20`
  - `ema_plain4_dynamic_w24`
  - `ema_plain4_dynamic_moment_w20`
- Rationale: moment matching and dynamic output heads can add style statistics/capacity without spatial warp. This tests whether the missing `~0.02` style can be recovered without reintroducing fragmentation.
- 2026-05-26 07:55 update:
  - `ema_plain4_moment_w20` epoch 6 eval is effectively identical to clean plain4: `clip_style=0.7006`, `content_lpips=0.4219`, `EC=0.4050`.
  - Visual grid remains mostly clean, but it does not increase style; endpoint moment matching is not enough.
  - Code-level fragmentation attribution: the diffeomorphic branch applied `diffeomorphic_metric_mask_gamma` to color delta but not to `spatial_warp` before `grid_sample`; meanwhile `_texture_tangent_warp` activates more strongly on high-gradient edges. This directly explains local broken branches/windows/body contours.
  - Patch applied: metric mask now also gates `spatial_warp` before `grid_sample` in both standard and factorized diffeomorphic paths.
  - Added targeted variant `ema_edgesafe_diffeo_w20`: same clean EMA base with a weak factorized diffeo style actuator, but edge/metric-masked spatial warp. This is the next direct test of whether warp can add the missing style without boundary fragmentation.
  - Non-geometric queue result:
    - `ema_plain4_moment_w20`: best `clip_style=0.7007`, LPIPS `0.422-0.425`; no gain over clean plain4.
    - `ema_plain4_premoment_w20`: best `clip_style=0.7008`, LPIPS `0.422-0.425`; no gain.
    - `ema_plain4_dynamic_w24`: best `clip_style=0.7127`, but LPIPS `0.4711`; dynamic head increases style capacity but damages content.
    - `ema_plain4_dynamic_moment_w20`: best content-compatible row epoch 6, `clip_style=0.7070`, `content_lpips=0.4366`, `EC=0.3984`; moment regularizes dynamic head back toward clean-base but loses style.
  - Started remote targeted run `exp\vae_backend\ema_edgesafe_diffeo\ema_edgesafe_diffeo_w20`.
  - `ema_edgesafe_diffeo_w20` completed:
    - epoch 6: `clip_style=0.6609`, `content_lpips=0.4768`, `EC=0.3458`
    - epoch 7: `clip_style=0.6619`, `content_lpips=0.4734`, `EC=0.3485`
    - epoch 8: `clip_style=0.6610`, `content_lpips=0.4707`, `EC=0.3499`
  - Visual read: edge-safe warp avoids the sharp local tearing pattern, but it washes out texture/style and still loses content. This confirms that the old diffeomorphic branch's high-style effect came from boundary/edge motion; once edge motion is masked, it no longer carries useful style.
  - Updated attribution: severe fragmentation is a warp/edge-resampling artifact, not terminal SWD alone. However, the recoverable style gap cannot be solved by simply making warp safer; the safer warp path collapses style.

## 2026-05-26 KL-f4 fair-mode follow-up

- Audit found that `latent-256-kl-f4-mode` was missing; prior KL-f4 result used posterior samples only.
- Existing KL-f4 sample-latent summaries are:
  - epoch 6: `clip_style=0.6601`, `content_lpips=0.4586`
  - epoch 7: `clip_style=0.6584`, `content_lpips=0.4595`
  - epoch 8: `clip_style=0.6595`, `content_lpips=0.4593`
- The previous `vae_backend_256_klf4_ema_full` CSV was mixed/overwritten by EMA rows, so future KL-f4 evidence must use a dedicated ledger.
- Started remote fair KL-f4 run at `exp\vae_backend\klf4_mode_fair`:
  - variant: `klf4_mode_stylepush_w40_p5917`
  - latent root: `latent-256-kl-f4-mode`
  - latent mode: posterior `mode`
  - patch sizes: `[5, 9, 17]`
  - terminal SWD: `40`
  - batch: `28`
  - eval epochs: `6/7/8`
- Purpose: verify whether KL-f4's 64x64 f4 topology only failed because sample-latent noise and copied SD15 patch scales were unfair.

### KL-f4 fair-mode result

- `klf4_mode_stylepush_w40_p5917` completed on the remote 3060:
  - epoch 6: `clip_style=0.6522`, `content_lpips=0.4887`, `EC=0.3335`
  - epoch 7: `clip_style=0.6540`, `content_lpips=0.4852`, `EC=0.3367`
  - epoch 8: `clip_style=0.6537`, `content_lpips=0.4852`, `EC=0.3365`
- This was the fair KL-f4 check: posterior mode latents, f4-scale larger patches `[5, 9, 17]`, strong terminal SWD `40`, and 10GB-class training memory.
- Conclusion: KL-f4 is not a promising 256x256 backend for the current LANCET objective. It is worse than the earlier KL-f4 sample-latent baseline (`clip_style~0.66`) and far below SD15 EMA (`clip_style~0.70-0.72` depending on warp). The f4 spatial grid alone does not recover style; the CompVis KL-f4 latent/statistical geometry is mismatched to the current CLIP-style/SWD objective.
- Action: stop spending broad sweep budget on KL-f4 unless a fundamentally different wavelet/f4-specific architecture is introduced. Continue with EMA non-geometric style capacity and warp-attribution probes.
- The waiting follow-up queue started automatically after KL-f4:
  - `ema_warptv_diffeo_w20`
  - `ema_dynamic_guard_w28`

### EMA fragmentation follow-up result

- `ema_warptv_diffeo_w20` completed:
  - epoch 6: `clip_style=0.6610`, `content_lpips=0.4764`, `EC=0.3461`
  - epoch 7: `clip_style=0.6616`, `content_lpips=0.4727`, `EC=0.3489`
  - epoch 8: `clip_style=0.6608`, `content_lpips=0.4707`, `EC=0.3498`
- Interpretation: adding warp energy/TV regularization and divergence-free projection does not rescue the diffeomorphic path. It lands in the same weak-style/washed-out region as `ema_edgesafe_diffeo_w20`. This confirms that the useful old diffeo style was the unsafe edge-resampling behavior, not a smooth geometric transport.
- `ema_dynamic_guard_w28` completed:
  - epoch 6: `clip_style=0.7078`, `content_lpips=0.4477`, `EC=0.3909`
  - epoch 7: `clip_style=0.7072`, `content_lpips=0.4650`, `EC=0.3783`
  - epoch 8: `clip_style=0.7075`, `content_lpips=0.4597`, `EC=0.3822`
- Interpretation: the non-warp dynamic head can preserve content if guarded, but style remains capped around `0.708`. The current EMA Pareto front is therefore:
  - clean/content: `ema_plain4_w20_anchor` (`0.7007 / 0.4215`)
  - guarded dynamic: `ema_dynamic_guard_w28` (`0.7078 / 0.4477`)
  - style probe: `ema_plain4_dynamic_w24` (`0.7127 / 0.4711`)
- Next cheap action: inference-strength sweep on the two dynamic checkpoints before launching more training.

### Dynamic inference-strength sweep

- `ema_dynamic_guard_w28` epoch 6:
  - `style_strength=1.05`: `clip_style=0.7076`, `content_lpips=0.4476`
  - `style_strength=1.10`: `clip_style=0.7076`, `content_lpips=0.4477`
  - `style_strength=1.15`: `clip_style=0.7075`, `content_lpips=0.4477`
- `ema_plain4_dynamic_w24` epoch 6:
  - `style_strength=0.85`: `clip_style=0.7056`, `content_lpips=0.4300`
  - `style_strength=0.90`: `clip_style=0.7083`, `content_lpips=0.4439`
  - `style_strength=0.95`: `clip_style=0.7107`, `content_lpips=0.4576`
- Interpretation: inference scaling cannot reveal a hidden `0.72 / 0.45` point. The existing dynamic weights trace a simple tradeoff curve: style rises only as LPIPS leaves the target band.
- Next action: train two no-warp dynamic frontier variants:
  - `ema_dynamic_frontier_w32`: stronger style capacity with moderate guard.
  - `ema_dynamic_frontier_guard_w36`: stronger style pressure but heavier content/edge guard.

### Dynamic frontier training result

- `ema_dynamic_frontier_w32` completed:
  - epoch 6: `clip_style=0.7093`, `content_lpips=0.4690`, `EC=0.3767`
  - epoch 7: `clip_style=0.7082`, `content_lpips=0.4841`, `EC=0.3653`
  - epoch 8: `clip_style=0.7086`, `content_lpips=0.4794`, `EC=0.3689`
- `ema_dynamic_frontier_guard_w36` completed:
  - epoch 6: `clip_style=0.7089`, `content_lpips=0.4637`, `EC=0.3802`
  - epoch 7: `clip_style=0.7077`, `content_lpips=0.4768`, `EC=0.3703`
  - epoch 8: `clip_style=0.7078`, `content_lpips=0.4730`, `EC=0.3730`
- Interpretation:
  - Stronger no-warp dynamic style capacity does not break the `~0.71` style ceiling.
  - More style pressure mostly damages LPIPS; heavier guards do not recover content enough to make the style gain useful.
  - Current best target-compatible EMA point remains `ema_dynamic_guard_w28` e6 (`0.7078 / 0.4477`), not the stronger frontier variants.
- Working EMA conclusion:
  - `sd-vae-ft-ema` is better than KL-f4 and SDXL for this 256x256 objective.
  - It can produce a clean content-preserving branch, but it has not produced `clip_style>0.72` with `content_lpips<0.45`.
  - Unsafe warp can reach style, but that route is visually broken and content-poor.
  - No-warp dynamic routes are visually cleaner but capped below the style target.

## 2026-05-26 Fragmentation attribution and next probes

- Current visual attribution is now strong:
  - `ema_plain4_w20_anchor` is clean and content-preserving (`epoch 6: clip_style=0.7007, content_lpips=0.4215`), so W20 terminal SWD / patch pressure alone does not cause the broken images.
  - `ema_plain4_dynamic_w24` lifts style (`clip_style=0.7127`) but damages content (`content_lpips=0.4711`) without the same sharp edge tearing, so dynamic style capacity causes broad content drift rather than local fragmentation.
  - `ema_edgesafe_diffeo_w20` removes the most dangerous edge motion but collapses style (`clip_style~0.661`) and remains worse than clean plain4. This means the old diffeo style boost was coming from exactly the high-gradient resampling path that breaks branches, windows, limbs, and fine contours.
- Code-level mechanism:
  - `_texture_tangent_warp` gates motion by latent/image gradient, so the warp is intentionally strongest on fragile edges.
  - The output then goes through `grid_sample`; local high-frequency, non-smooth displacement at 256x256 creates visible broken contours.
  - Metric masking now gates `spatial_warp`, but that turns off the risky style mechanism instead of turning it into a useful one.
- Added two targeted probes for the next free GPU slot:
  - `ema_warptv_diffeo_w20`: keep weak diffeo but add raw warp energy/TV tax plus divergence-free projection. This tests whether the artifact is mainly high-frequency displacement discontinuity.
  - `ema_dynamic_guard_w28`: no warp at all; stronger dynamic style head with tighter content/edge guards. This tests the current best non-geometric rescue path.
- Operational note: remote GPU is currently occupied by `klf4_mode_fair`, so these probes are prepared but not launched over it.
