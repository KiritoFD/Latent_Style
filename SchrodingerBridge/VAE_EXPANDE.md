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
