# VAE 扩展实验备忘

更新时间：2026-05-25 22:05

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
