# 616 Tools

这份文档固定记录 616 三阶段实验当前可复用的工具、远程入口、日志位置和运行约束。

## 1. Remote machine

远程 Windows:

```bash
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

远程仓库:

```text
Windows: I:\Github\Latent_Style\SchrodingerBridge
WSL:     /mnt/i/Github/Latent_Style/SchrodingerBridge
```

当前稳定 WSL distro:

```text
Ubuntu-26.04
```

## 2. Core automatic runner

主控脚本:

```bash
tools/experiments/phase616_auto.py
```

本地 WSL 分阶段入口:

```bash
tools/experiments/run_phase616_stage1_auto.sh
tools/experiments/run_phase616_stage2_auto.sh
tools/experiments/run_phase616_stage3_auto.sh
tools/experiments/run_phase616_auto_tree.sh
```

默认行为:

- 每个实验先做 `20 steps` probe
- probe timeout 默认 `40s`
- 自动选 batch, 目标显存 `9.0-10.8 GB`
- `> 11.3 GB` 视为 OOM
- 优先 8/16 倍数 batch
- 启动后 `1 分钟` 做健康检查
- `10 分钟` 估 ETA
- 然后 sleep 到 `ETA - 5 分钟` 再回来看
- 每个 epoch 做 full eval
- 用 `CLIP-S + LPIPS` 做收敛早停

默认三阶段目录:

```text
stage1: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto
stage2: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_ablation_auto
stage3: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_best_auto
```

## 3. Remote scheduled-task entry

3060 远程机不要再手工保活单个 shell. 用 Windows 计划任务直接进 WSL 跑 `.sh`.

远程启动脚本:

```bash
tools/experiments/launch_phase616_stage1_auto_remote.sh
tools/experiments/launch_phase616_stage2_auto_remote.sh
tools/experiments/launch_phase616_stage3_auto_remote.sh
tools/experiments/launch_phase616_auto_tree_remote.sh
```

这些脚本统一复用:

```bash
tools/experiments/launch_remote_wsl_command.py
```

当前验证通过的稳定任务动作:

```text
wsl.exe -d Ubuntu-26.04 --exec /bin/bash -lc "cd /mnt/i/Github/Latent_Style/SchrodingerBridge && bash <script>.sh"
```

约束:

- 计划任务动作直接用 `wsl.exe` 进入 WSL 跑 `.sh`
- 616 阶段执行不要再用 `.ps1` 作为任务动作包装器
- 保留单卡 single-lane guard:
  - 如果已经有 WSL 训练在跑
  - 或 GPU 显存高于 idle ceiling
  - launcher 必须拒绝再开第二条
- runtime ceiling 统一按 `11264 MiB`

## 4. Common launch commands

在远端 WSL 直接跑:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase616_auto_tree.sh
```

从本地发起远程计划任务:

```bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
bash tools/experiments/launch_phase616_auto_tree_remote.sh
```

单独跑阶段 2:

```bash
bash tools/experiments/run_phase616_stage2_auto.sh \
  --stage1-root /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical
```

## 5. Logs and artifacts

关键产物:

```text
<run>/logs/training_*.csv
<run>/full_eval_transfer/clip_lpips_curve.csv
<run>/full_eval_transfer/round2_convergence.json
<run>/auto_run_summary.json
<run>/_probe/probe_summary.json
```

remote launcher 相关:

```text
/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/616/logs/*.log
/mnt/i/Github/Latent_Style/SchrodingerBridge/SchrodingerBridge/_codex_rt/*.sh
/mnt/i/Github/Latent_Style/SchrodingerBridge/SchrodingerBridge/_codex_rt/*.pid
```

注意:

- 如果旧配置把 `checkpoint.save_dir` 写成了相对路径, 结果可能错误落到
  `SchrodingerBridge/mnt/i/...`
- 这种嵌套目录先按有效证据读取, 不要因为路径丑就立刻重跑

## 6. Metrics and stop policy

主指标:

- `transfer_clip_style`
- `transfer_content_lpips`

训练内部探针:

- `ot_target_gini`
- `gpu_vram_used_gb_peak`
- `gpu_power_w_peak`
- `ot_topogate_diag_mean`
- `ot_topogate_entropy_mean`
- `ot_topogate_cost_mean`

目标:

- `style >= 0.74`
- `lpips <= 0.30`

资源目标:

- VRAM `9-10.8 GB`
- 功率尽量接近 `135W+`

安全阈值:

- `VRAM > 11.3 GB` 视为 OOM
- `LPIPS > 0.45` 可判危险
- `ot_target_gini > 0.6` 可判危险

## 7. OT implementation focus

当前 616 OT 关键点:

- `bridge_path_mode="vertical"` 仍是主基线
- `topogate_attention_gw` 替代旧 tokenizer-entropy 路线
- OT 结构代价应来自模型内生特征, 不再依赖 style-map tokenizer

主要代码位置:

```text
src/losses.py
src/run.py
src/trainer.py
src/utils/training.py
```

## 8. Practical rules

- 如果阶段 1 已在跑, 不要把自动树插进同一个目录抢写
- 阶段 2/3 优先复用自动树, 因为它统一处理:
  - batch probe
  - 1min 健康检查
  - 10min ETA
  - sleep-to-ETA
  - full-eval convergence stop
- 先看 full eval 结果, 不要只盯 probe
- 所有阶段都从头训练, 除非明确是在恢复同一条已经开始的 run
