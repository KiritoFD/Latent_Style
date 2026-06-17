# 工具脚本目录

## SSH / 远程控制

```bash
# SSH 到远程 GPU 机器
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62

# WSL 内执行命令
ssh ... "wsl -d Ubuntu-26.04 <command>"

# 拷贝文件到远程
scp -P 2222 local.json administrator@100.115.18.62:C:/tmp/
# 然后从 WSL 内 cp /mnt/c/tmp/local.json /mnt/i/Github/Latent_Style/.../configs/
```

**重要**: WSL 内用 `/mnt/i/` 访问 I: 盘。管道 `|` 在 SSH 命令中被 PowerShell 截获，需避免或使用 `Select-String` 替代。

## 状态监控

| 脚本 | 用途 |
|------|------|
| `tools/experiments/report_remote_experiment_status.py` | 全量远程实验状态报告 |
| `tools/experiments/report_phase2_queue_state.py` | Phase2 队列状态 |
| `tools/experiments/build_phase2_status_note.py` | 生成 phase2 状态文档 |
| `tools/experiments/refresh_phase2_guide_watch.py` | 监控 phase2 进展 |
| `tools/experiments/refresh_phase2_safe_successors.py` | 自动排队继任实验 |

## 实验启动

| 脚本 | 用途 |
|------|------|
| `tools/experiments/launch_remote_experiment_train.py` | 通用远程训练启动 |
| `tools/experiments/launch_remote_phase2_eval_only_override.py` | 仅 eval（不改训练） |
| `tools/experiments/run_phase2_eval_only_override.py` | 本地 eval override |
| `tools/experiments/watch_launch_round1_queue_when_idle.py` | GPU 空闲时自动启动队列 |

## 616 专用脚本

| 脚本 | 用途 |
|------|------|
| `tools/experiments/run_phase616_clean_unbalanced_dummy_vertical_affine.sh` | 综合实验：Unbalanced OT + Vertical FM + AffineTokenizer |
| `tools/experiments/run_phase616_ot_vertical_round1.sh` | OT + Vertical 组合 |
| `tools/experiments/run_phase616_clean_ot_probe_round*.sh` | OT 结构代价模式对比 (round3-8) |
| `tools/experiments/run_phase616_clean_vertical_target_probe_round*.sh` | 垂直 FM 扫描 |
| `tools/experiments/run_phase616_clean_stats_bridge_combo_round1_authoritative.sh` | 统计量 + 桥组合 |
| `tools/experiments/launch_phase616_clean_ot_rebuild_stage1.sh` | OT 重建阶段 1 |
| `tools/experiments/build_phase616_ot_probe_table.py` | OT probe 结果汇总表 |
| `tools/experiments/build_phase616_projection_probe_table.py` | 投影 probe 结果汇总 |
| `tools/experiments/build_phase616_style_stats_bank.py` | 风格统计量银行 |

## Eval 工具

| 脚本 | 用途 |
|------|------|
| `tools/experiments/rerun_full_eval_for_run.py` | 对已有 ckpt 重新跑 full eval |
| `tools/experiments/build_clip_lpips_curve_from_eval_root.py` | 从 eval 目录提取 CLIP-S/LPIPS 曲线 |
| `tools/experiments/compare_distinct5_eval_curve.py` | 对比多个实验的 eval 曲线 |

## CSV/数据分析

| 脚本 | 用途 |
|------|------|
| `docs/612-lookback/compile_all.py` | 合并所有实验 CSV |
| `tools/experiments/build_inmortal_epoch_eval_table.py` | 生成 immortal 实验 epoch 表 |
| `tools/experiments/csv_utils.py` | CSV 读写通用工具 |

## GPU/VRAM 管理

- **上限**: 3060 12GB, 约束 `< 11.3 GB`
- 当前 phase2 warmstart batch=12 (b12a1) 使用约 7-10 GB
- I2SB endpoint 模式下显存需求较低 (b8a2 约 4-6 GB)

## 代码修改的核心位置

| 文件 | 关键改动 |
|------|----------|
| `src/losses.py` | `_bridge_path_state` — `vertical` 模式 (L1201-1220) |
| `src/losses.py` | `_bridge_state_and_velocity` — 垂直目标速度 (L1152-1168) |
| `src/losses.py` | `_structure_pairwise_cost` — 7 种 OT 代价模式 (L745) |
| `src/losses.py` | `_sinkhorn_plan` — Unbalanced Sinkhorn (L421) |
| `src/losses.py` | `_coupling_cost_matrix` — 结构+外观混合代价 (L953) |
| `src/losses.py` | `__init__` — bridge_path_mode validation (L160-166) |
