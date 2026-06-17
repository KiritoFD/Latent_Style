# 616 OT Scratch 长训练实验计划

> **状态**: 实现完成，待远程执行
> **创建日期**: 2026-06-17
> **关联文档**: [design.md](./design.md) · [unbalanced_ot.md](./unbalanced_ot.md) · [infra.md](./infra.md) · [launch.md](./launch.md)

## 1. 背景与目标

### 1.1 前序工作

OT 快速探针（Round 1-8）已完成，结论：OT 效果有限，被降级为"诊断/基础设施保留"。但快速探针受限于 1-epoch / fast-step 训练，无法反映长训练收敛后的 OT 贡献。

### 1.2 本实验目标

按 unified-plan 从头训练（scratch，无 warmstart），完成 24-epoch 长训练，对两种结构代价模式做 matched A/B 对比：

| 模式 | 结构代价来源 | 假设 |
|------|-------------|------|
| `self_affinity_gw` | 像素空间自亲和矩阵 | 结构信息直接，但对高频噪声敏感 |
| `tokenizer_entropy_affinity_gw` | structured_style_tokenizer 熵图 | 结构信息更语义化，需额外前向但抗噪性更好 |

### 1.3 关键修复

前序探针中 unbalanced Sinkhorn 实际未激活（tau=1.0 等效 balanced，dummy_cost=0 跳过 dummy 列）。本实验显式设置 `tau=0.5, dummy_cost=1.0` 以激活真正的质量截断和不匹配样本丢弃能力。

## 2. 配置矩阵

### 2.1 共享参数（两种模式相同）

| 参数 | 值 | 说明 |
|------|-----|------|
| `_base` | `phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json` | 继承链根 |
| `batch_size` | 8 | 3060 12GB 显存约束 |
| `accumulation_steps` | 2 | effective batch = 16 |
| `num_epochs` | 24 | 长训练 |
| `save_interval` | 1 | 每 epoch 存 checkpoint |
| `resume_checkpoint` | `""` | scratch 从头训练 |
| `ot_cost_mode` | `l2` | appearance cost |
| `coupling_solver` | `sinkhorn_unbalanced` | OT 求解器 |
| `coupling_cost_composition` | `structure_only` | 仅用结构代价 |
| `coupling_target_mode` | `barycentric_full` | barycentric 投影 |
| `sinkhorn_unbalanced_tau_src` | 0.5 | rho≈0.5，源端质量截断 |
| `sinkhorn_unbalanced_tau_tgt` | 0.5 | 目标端质量截断 |
| `sinkhorn_unbalanced_dummy_cost` | 1.0 | 激活 dummy 列 |
| `sinkhorn_unbalanced_dummy_offdiag_cost` | 8.0 | dummy 非对角惩罚 |
| `training_target_projection_mode` | `pure_vertical_flow` | 垂直流目标投影 |
| `full_eval_each_epoch` | true | 每 epoch transfer_only 快速评估 |
| `full_eval_in_process` | true | 同进程评估（offload 机制保护） |

### 2.2 差异参数

| 参数 | self_affinity_gw | tokenizer_entropy_affinity_gw |
|------|:---:|:---:|
| `coupling_structure_cost_mode` | `self_affinity_gw` | `tokenizer_entropy_affinity_gw` |
| `checkpoint.save_dir` | `./exp/aaai2027_phase616_ot_vertical_scratch_b8a2_e24` | `./exp/aaai2027_phase616_ot_tokentropy_scratch_b8a2_e24` |
| `ablation.name` | `phase616_ot_vertical_scratch_b8a2_e24` | `phase616_ot_tokentropy_scratch_b8a2_e24` |

## 3. 代码修复清单

### 3.1 OT 实现（src/losses.py）

| 问题 | 严重度 | 修复 | 状态 |
|------|:---:|------|:---:|
| A. Unbalanced 参数失效 | 严重 | 配置显式设 tau=0.5, dummy_cost=1.0 | ✅ |
| B. vertical lowpass 代码重复 | 中等 | 抽取 `_vertical_lowpass` 方法 | ✅ |
| C. `_project_training_target` 死代码 | 轻微 | 删除 `projected_low = anchor_low` | ✅ |
| D. 冗余 `getattr` | 轻微 | `__init__` 中读取 `bridge_vertical_base_stride` | ✅ |

### 3.2 Infra 评估

| 问题 | 严重度 | 决策 | 理由 |
|------|:---:|------|------|
| E. `full_eval_in_process` 内存风险 | 中等 | 保留 | trainer offload 机制（CPU 移模型 + GC + empty_cache）已有效缓解；transfer_only 快速评估 max_src_samples=10 足够 |
| F. smoke 测试开销 | 轻微 | 跳过 | 配置已验证，启动脚本默认 `--skip-smoke` |

## 4. 远程环境参数

| 参数 | 值 |
|------|-----|
| SSH host | 100.115.18.62:2222 |
| SSH user | administrator |
| WSL distro | Ubuntu-26.04 |
| remote_workspace_root | /mnt/i/Github/Latent_Style |
| remote_python | /home/xy/venvs/samam312/bin/python |
| GPU | NVIDIA RTX 3060 12GB |

### 4.1 GPU 内存守卫参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max-prelaunch-memory-mib` | 7000 | 启动前 VRAM 上限 |
| `min-runtime-memory-mib` | 9216 | 训练最低需求 |
| `max-runtime-memory-mib` | 10800 | 训练硬上限 |
| `runtime-guard-max-memory-mib` | 11000 | 持续超此则 kill |
| `runtime-guard-min-mode` | warn | full_eval_each_epoch 自动降级 stop→warn |

## 5. 执行方式

```bash
# 本地执行（顺序启动两种模式）
bash tools/experiments/run_phase616_ot_scratch_matched.sh

# 环境变量覆盖（可选）
REMOTE_WSL_CWD=/mnt/i/Github/Latent_Style \
REMOTE_PYTHON=/home/xy/venvs/samam312/bin/python \
SKIP_SMOKE=1 \
CONTINUE_ON_FAILURE=0 \
bash tools/experiments/run_phase616_ot_scratch_matched.sh
```

脚本会顺序启动两个配置，每个通过 `launch_remote_experiment_train.py` 完成：
1. WSL 健康检查
2. 配置依赖同步到远程
3. 远程 `nohup` 启动训练 + GPU 守卫监控
4. 健康失败时自动 fallback 到 direct nohup

## 6. 预期指标

### 6.1 OT 诊断指标（验证 unbalanced 激活）

修复后以下指标应出现非零值（此前 tau=1.0 时接近零）：

- `ot_source_truncation`: 源端被截断的质量比例
- `ot_target_truncation`: 目标端被截断的质量比例
- `ot_dummy_mass`: 分配到 dummy 列的总质量

### 6.2 训练收敛指标

- `loss`: 24 epoch 内应稳定下降
- `lr`: 按 base 配置的 cosine schedule
- `gpu_memory_mib`: 峰值应 < 10800 MiB

### 6.3 评估指标（每 epoch transfer_only）

- `transfer_fid`: 跨风格迁移 FID
- `transfer_clip_style_acc`: CLIP 风格准确率
- `transfer_content_preservation`: 内容保留度

## 7. 结果路径

| 文件 | 路径 |
|------|------|
| self_affinity_gw checkpoints | `./exp/aaai2027_phase616_ot_vertical_scratch_b8a2_e24/` |
| tokenizer_entropy_gw checkpoints | `./exp/aaai2027_phase616_ot_tokentropy_scratch_b8a2_e24/` |
| 远程训练日志 | 远程 `exp/inmortal-exp/<run_name>_wrapper_nohup.log` |
| full_eval 输出 | `<checkpoint_dir>/full_eval_transfer_fast10/` |
| GPU 监控 | 远程 `exp/inmortal-exp/<run_name>_gpu_metrics.csv` |

## 8. 结论文档

实验完成后，结果对比和结论应写入：`docs/616/ot_scratch_results.md`（待创建）

应包含：
- 两种结构代价模式的收敛曲线对比
- OT 诊断指标随 epoch 的变化
- 最终 transfer 评估指标 A/B 对比
- unbalanced OT 是否带来实质性改善的结论
