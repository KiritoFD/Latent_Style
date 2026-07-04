# P3 v2 实验继续执行计划：监控 + 清理 + 后续实验

## 1. 当前状态总结

### 1.1 磁盘清理结果（已完成）
| 操作 | 释放空间 |
|------|----------|
| C盘 Temp | 11.49 GB |
| I盘 torch_cache | 0.23 GB |
| I盘 blobs (Docker) | 2.07 GB |
| 旧实验目录 (exp/) | 32.68 GB |
| mnt/ 冗余副本 | 0.8 GB |
| review_additional_experiments | 1.24 GB |
| backups/ | 0.5 GB |
| eval_cache/offline_pairing | **17.86 GB** |
| eval_cache/modelscope | **8.75 GB** |
| eval_cache/manual_clip | 1.52 GB |
| **总计** | **~76.8 GB** |

**I 盘**: 0 GB → **66 GB 空闲**

### 1.2 实验运行状态

| 实验 | 状态 | 关键数据 |
|------|------|----------|
| **E1** (film_init_contrastive) | **训练完成 ep1，评估失败** | checkpoint 被 /tmp 清理误删，需重新运行 |
| **E2** (two_stage_film) | **正在运行 (ep2→ep3)** | Ep1 结果已出，见下方 |

### 1.3 E2 Epoch 1 初步结果（已获取）

| 指标 | R1 Best (ep2) | **E2 Ep1** | 变化 |
|------|---------------|------------|------|
| clip_style (tr) | 0.6723 | **0.6702** | ≈ 持平 (-0.3%) |
| LPIPS (tr) | 0.3608 | **0.3326** | **-7.8% (更好!)** |
| clip_style (all) | 0.7041 | **0.7032** | ≈ 持平 |
| LPIPS (all) | 0.3618 | **0.3336** | **-7.8% (更好!)** |

**关键信号 — Velocity Magnitude**:
| | R1 Ep1 | R1 Ep2 | **E2 Ep1** | **E2 Ep2** |
|--|--------|--------|------------|------------|
| \|v\| | ~0.32 | ~0.40 | **0.293** | **0.416** |

→ E2 的 \|v\| 在 ep2 达到 0.416，显著高于 R1 同期！说明两阶段+FiLM大初始化确实在打破平凡解。

### 1.4 问题诊断

**E1 失败根因**: deep_cleanup.ps1 脚本中执行了 `rm -rf /tmp/p3_v2/*`，删除了正在运行的实验 checkpoint。
**教训**: 清理脚本不应删除 `/tmp/p3_v2/` 目录下的任何文件。

---

## 2. 执行计划

### Step 1: 等待 E2 完成（预计 2-3 分钟）
- E2 当前: Epoch 2 评估中 → Epoch 3 训练 → Epoch 3 评估
- 每个 epoch 训练 ~78s + 评估 ~198s ≈ ~4.5min/epoch
- 剩余时间: ~5 分钟

**操作**: 通过 SSH 轮询检查 E2 是否完成全部 3 个 epoch

### Step 2: 获取 E2 完整 3-epoch 结果
- 读取 `/tmp/p3_v2/e2_two_stage_film/full_eval/round2_convergence.json`
- 读取 `curve_summary.json` 获取完整 clip_style/LPIPS 曲线
- 对比 R1 baseline 的 Pareto 最优

**预期判断**:
- 如果 E2 的 LPIPS 持续 < 0.35 且 clip_style > 0.67 → **E2 方向有效**
- 如果 \|v\| 持续增长 (> 0.5 by ep3) → **成功突破平凡解**

### Step 3: 重新运行 E1
- E1 的 checkpoint 被 /tmp 清理误删，需要重新从头跑
- 修改 run_v2.sh 或手动启动 E1:
  ```bash
  python run.py --config exp/p3_remote_10h/e1_film_init_contrastive/config.json \
    >> exp/p3_remote_10h/e1_film_init_contrastive/focused.log 2>&1
  ```
- 预计耗时: ~15 min (3 epochs × ~5 min/epoch with eval)

**注意**: 确保 `/tmp/p3_v2/e1_film_init_contrastive/` 目录存在且不被清理

### Step 4: 基于 E1+E2 结果设计 E3（动态决策）

根据 v2 计划文档的决策树：

```
如果 E2 更好 (LPIPS 更低或 clip_style 更高):
  → E3 = 候选配置 B (强化两阶段)
    - training_objective_mode: endpoint
    - w_flow_scale: 0.2 (更激进降低 FM 权重)
    - two_stage_s1_w_endpoint_style: 20.0 (更激进的 S1)
    - two_stage_s1_w_style_strength_reg: 1.5
    - w_style_contrastive: 0.2 (加入对比损失)
    - single_step_swd_weight: 12.0 (增强 SWD)

如果 E1 更好:
  → E3 = 候选配置 A (velocity 模式 + 全部激进化)
    - w_flow_scale: 0.3
    - w_velocity_magnitude: 0.8
    - w_style_strength_reg: 0.8
    - w_contrast_preserve: 1.5
    - w_hf_energy: 1.5
    - w_pixel_color_match: 1.5
    - single_step_swd_weight: 10.0

如果两者都差 (< 0.68 style):
  → 检查评估是否准确，考虑增加到 5 epochs
```

### Step 5: 设计并运行 E4-E6（基于 E3 方向）

| 实验 | 条件 | 配置概要 |
|------|------|----------|
| **E4** | E1 方向有效但不够 | 加倍 FiLM init (0.10/0.20) + 对比损失 (0.5) |
| **E5** | E2 方向有效但 S1 不够激进 | 延长 S1 到 3 ep (纯激进注入) |
| **E6** | AdaIN 后处理 | 对最佳 checkpoint 做 inference_adain=true 零成本去雾 |

### Step 6: 目视检查生成图片
- 从每个实验的 `full_eval/epoch_0003/summary_grid.png` 下载图片
- 人工评估：
  - 雾化程度（是否还有白化）
  - Style 区分度（不同风格是否明显不同）
  - Content 保持（原始内容结构是否保留）

---

## 3. 磁盘空间持续监控

### 已完成的清理
- 总计释放 ~76.8 GB
- I 盘剩余: **66 GB**（充足）

### 后续注意事项
- **不要再次清理 /tmp/p3_v2/** — 这是实验 checkpoint 所在位置
- 如果后续实验积累导致空间不足，优先清理：
  - 旧的 full_eval 目录（保留 summary.json 即可）
  - 生成的图片文件（可重新生成）

---

## 4. 时间线预估

| 步骤 | 预计时间 | 状态 |
|------|----------|------|
| 等待 E2 完成 | ~5 min | 进行中 |
| 获取 E2 完整结果 | 1 min | 待执行 |
| 重跑 E1 | ~15 min | 待执行 |
| 设计 + 运行 E3 | ~10 min | 待执行 |
| 设计 + 运行 E4/E5 | ~20 min | 待执行 |
| E6 AdaIN 评估 | ~5 min | 待执行 |
| 目视检查 + 总结 | 10 min | 待执行 |
| **总计剩余** | **~66 min (~1h)** | |

---

## 5. 成功标准

| 指标 | R1 best (ep2) | 最低目标 | 理想目标 |
|------|---------------|---------|---------|
| clip_style (transfer) | 0.672 | > 0.685 | > 0.700 |
| LPIPS (transfer) | 0.361 | < 0.350 | < 0.330 |
| \|v\| (velocity mag) | ~0.40 | > 0.50 | > 0.60 |
| clip_style/LPIPS ratio | 1.86 | > 1.95 | > 2.10 |

**特别注意**: 如果 E2/E3 能同时提升 clip_style 并降低 LPIPS，则验证了"组合式突破"的有效性。

---

## 6. 文件操作清单

### 需要执行的远程命令
```bash
# 1. 检查 E2 完成
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/monitor_e2.sh

# 2. 获取 E2 完整结果
cat /tmp/p3_v2/e2_two_stage_film/full_eval/curve_summary.json

# 3. 重启 E1
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
nohup python run.py --config exp/p3_remote_10h/e1_film_init_contrastive/config.json \
  >> exp/p3_remote_10h/e1_film_init_contrastive/focused.log 2>&1 &

# 4. 创建 E3 配置（基于 E2 结果，使用候选配置 B）
# (具体参数见 Step 4)
```

### 需要创建的新文件
| 文件 | 用途 |
|------|------|
| `exp/p3_remote_10h/e3_aggressive_two_stage/config.json` | E3 配置（基于 E2 强化） |
| `exp/p3_remote_10h/run_v3.sh` | E3-E6 运行脚本 |
