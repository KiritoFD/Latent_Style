# FC-SB Phase 3 — Implementation Tasks

## 阶段 0: 理论预研与基线锁定（已完成）
- [x] 阅读 FC.md 理论，确认 FC-SB 核心命题
- [x] 5-style 完整验证 H5/N8/O4/P3/P7，确认 CLIP-LPIPS trade-off
- [x] Q 系列 WCT 验证（C=4 退化，证明二阶统计无效）

## 阶段 1: 代码实现（5 方向，已完成）
- [x] Task 1: 方向 R — Fiber-CFG（已作为 K1 存在，model620.py L839-849）
- [x] Task 2: 方向 T — Multi-band AdaIN（model620.py L713-767）
- [x] Task 3: 方向 U — Style Extrap（model620.py L663-668）
- [x] Task 4: 方向 V — Patch AdaIN（model620.py L768-807）
- [x] Task 5: 方向 W — 风格排斥 Loss（losses620.py L635-671, model620.py L282-296 style_disc_head）

## 阶段 1.5: 初始化策略探索（前置，必先于阶段 2-5）

### Task 5.5: 代码改动 — 让 style_film_init_std 可配置
**文件**（已完成）:
- `src/config_schema.py` — 新增 `style_film_init_std: float = 0.02` 字段
- `src/blocks620.py` — `SpatialBridgeBlock620.__init__` 添加 `film_init_std: float = 0.02` 参数；0.0 分支走 zero-init，>0.0 分支走 normal_(std)
- `src/model620.py` — `SpatialBridge620.__init__` 从 `model_cfg.style_film_init_std` 读取并传入 block

**验证**:
- [x] 3 个文件语法检查通过
- [ ] 远程 smoke test: 用 `style_film_init_std=0.0` 启动训练，确认无报错
- [ ] 远程 smoke test: 用 `style_film_init_std=0.1` 启动训练，确认无报错

### Task 5.6: 生成 L9 正交 9 组初始化探索配置
**文件**: `exp/625_fc_sb/gen_init_explore_configs.py`（重写）
**协议**: L9 正交表，3 维 × 3 水平 = 9 组
- `style_film_init_std`: [0.0, 0.02, 0.1]
- `style_embed_scale`: [1.0, 2.0, 4.0]
- `endpoint_delta_scale`: [0.5, 1.0, 2.0]

固定: `gate_mode=fixed_one`、`lr=5e-5`、`gate_warmup=0`、`batch=24`、`num_epochs=2`

**L9 正交表**:
| 组 | film_init_std | embed_scale | delta_scale |
|----|---------------|-------------|-------------|
| I1 | 0.0 | 1.0 | 0.5 |
| I2 | 0.0 | 2.0 | 1.0 |
| I3 | 0.0 | 4.0 | 2.0 |
| I4 | 0.02 | 1.0 | 1.0 |
| I5 | 0.02 | 2.0 | 2.0 |
| I6 | 0.02 | 4.0 | 0.5 |
| I7 | 0.1 | 1.0 | 2.0 |
| I8 | 0.1 | 2.0 | 0.5 |
| I9 | 0.1 | 4.0 | 1.0 |

**输出**: `exp/625_fc_sb/from_scratch/init_configs/I{1-9}.json`

**验证**:
- [ ] 9 个 .json 配置全部生成
- [ ] 每个配置的 `style_film_init_std` 字段正确
- [ ] 每个配置的 `style_embed_scale` 和 `endpoint_delta_scale` 字段正确

### Task 5.7: 执行初始化探索（9 组 × 2 epoch + probe + 5-style 评估）
**文件**: `exp/625_fc_sb/run_init_explore.sh`（重写）
**协议**:
- 每组: 2 epoch 从头训练 → probe 指标提取 → 5-style 评估
- probe 指标: style_gate_value, cross_attn_delta_abs, film_gamma_abs, film_beta_abs, cos_sim(v1,v2)
- 5-style 评估: t_clip, t_lpips, a_clip, a_lpips

**输出**: `exp/625_fc_sb/from_scratch/init_explore_summary.csv`
- 列: candidate, film_init_std, embed_scale, delta_scale, t_clip, t_lpips, a_clip, a_lpips, style_gate_value, cross_attn_delta_abs, film_gamma_abs, film_beta_abs, cos_sim_v1v2, status

**总耗时**: 9 × ~15min = ~2.5h

**验证**:
- [ ] 9 组训练完成
- [ ] 9 组 probe 指标提取完成
- [ ] 9 组 5-style 评估完成
- [ ] summary.csv 包含所有 14 列

### Task 5.8: 分析初始化探索结果，确定最佳策略
**协议**:
- 主指标: t_clip + t_lpips（综合排序）
- 辅助指标: probe 解释为什么好/坏
- 选择标准:
  - 优先选 t_clip 最高且 t_lpips < 0.50 的组合
  - 若多组接近，选 probe 指标更"健康"的（style_gate_value 高、cos_sim_v1v2 低）
- 输出: 最佳 (film_init_std, embed_scale, delta_scale) 三元组

**验证**:
- [ ] 最佳初始化策略确定
- [ ] 选择有数据支撑（9 组对比 + probe 解释）

## 阶段 2: 参数搜索配置生成（每方向 4-6 候选）

### Task 6: 生成各方向参数搜索配置
**文件**: `exp/625_fc_sb/gen_param_search.py`（新建统一脚本）
**协议**: 每个方向的关键参数做 4-6 候选值，基于 H5 训练配置生成快速训练配置
**快速训练参数**:
```json
{
  "training.batch_size": 24,
  "training.num_epochs": 2,
  "training.learning_rate": 5e-5,
  "bridge.bridge_sigma": 0.08,
  "bridge.bridge_sigma_schedule": "brownian_bridge"
}
```
**SRC_CKPT**: `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/e4_long_10ep/checkpoints/epoch_0008.pt`

**各方向参数搜索候选值**:

**R 方向（Fiber-CFG, fiber_cfg_scale）** — 5 候选:
| 候选 | fiber_cfg_scale | 假设 |
|------|----------------|------|
| R_s1 | 0.3 | 温和 |
| R_s2 | 0.7 | 中等 |
| R_s3 | 1.0 | 标准 |
| R_s4 | 1.5 | 强 |
| R_s5 | 2.5 | 激进 |

**T 方向（Multi-band, mid×hh）** — 5 候选:
| 候选 | mid_adain_scale | hh_adain_scale | 假设 |
|------|----------------|----------------|------|
| T_s1 | 0.3 | 0.3 | 退化验证（应≈P3）|
| T_s2 | 0.5 | 0.3 | 粗纹理增强 |
| T_s3 | 0.3 | 0.5 | 细纹理增强 |
| T_s4 | 0.7 | 0.5 | 强粗+中细 |
| T_s5 | 0.5 | 0.7 | 中粗+强细 |

**U 方向（Style Extrap, style_extrap_alpha）** — 5 候选:
| 候选 | style_extrap_alpha | 假设 |
|------|-------------------|------|
| U_s1 | 0.2 | 温和 |
| U_s2 | 0.5 | 中等 |
| U_s3 | 0.8 | 强 |
| U_s4 | 1.2 | 激进 |
| U_s5 | 1.8 | 极端 |

**V 方向（Patch AdaIN, patch_adain_kernel）** — 4 候选:
| 候选 | patch_adain_kernel | 假设 |
|------|-------------------|------|
| V_s1 | 4 | 小 patch，精细局部 |
| V_s2 | 8 | 中等 |
| V_s3 | 16 | 大 patch，粗区域 |
| V_s4 | 32 | 全局（退化验证）|

**W 方向（三机制分别搜索）** — 7 候选:
| 候选 | 配置 | 假设 |
|------|------|------|
| W1_s1 | w_fiber_repulsion=0.05, m=0.5 | 温和排斥 |
| W1_s2 | w_fiber_repulsion=0.2, m=0.5 | 中等排斥 |
| W1_s3 | w_fiber_repulsion=0.5, m=0.5 | 强排斥 |
| W2_s1 | w_anti_input_style=0.05, m=0.3 | 温和输入排斥 |
| W2_s2 | w_anti_input_style=0.2, m=0.3 | 中等输入排斥 |
| W3_s1 | w_style_disc=0.1, dim=128 | 轻量判别 |
| W3_s2 | w_style_disc=0.3, dim=128 | 中等判别 |

**总候选数**: R(5) + T(5) + U(5) + V(4) + W(7) = 26 个快速训练配置

**验证**:
- [ ] 26 个参数搜索配置全部生成
- [ ] 所有配置包含 H5_BASE + 快速训练参数 + 方向参数
- [ ] T_s1 (mid=hh=0.3) 与 P3 退化验证

## 阶段 3: 参数搜索训练（26 个快速训练）

### Task 7: 执行参数搜索训练
**协议**: 每个候选配置训练 2 epoch（~20-30min/候选），5-style 评估（~6min/候选）
**总耗时**: 26 × ~30min = ~13 小时（可并行或分批）
**显存控制**: batch=24, 峰值 9-10GB
**输出**: 每个候选的 2-epoch checkpoint + 5-style 评估结果

**验证**:
- [ ] 26 个候选训练完成（2 epoch each）
- [ ] 26 个候选 5-style 评估完成
- [ ] 每个方向的参数-性能曲线绘制（找最佳点）

### Task 8: 分析参数搜索结果，确定各方向最佳参数
**协议**: 对每个方向，根据 5-style t_clip 和 t_lpips 综合表现选 1-2 个最佳参数
**选择标准**:
- 优先选 t_clip 最高且 t_lpips < 0.50 的参数（vs H5 0.7026/0.4936）
- 若无候选优于 H5，选该方向内相对最优的
**输出**: 每个方向 1-2 个最佳参数，共 5-10 个进入阶段 B

**验证**:
- [ ] 5 个方向各自的最佳参数确定
- [ ] 最佳参数选择有数据支撑（参数-性能曲线）

## 阶段 4: 最佳点训练到收敛

### Task 9: 各方向最佳参数训练到收敛
**协议**: 每个方向的最佳参数（1-2 个）训练到收敛，max_epochs=10, patience=2, min_epochs=5
**总训练数**: 5-10 个完整训练
**显存控制**: batch=24, 峰值 9-10GB，超 11GB 降 batch 到 16
**训练日志**: `exp/625_fc_sb/<direction>_best/train.log`
**输出**: 每个最佳参数的 best checkpoint（按 val_loss）+ last checkpoint

**验证**:
- [ ] 5-10 个最佳参数训练完成（至少 5 epoch）
- [ ] 无 OOM / NaN
- [ ] 每个保存 best + last checkpoint

## 阶段 5: 最终 5-style 评估与方法结论

### Task 10: 最佳点训练后 5-style 完整评估
**协议**: 5-10 个训练后 best checkpoint 在 5-style 测试集完整评估
**基线对比**:
- H5: t_clip=0.7026, t_lpips=0.4936
- P3: t_clip=0.6638, t_lpips=0.2658
**判定标准**:
- 方向有效: t_clip > 0.705 或 t_lpips < 0.49（vs H5）
- 突破: t_clip > 0.72 且 t_lpips < 0.45
- 重大突破: t_clip > 0.72 且 t_lpips < 0.30

**验证**:
- [ ] 5-10 个 best checkpoint 5-style 评估完成
- [ ] 所有结果客观记录

### Task 11: 更新 EXPERIMENT_LOG.md
**客观记录**:
- 每个方向的参数搜索曲线（参数 vs 5-style t_clip/t_lpips）
- 每个方向的最佳参数及选择依据
- 每个方向最佳点训练后 5-style 真实结果
- 哪些方向有效/无效及理论解释
- 不夸大任何结果

## 阶段 6: 组合探索（如有多个方向有效）

### Task 12: 有效方向组合
若 ≥2 个方向在 5-style 上有效，探索组合：
- 推理侧组合: R + T, R + V, T + V
- 训练+推理组合: W + R, W + T
**协议**: 组合用各自最佳参数，训练到收敛后 5-style 评估

## Task Dependencies

```
阶段 1 (Task 1-5) ──── 代码实现，已完成
    ↓
阶段 1.5 (Task 5.5-5.8) ── 初始化探索（前置必做）
    ↓
阶段 2 (Task 6) ──── 参数搜索配置生成（用最佳初始化策略）
    ↓
阶段 3 (Task 7-8) ── 26 个快速训练 + 分析找最佳点
    ↓
阶段 4 (Task 9) ──── 5-10 个最佳点训练到收敛
    ↓
阶段 5 (Task 10-11) ── 最终评估 + 文档
    ↓ (如有突破)
阶段 6 (Task 12) ──── 组合探索
```

## 显存预算分配（RTX 3060 12GB，目标 9-11G）

| 阶段 | 类型 | 显存 | 策略 |
|------|------|------|------|
| 阶段 1.5 初始化探索训练 | 训练 | ~9-10G | batch=24, 2 epoch |
| 阶段 1.5 probe 评估 | 推理 | ~6-8G | batch=2, 单 forward |
| 阶段 1.5 5-style 评估 | 推理 | ~9-11G | batch=1, num_steps=12 |
| 阶段 3 快速训练 | 训练 | ~9-10G | batch=24, 2 epoch |
| 阶段 4 收敛训练 | 训练 | ~9-10G | batch=24, 5-10 epoch |
| 阶段 5 评估 | 推理 | ~9-11G | batch=1, num_steps=12 |

## 时间预算（粗估）

| 阶段 | 数量 | 单位耗时 | 总耗时 |
|------|------|---------|--------|
| 阶段 1.5 初始化探索 | 9 | ~15min | ~2.5h |
| 阶段 3 参数搜索训练 | 26 | ~30min | ~13h |
| 阶段 4 最佳点训练 | 5-10 | ~1.5h | ~8-15h |
| 阶段 5 最终评估 | 5-10 | ~6min | ~1h |
| **总计** | | | **~25-32h** |
