# Phase 2: 实验架构与设计文档

**Created**: 2026-06-28
**Status**: Draft
**Spec**: spec/comprehensive-ablation/spec.md

---

## 一、实验基线定义

### 基线 Checkpoint
- **T5 ep7**: `exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt`
- **all_pairs_clip_style**: 0.7307
- **all_pairs_content_lpips**: 0.3403
- **transfer_clip_style**: 0.7016
- **transfer_content_lpips**: 0.3515

### T5 ep7 训练配置关键参数（作为消融起点）
```
endpoint_adain_scale: 1.0          ← T5显式设置
style_extrap_alpha: 0.1            ← T5显式设置 (U方向)
lowpass_mode: dwt_haar             ← T5显式设置 (D2)
bridge_path_mode: vertical         ← T5显式设置
bridge_sigma: 0.02                 ← T5显式设置
style_cross_attn_gate_init: 0.05   ← T5显式设置
style_film_init_std: 0.02          ← T5显式设置

# 以下均为NOT_SET → 走代码默认值
multiband_adain_mode: single       ← 默认
patch_adain_kernel: 0              ← 默认（禁用）
mid_adain_scale: 0.3               ← 默认
hh_adain_scale: 0.3                ← 默认
fiber_cfg_scale: 0.0               ← 默认（禁用）
fiber_velocity_scale: 1.0          ← 默认（无放大）
fiber_source_repulse_scale: 0.0    ← 默认（禁用）
tri_band_inference_lock: False     ← 默认
endpoint_film_use_rmsnorm: False   ← 默认（GroupNorm白化）
gate_warmup_steps: 0               ← 默认（无warmup）
w_contrast_preserve: 0.0           ← 默认（禁用）
w_channel_variance: 0.0            ← 默认（禁用）
w_hf_energy: 0.0                   ← 默认（禁用）
w_velocity_magnitude: 0.0          ← 默认（禁用）
style_embed_scale: 1.0             ← 默认（无放大）
endpoint_delta_scale: 1.0          ← 默认（无放大）
body_norm_type: group_norm         ← 默认（白化）
endpoint_film_enabled: False       ← 默认（无FiLM head）
spectral_w_ll: 0.0                 ← 默认（spectral FM禁用）
```

---

## 二、完整消融机制清单（18个）

### Tier 1: 推理侧单因素（10个，无需重新训练）

| # | 机制 | 参数 | T5值 | 测试值 | 已有数据 |
|---|------|------|------|--------|---------|
| I1 | N1 Endpoint AdaIN开关 | endpoint_adain_scale | 1.0 | 0.0, 0.3, 0.5, 0.7 | ✅ Phase4有0.0(D0), 1.0(D4) |
| I2 | Style外推 | style_extrap_alpha | 0.1 | 0.0, 0.05, 0.15, 0.2 | ✅ Phase4有0.05-0.3 |
| I3 | Patch AdaIN核 | patch_adain_kernel | 0 | 8, 16, 32 | ✅ Phase4有8/16/32/48 |
| I4 | 多频带AdaIN | multiband_adain_mode | single | two_level + mid/hh组合 | ✅ Phase3 T1-T4 |
| I5 | Fiber CFG | fiber_cfg_scale | 0.0 | 1.0, 2.0, 3.0 | ❌ 无数据 |
| I6 | Fiber速度放大 | fiber_velocity_scale | 1.0 | 0.5, 1.5, 2.0 | ❌ 无数据 |
| I7 | Fiber源排斥 | fiber_source_repulse_scale | 0.0 | 0.5, 1.0 | ❌ 无数据 |
| I8 | 三频带锁 | tri_band_inference_lock | False | True + alpha=0.3/0.5/0.7 | ❌ 无数据 |
| I9 | Fiber-only endpoint | fiber_only_endpoint | False | True | ❌ 无数据 |
| I10 | Lowpass模式 | lowpass_mode | dwt_haar | avg_pool | ✅ Phase4 D1 |

### Tier 2: 训练侧单因素（8个，需训练3 epoch）

| # | 机制 | 参数 | T5值 | 测试值 | 已有数据 |
|---|------|------|------|--------|---------|
| T1 | Gate warmup | gate_warmup_steps | 0 | 500, 1000 | ⚠️ 3-axis fix有训练数据但基线不同 |
| T2 | RMSNorm(head) | endpoint_film_use_rmsnorm | False | True | ⚠️ 同上 |
| T3 | 反白化-对比度 | w_contrast_preserve | 0.0 | 2.0 | ⚠️ 同上 |
| T4 | 反白化-通道方差 | w_channel_variance | 0.0 | 0.5 | ❌ 无数据 |
| T5 | 反白化-高频能量 | w_hf_energy | 0.0 | 1.0 | ❌ 无数据 |
| T6 | 速度幅度正则 | w_velocity_magnitude | 0.0 | 1.0 | ❌ 无数据 |
| T7 | Gate init | style_cross_attn_gate_init | 0.05 | 0.1, 0.3 | ✅ Phase4 N11=0.3 |
| T8 | Spectral FM | spectral_w_ll + w_hh | 0/0 | 0.5/2.0 | ✅ Phase4 D3 |

---

## 三、数据复用策略

### 可直接复用的Phase4数据（25+组）
从`exp/p4_fusion_breakout/t5_b2v2_d2_d4/full_eval_p4_*`提取：

| 实验名 | 对应消融 | clip | lpips | 复用为 |
|--------|---------|------|-------|--------|
| T5_D0_baseline | I1(endpoint_adain=0.0) + I2(alpha=0.0) | 0.7307 | 0.3403 | I1/I2联合基线 |
| T5_D1_dwt | I10(lowpass=avg_pool→dwt_haar) | — | — | I10比较 |
| T5_D4_u01_v3 | I1(1.0)+I2(0.1)+I3(16) | 0.7323 | 0.3534 | I1+I2+I3组合 |
| P3_V_k08 | I3=8 | 0.7348 | 0.3868 | I3=8 |
| P3_V_k32 | I3=32 | 0.7307 | 0.3403 | I3=32 |
| P3_V_k48 | I3=48(裁剪=32) | 0.7307 | 0.3403 | I3=48无效 |
| P3_U_a015 | I2=0.15 | 0.7323 | 0.3564 | I2=0.15 |
| P3_U_a020 | I2=0.20 | 0.7318 | 0.3613 | I2=0.20 |
| P3_U_a030 | I2=0.30 | 0.7299 | 0.3751 | I2=0.30 |
| P3_V08_U005 | I2=0.05+I3=8 | 0.7342 | 0.3853 | I2+I3组合 |
| T5 baseline ep7 | 基线(endpoint_adain=1.0,alpha=0.1,lowpass=dwt_haar) | 0.7307 | 0.3403 | 基线 |

### 需要新跑的推理消融（5个机制，约10组）
- I5 fiber_cfg_scale: 3组 (1.0, 2.0, 3.0)
- I6 fiber_velocity_scale: 3组 (0.5, 1.5, 2.0)
- I7 fiber_source_repulse_scale: 2组 (0.5, 1.0)
- I8 tri_band_inference_lock: 2组 (True+alpha=0.3, True+alpha=0.7)
- I9 fiber_only_endpoint: 1组 (True)

### 需要新跑的训练消融（8个机制，8组，各3 epoch）
- T1-T8 各1组，3 epoch训练 + eval

---

## 四、实验执行架构

### 4.1 远程执行流程

```
[本地] 生成config JSON → SCP上传 → [远程WSL] 训练/eval → SCP下载summary.json → [本地] 汇总
```

### 4.2 推理侧消融脚本
- 基于`_p4_infer_ablation.py`模式：加载T5 ep7 checkpoint，修改推理config，跑eval
- 每组~2min eval时间
- 批量执行脚本：`628_infer_ablation_batch.py`

### 4.3 训练侧消融脚本
- 从T5 config.json派生，修改目标参数 + save_dir + ablation block
- 每组3 epoch训练(~45min) + eval(~2min) = ~50min
- 批量执行：先串行跑8组训练(~7h)，再批量eval
- 或：训练完成后批量eval

### 4.4 数据提取与汇总
- 从每组实验的`summary.json`提取all_pairs_clip_style, content_lpips
- 汇总到`ablation_results.csv`
- Python脚本自动计算Δvs基线

---

## 五、2因素组合实验设计

### 5.1 选取原则
从单因素结果中选取：
- **推理侧top-3**: 按|Δclip|最大的3个有效推理参数
- **训练侧top-3**: 按|Δclip|最大的3个有效训练参数
- **跨侧top-2**: 1个推理+1个训练的最佳组合

### 5.2 组合数量估算
- 推理3×3=6组（含交叉）
- 训练3×3=6组
- 跨侧2×2=4组
- 总计约16组

### 5.3 协同/拮抗判断标准
```
synergy_ratio = actual_Δ / (expected_Δ_A + expected_Δ_B)
- ratio > 1.2 → 协同 (1+1 > 2)
- ratio 0.8~1.2 → 独立 (1+1 ≈ 2)
- ratio < 0.8 → 拮抗 (1+1 < 2)
```

---

## 六、最终文档结构（docs/628/ablation_conclusions.md）

```
1. 实验概览
   1.1 基线定义
   1.2 消融机制清单（18个）
   1.3 实验方法说明

2. 推理侧单因素消融结果
   2.1 结果总表（参数×值×clip×lpips×Δ）
   2.2 各机制详细分析
   2.3 推理侧机制排名

3. 训练侧单因素消融结果
   3.1 结果总表
   3.2 各机制详细分析
   3.3 训练侧机制排名

4. 2因素组合实验
   4.1 推理×推理组合
   4.2 训练×训练组合
   4.3 推理×训练跨侧组合
   4.4 协同/拮抗分析表

5. 综合结论
   5.1 确定性结论清单
   5.2 推荐保留机制
   5.3 推荐移除/禁用机制
   5.4 最优配置推荐
   5.5 已知天花板与未解问题
```

---

## 七、时间预算

| 阶段 | 实验量 | 单次耗时 | 总耗时 |
|------|--------|---------|--------|
| Phase 0: 数据复用提取 | 25+组 | — | 0.5h |
| Phase 1: 推理侧新消融 | 10组 | ~2min | 0.5h |
| Phase 2: 训练侧消融 | 8组 | ~50min | ~7h |
| Phase 3: 2因素组合 | 16组 | ~2min(推理)~50min(训练) | ~4h |
| Phase 4: 文档撰写 | 1份 | — | 1h |
| **总计** | | | **~13h** |

考虑到远程GPU串行限制，训练侧消融为瓶颈。可通过以下方式优化：
- 训练侧先用1 epoch smoke test筛选，对有效的再跑3 epoch确认
- 1 epoch smoke: 8组×~20min = ~2.7h → 筛到3-4个有效 → 3-4组×3ep×~50min = ~3h → 总~6h训练

---

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| 训练崩溃/NaN | 记录为"不稳定"，跳过；检查grad_clip_norm=1.0 |
| Fiber CFG需2x计算 | eval时间×2，但仍可接受（~4min/组） |
| Source repulsion需额外forward | 同上 |
| 训练1-3 epoch不够判断 | 对关键机制扩展到7 epoch |
| 3-axis fix与T5基线不一致 | T5不含3-axis修复，正好可独立测试每个3-axis组件 |
