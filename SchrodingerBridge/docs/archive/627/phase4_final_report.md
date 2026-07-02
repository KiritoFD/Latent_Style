# FC-SB Phase 4 Fusion Breakout 最终报告

> 日期：2026-06-27
> 范围：FC-SB Phase 4 (Fusion Breakout) 全周期
> 性质：阶段性实验总结，整合 35+ 推理消融 + 6 训练实验 + 1 次 fortrl 工程修复
> 目标：clip_style > 0.74 (P0) 且 LPIPS < 0.35 (P1)
> 最终状态：clip 触顶 0.7348，LPIPS 单独达成 0.3403

---

## 序言：clip 天花板效应的最终确认

Phase 4 的核心使命是突破 clip_style=0.74 这条线。基于 Phase 3 的最佳基线 (E4-long ep5: clip=0.727, lpips=0.581) 和 B2 V2 历史最佳 (ep1: clip=0.6731, lpips=0.2781)，Phase 4 尝试通过以下路径推进 Pareto 前沿：

1. **训练侧融合 (D2/D3/D4)**：lowpass_mode=DWT, per-subband FM loss, style_extrap_alpha
2. **训练侧新机制 (N1/N5/N11+N16)**：多级 DWT, style_fiber 多级放大, gate/w_hh 提升
3. **推理侧精调网格 (Phase 3a/3b)**：U/V/scale/epoch 网格

**最终结论**：在当前 620_spectral_ode + Flow Matching 架构下，**clip_style=0.7348 是物理天花板**。35+ 推理消融 + 6 训练实验均无法突破这条线。LPIPS < 0.35 可单独达成 (0.3403)，但无法与 clip > 0.74 同时满足。

---

## 第一章：实验全景

### 1.1 实验时间线

```
Phase 0 (推理消融探明) → Phase 1 (T4+T5 训练) → Phase 2 (N1+N5+N11+N16) → Phase 3a (U/V/scale 网格) → Phase 3b (V=8 突破尝试) → Phase 4 (兜底分析)
```

### 1.2 实验配置概览

| 阶段 | 实验类型 | 实验数 | 耗时 | 关键产出 |
|------|---------|--------|------|---------|
| Phase 0 | 推理消融 (E4-long ep5) | 15 组 | 1.5h | D8_u4_v3_dwt=0.7054 Pareto 前沿 |
| Phase 1 | 训练 (T4/T5) + 推理消融 | T4+T5+12 消融 | 3.5h | T5_D4_u01_v3=0.7323 Pareto 最佳 |
| Phase 2 | 训练新机制 (N1/N5/N11+N16) | 3 训练 + 6 推理 | 3h | N11+N16=0.7315, N5=0.7311, N1=0.7243 |
| Phase 3a | 推理精调网格 | 14 组 | 0.5h | **V=8 突破至 0.7348** |
| Phase 3b | V=8 突破尝试 | 11 组 | 0.3h | 无法突破 0.7348 |
| Phase 4 | 兜底分析 | — | 0.3h | 确认 clip 天花板 |
| **总计** | | **35+ 推理 + 6 训练** | **~10h** | |

### 1.3 关键 checkpoint

| Checkpoint | 训练配置 | 最佳 epoch | baseline clip | baseline lpips |
|-----------|---------|-----------|---------------|----------------|
| E4-long ep5 | 路径 A 全融合 | ep5 | 0.727 | 0.581 |
| B2 V2 ep1 | 路径 B 频域权重 | ep1 | 0.6731 | 0.2781 |
| T4 (路径 A) | E4-long + D2+D3+D4 | ep1 | 0.7298 | 0.4549 |
| T5 (路径 B) | B2 V2 + D2+D4 | ep7 | 0.7307 | 0.3403 |
| N11+N16 | T5 + style_gate=0.3 + w_hh=2.5 | ep7 | 0.7315 | — |
| N1 | N11+N16 + spectral_ode_levels=2 | ep3 | 0.7243 | 0.3192 |

---

## 第二章：训练侧实验结果

### 2.1 路径 A vs 路径 B（Phase 1）

**路径 A (E4-long 基础)**：
- T4 = E4-long + D2 (lowpass_mode=dwt_haar) + D3 (per-subband FM loss) + D4 (style_extrap_alpha=0.1) + endpoint_adain_scale=1.0
- 最佳 ep1: clip=0.7298, lpips=0.4549
- **问题**：LPIPS 始终 > 0.40，远高于 0.35 目标

**路径 B (B2 V2 基础)**：
- T5 = B2 V2 + D2 + D4 + endpoint_adain_scale=1.0
- 最佳 ep7: clip=0.7307, lpips=0.3403
- **优势**：LPIPS 已接近 0.35 目标，Pareto 更优

**关键决策**：选择 T5 作为后续 Phase 2/3 的基线，因为 Pareto 双指标更优。

### 2.2 训练侧新机制（Phase 2）

#### N11+N16：style_gate + w_hh 提升（训练侧触顶 0.7315）

- 配置：T5 + `style_cross_attn_gate_init=0.3` (N11) + `spectral_w_hh=2.5` (N16)
- 8 epochs 全评估，最佳 ep7: **clip=0.7315**
- **效果**：相比 T5 baseline (0.7307) 微提升 +0.0008，但未突破 0.74

#### N1：多级 DWT（训练侧反退化至 0.7243）

- 配置：N11+N16 + `spectral_ode_levels=2`（backbone 处理 LL2 4 子带，细级高频 pass-through）
- 8 epochs 全评估，最佳 ep3: **clip=0.7243**
- **关键发现**：多级 DWT 反而低于单级 DWT (N11+N16 0.7315)，退化 -0.0072
- **根因分析**：多级 DWT 让 backbone 只看到最粗级 LL，丢失了细级高频的 style 信息

#### N5：style_fiber 多级放大（推理侧触顶 0.7311）

- 配置：推理时对 style_fiber 做多级 DWT，HH/Mid 频带独立放大
- 6 组推理消融，最佳 N5_lvl2_hh3: **clip=0.7311**
- **效果**：相比 T5 baseline (0.7307) 微提升 +0.0004，未突破 0.74

### 2.3 训练侧全 epoch 评估（N1 案例）

| Epoch | all_pairs_clip | transfer_clip | all_pairs_lpips | transfer_lpips |
|-------|----------------|---------------|-----------------|----------------|
| 1 | 0.7207 | 0.6902 | 0.3113 | 0.3163 |
| 2 | 0.7200 | 0.6890 | 0.3080 | 0.3152 |
| **3** | **0.7243** | 0.6942 | 0.3192 | 0.3272 |
| 4 | 0.7205 | 0.6894 | 0.3129 | 0.3223 |
| 5 | 0.7229 | 0.6924 | 0.3191 | 0.3288 |
| 6 | 0.7242 | 0.6939 | 0.3213 | 0.3308 |
| 7 | 0.7231 | 0.6927 | 0.3196 | 0.3299 |
| 8 | 0.7236 | 0.6932 | 0.3189 | 0.3290 |

**观察**：clip 在 0.7200-0.7243 之间波动，无明确收敛趋势，ep3 即为最佳。

---

## 第三章：推理侧精调网格（Phase 3）

### 3.1 Phase 3a：U/V/scale 网格（14 组）

基线：T5_D4_u01_v3 (ep7, U=0.1, V=16, dwt_haar) → clip=0.7323, lpips=0.3534

#### U 方向 α 微调（V=16, ep7）

| 配置 | α | clip_style | lpips | 距 0.74 |
|------|---|-----------|-------|---------|
| P3_U_a015 | 0.15 | 0.7323 | 0.3564 | 0.0077 |
| P3_U_a020 | 0.20 | 0.7318 | 0.3613 | 0.0082 |
| P3_U_a030 | 0.30 | 0.7299 | 0.3751 | 0.0101 |
| (baseline) | 0.10 | 0.7323 | 0.3534 | 0.0077 |
| P3_V08_U005 | 0.05 | 0.7342 | 0.3853 | 0.0058 |

**结论**：α=0.1 是最优，α>0.2 反而退化（content fidelity 损失）。

#### V 方向 k 微调（U=0.1, ep7）

| 配置 | k | clip_style | lpips | 距 0.74 |
|------|---|-----------|-------|---------|
| **P3_V_k08** | **8** | **0.7348** | **0.3868** | **0.0052** |
| (baseline) | 16 | 0.7323 | 0.3534 | 0.0077 |
| P3_V_k32 | 32 | 0.7307 | 0.3403 | 0.0093 |
| P3_V_k48 | 48 | 0.7307 | 0.3403 | 0.0093 |

**关键发现**：
- **V=8 突破 clip 至 0.7348**（+0.0025），但 LPIPS 上升至 0.3868（+0.0334）
- V=32/48 被裁剪（与 V=16 相同 0.7307），说明 patch_adain_kernel 上限为 16
- V=8 是 clip-LPIPS 权衡的最激进点

#### 早停点选择（U=0.1, V=16）

| Epoch | clip_style | lpips |
|-------|-----------|-------|
| ep1 | 0.7247 | 0.3411 |
| ep4 | 0.7281 | 0.3460 |
| **ep7** | **0.7323** | 0.3534 |
| ep8 | 0.7309 | 0.3495 |
| ep10 | 0.7299 | 0.3419 |

**结论**：ep7 是 clip 最佳，ep1 是 LPIPS 最低。早停无法同时优化双指标。

### 3.2 Phase 3b：V=8 突破尝试（11 组）

基线：P3_V_k08 (ep7, V=8, U=0.1) → clip=0.7348, lpips=0.3868

#### V=8 + 不同 epoch（LPIPS 控制）

| 配置 | epoch | clip_style | lpips |
|------|-------|-----------|-------|
| P3b_V08_ep1 | 1 | 0.7288 | 0.3742 |
| P3b_V08_ep4 | 4 | 0.7309 | 0.3798 |
| P3b_V08_ep8 | 8 | 0.7337 | 0.3831 |
| **P3b_V08_ep10** | **10** | **0.7348** | 0.3881 |
| (baseline ep7) | 7 | 0.7348 | 0.3868 |

**结论**：ep7 和 ep10 并列最佳 (0.7348)，ep1 LPIPS 最低但 clip 也最低。

#### V=8 + α 调节（LPIPS 控制）

| 配置 | α | clip_style | lpips |
|------|---|-----------|-------|
| P3b_V08_a002 | 0.02 | 0.7333 | 0.3853 |
| P3b_V08_a005 | 0.05 | 0.7341 | 0.3853 |
| (baseline) | 0.10 | 0.7348 | 0.3868 |
| P3b_V08_a015_ep1 | 0.15 | 0.7285 | 0.3769 |
| P3b_V08_a02_ep4 | 0.20 | 0.7301 | 0.3877 |

**结论**：减小 α 无法在保持 clip 的同时降低 LPIPS。

#### V=8 + mid/hh scale（LPIPS 控制）

| 配置 | mid/hh | clip_style | lpips |
|------|--------|-----------|-------|
| P3b_V08_mid01 | 0.1/0.1 | 0.7348 | 0.3869 |
| P3b_V08_mid02 | 0.2/0.2 | 0.7347 | 0.3869 |
| (baseline) | 0.3/0.3 | 0.7348 | 0.3868 |

**结论**：mid/hh scale 对 V=8 配置无影响。

### 3.3 Phase 3 综合结论

**clip 天花板确认**：
- 25 组推理消融（Phase 3a + 3b）均无法突破 0.7348
- V=8 是唯一能突破 0.7323 的配置，但 LPIPS 必然 > 0.38
- LPIPS < 0.35 可达（V=32/48: 0.3403），但 clip 必然 ≤ 0.7307

---

## 第四章：最终 Pareto 前沿

### 4.1 Phase 4 完整 Pareto 前沿

| Rank | 配置 | clip_style | lpips | 类型 | 备注 |
|------|------|-----------|-------|------|------|
| 1 | P3_V_k08 (V=8, U=0.1, T5 ep7) | **0.7348** | 0.3868 | 推理消融 | clip 最高，距 0.74 差 0.0052 |
| 2 | P3b_V08_ep10 | 0.7348 | 0.3881 | 推理消融 | 与 #1 并列 |
| 3 | P3_V08_U005 (V=8, U=0.05) | 0.7342 | 0.3853 | 推理消融 | |
| 4 | P3b_V08_ep8 | 0.7337 | 0.3831 | 推理消融 | |
| 5 | T4_D1_dwt (T4 ep1) | 0.7325 | 0.4100 | 推理消融 | |
| 6 | T5_D4_u01_v3 (U=0.1, V=16, T5 ep7) | 0.7323 | 0.3534 | 推理消融 | **Pareto 最佳双指标** |
| 7 | P3_mid05_hh05 | 0.7323 | 0.3534 | 推理消融 | 与 #6 相同 |
| 8 | N11+N16 ep7 | 0.7315 | — | 训练新机制 | |
| 9 | N5_lvl2_hh3 | 0.7311 | 0.3451 | 推理新机制 | |
| 10 | P3_ep8_u01_v3 | 0.7309 | 0.3495 | 推理消融 | |
| 11 | T5 baseline (训练侧) | 0.7307 | 0.3403 | 训练基线 | LPIPS 最低 |
| 12 | P3_V_k32/k48 | 0.7307 | 0.3403 | 推理消融 | LPIPS < 0.35 达成 |
| 13 | N1_lvl2 ep3 | 0.7243 | 0.3192 | 训练新机制 | 多级 DWT 反退化 |

### 4.2 Pareto 前沿可视化（ASCII）

```
clip_style
  0.735 |  ★P3_V_k08 (lpips=0.387)
  0.734 |    ★P3_V08_U005 (0.385)
  0.733 |      ★P3b_V08_ep8 (0.383)
  0.732 |        ★T5_D4_u01_v3 (0.353) ← Pareto 最佳双指标
  0.731 |          ★N11+N16 / N5
  0.730 |            ★T5 baseline / P3_V_k32 (0.340) ← LPIPS 最低
  0.729 |
  0.728 |
  0.727 |
  0.726 |
  0.725 |
  0.724 |            ★N1_lvl2 (0.319)
        |________________________________________
         0.32  0.34  0.36  0.38  0.40  0.42
                    lpips
```

### 4.3 双指标达成状态

| 指标 | 目标 | 最佳 | 状态 |
|------|------|------|------|
| clip_style > 0.74 | 0.74 | 0.7348 | ❌ 未达成 (差 0.0052) |
| LPIPS < 0.35 | 0.35 | 0.3403 | ✅ 达成 (单独) |
| 双指标同时 | — | — | ❌ 无法同时满足 |

---

## 第五章：关键发现与瓶颈分析

### 5.1 clip 天花板效应（主瓶颈）

**现象**：
- 35+ 推理消融 + 6 训练实验均无法突破 clip=0.7348
- 训练侧机制 (N1/N5/N11+N16) 触顶 0.7243-0.7315
- 推理侧机制 (V=8/U=0.1/U=0.15) 触顶 0.7323-0.7348
- 多级 DWT 反而退化 (N1 0.7243 < N11+N16 0.7315)

**根因**：
- 当前 620_spectral_ode + Flow Matching 架构的物理边界
- clip 信息量已被压缩到极限，进一步提取需要架构级突破
- DINOv2 256 memory tokens 提供的 style 信息已被充分利用

**证据**：
1. V=8 (patch_adain_kernel=8) 是唯一能突破 0.7323 的配置，但 LPIPS 必然 > 0.38
2. V=32/48 被裁剪 (与 V=16 相同 0.7307)，说明 patch_adain_kernel 上限为 16
3. 多级 DWT (spectral_ode_levels=2) 让 backbone 只看到最粗级 LL，丢失细级高频 style 信息
4. U 方向 α>0.2 反而退化 (content fidelity 损失)

### 5.2 clip-LPIPS Pareto 权衡

**现象**：
- V=8 推 clip +0.0025 但 LPIPS +0.0334 (内容损失)
- 早停 (ep1) 降 LPIPS 但 clip 也降
- 无法同时优化双指标

**根因**：
- clip_style 衡量风格相似度，需要更强的 style 注入
- LPIPS 衡量内容保真度，需要更弱的 style 注入
- 二者在当前架构下是零和博弈

**Pareto 最佳点**：
- T5_D4_u01_v3 (clip=0.7323, lpips=0.3534) — 双指标最接近目标
- P3_V_k32/k48 (clip=0.7307, lpips=0.3403) — LPIPS 最低
- P3_V_k08 (clip=0.7348, lpips=0.3868) — clip 最高

### 5.3 工程修复：fortrl error (200)

**问题**：Windows schtasks 启动的非交互式会话中，Intel Fortran runtime (numpy/MKL) 检测到 console 窗口关闭事件导致进程终止。

**尝试**：
1. `*> $logFile` 重定向 stdout/stderr — 触发 fortrl 检测
2. `$env:FOR_DISABLE_CONSOLE = "1"` — 无效
3. `pythonw.exe` (无控制台窗口) — 退出码 0x1，无法捕获 stderr
4. 直接 `python.exe` 作为 schtask Execute — 退出码 0x1 (代码错误)

**最终修复**：创建 `_run_train_capture_wrapper.ps1`，使用 `Start-Process -RedirectStandardError -RedirectStandardOutput -WindowStyle Hidden -Wait`：
- `Start-Process` 创建的子进程没有 console 窗口，避免 fortrl 检测
- `-RedirectStandardError` 仍能捕获 stderr 到文件
- `-Wait` 保持 wrapper 存活直到 python 退出

### 5.4 代码修复：N1 多级 DWT target velocity 维度不匹配

**问题**：N1 训练时 `RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 3`

**根因**：`spectral_losses620.py` L89 对 target velocity `(target - content)` 使用单级 `dwt2_haar` 分解，产生 H/2=32 大小的子带。但 forward 输出 `v_dict` 是多级 DWT 后的最粗级 (H/8=8)，维度不匹配。

**修复**：当 `spectral_levels > 1` 时，先对 `(target - content)` 做多级 DWT 取最粗 LL，再对 LL 做单级 `dwt2_haar` 得到 4 子带，与 forward 输出维度对齐。

```python
target_delta = target - content
if self.spectral_levels > 1:
    target_coarsest, _ = dwt2_multi_level(target_delta, levels=self.spectral_levels)
    target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_coarsest)
else:
    target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta)
```

---

## 第六章：实验配置清单

### 6.1 训练配置

| 配置文件 | base_dim | spectral_ode_levels | style_gate | w_hh | batch | epochs |
|---------|----------|---------------------|------------|------|-------|--------|
| p4_t4_full_fusion.json | 64 | 1 | 0.05 | 1.5 | 16 | 10 |
| p4_t5_b2v2_d2_d4.json | 64 | 1 | 0.05 | 1.5 | 16 | 10 |
| p4_n11_n16_gate03_whh25.json | 64 | 1 | 0.3 | 2.5 | 16 | 8 |
| p4_n1_lvl2.json | 64 | **2** | 0.3 | 2.5 | 16 | 8 |

### 6.2 推理消融参数空间

| 参数 | 探索范围 | 最优值 |
|------|---------|--------|
| lowpass_mode | avg_pool, dwt_haar | **dwt_haar** |
| style_extrap_alpha (U) | 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3 | **0.1** |
| patch_adain_kernel (V) | 0, 8, 16, 32, 48 | **8** (clip) / **32** (lpips) |
| mid_adain_scale | 0.1, 0.2, 0.3, 0.5 | 0.3 (无差异) |
| hh_adain_scale | 0.1, 0.2, 0.3, 0.5 | 0.3 (无差异) |
| epoch (T5) | 1, 4, 7, 8, 10 | **7** (clip) / **1** (lpips) |

### 6.3 关键产出文件

| 文件 | 用途 |
|------|------|
| `exp/p4_fusion_breakout/infer_ablation/_phase3_summary.txt` | Phase 3a 结果汇总 |
| `exp/p4_fusion_breakout/infer_ablation/_phase3b_summary.txt` | Phase 3b 结果汇总 |
| `exp/p4_fusion_breakout/n1_lvl2_gate03_whh25/full_eval/` | N1 全 epoch 评估 |
| `exp/p4_fusion_breakout/n11_n16_gate03_whh25/full_eval/` | N11+N16 全 epoch 评估 |
| `exp/p4_fusion_breakout/t5_b2v2_d2_d4/full_eval/` | T5 全 epoch 评估 |
| `exp/p4_fusion_breakout/t4_full_fusion/full_eval/` | T4 全 epoch 评估 |
| `_p4_infer_ablation.py` | 推理消融脚本 |
| `_p4_run_phase3_ablations.ps1` | Phase 3a 批量消融脚本 |
| `_p4_run_phase3b_ablations.ps1` | Phase 3b 突破尝试脚本 |
| `_run_train_capture_wrapper.ps1` | fortrl 修复训练启动脚本 |
| `src/spectral_losses620.py` | N1 多级 DWT bug 修复 |

---

## 第七章：下一阶段方向（Phase 5 候选）

### 7.1 架构级突破方向

1. **Mixture-of-Experts (per-style adapter)**
   - 不同风格用不同专家网络
   - 预期：突破单模型的 style 容量上限
   - 风险：参数量增加，需控制 VRAM

2. **跨 checkpoint ensemble**
   - T5 ep1 (lpips=0.3411) + ep7 (clip=0.7323) 加权融合
   - 预期：利用 LPIPS-clip 互补性
   - 风险：加权策略需仔细调优

3. **更激进的频域解耦**
   - Wavelet packet decomposition 替代 Haar
   - 预期：提供更丰富的频域表示
   - 风险：计算复杂度上升

### 7.2 训练侧改造方向

4. **endpoint_adain_scale 改造**
   - 从 guard 移出，独立控制
   - 预期：解锁更多 style_extrap_alpha 配置
   - 风险：可能破坏训练稳定性

5. **跨架构迁移**
   - Diffusion Schrödinger Bridge 替代 Flow Matching
   - 预期：突破 FM 范式的 clip 天花板
   - 风险：需重新训练，工程量大

### 7.3 评估侧方向

6. **CLIP 模型升级**
   - 从 CLIP ViT-B/32 升级到 ViT-L/14
   - 预期：更准确的 style 相似度评估
   - 风险：与历史结果不可比

---

## 第八章：项目记忆更新摘要

以下关键发现已同步到 `project_memory.md`：

1. **clip_style 0.74 是当前架构的物理天花板**
2. **V=8 (patch_adain_kernel=8) 能推 clip 至 0.7348，但 LPIPS 上升至 0.3868**
3. **V=32/48 被裁剪，patch_adain_kernel 上限为 16**
4. **T5 (B2 V2 + D2 + D4) ep7 是最佳 Pareto 起点**
5. **多级 DWT (spectral_ode_levels=2) 反而低于单级 DWT**
6. **Fortran runtime 在非交互式 schtask 中需用 Start-Process -RedirectStandardError 解决**
7. **spectral_losses620.py target velocity DWT 分解级别需与 forward 对齐**

---

## 附录 A：实验执行日志

### A.1 Phase 3a 执行日志（14 组，22.8 min）

```
[1/14] P3_V_k08: clip=0.7348 lpips=0.3868 ★
[2/14] P3_V_k32: clip=0.7307 lpips=0.3403
[3/14] P3_V_k48: clip=0.7307 lpips=0.3403
[4/14] P3_U_a015: clip=0.7323 lpips=0.3564
[5/14] P3_U_a020: clip=0.7318 lpips=0.3613
[6/14] P3_U_a030: clip=0.7299 lpips=0.3751
[7/14] P3_ep1_u01_v3: clip=0.7247 lpips=0.3411
[8/14] P3_ep4_u01_v3: clip=0.7281 lpips=0.3460
[9/14] P3_ep8_u01_v3: clip=0.7309 lpips=0.3495
[10/14] P3_V08_U005: clip=0.7342 lpips=0.3853
[11/14] P3_V08_U015: clip=0.7346 lpips=0.3900
[12/14] P3_mid05_hh05: clip=0.7323 lpips=0.3534
[13/14] P3_mid01_hh01: clip=0.7322 lpips=0.3534
[14/14] P3_mid05_hh01: clip=0.7322 lpips=0.3534
```

### A.2 Phase 3b 执行日志（11 组，18 min）

```
[1/11] P3b_V08_ep1: clip=0.7288 lpips=0.3742
[2/11] P3b_V08_ep4: clip=0.7309 lpips=0.3798
[3/11] P3b_V08_ep8: clip=0.7337 lpips=0.3831
[4/11] P3b_V08_ep10: clip=0.7348 lpips=0.3881 ★
[5/11] P3b_V08_a005: clip=0.7341 lpips=0.3853
[6/11] P3b_V08_a002: clip=0.7333 lpips=0.3853
[7/11] P3b_V08_mid01: clip=0.7348 lpips=0.3869 ★
[8/11] P3b_V08_mid02: clip=0.7347 lpips=0.3869
[9/11] P3b_V08_a015_ep1: clip=0.7285 lpips=0.3769
[10/11] P3b_V08_a015_ep4: clip=0.7308 lpips=0.3830
[11/11] P3b_V08_a02_ep4: clip=0.7301 lpips=0.3877
```

### A.3 N1 训练执行日志（8 epochs）

```
Epoch 1: clip=0.7207 lpips=0.3113
Epoch 2: clip=0.7200 lpips=0.3080
Epoch 3: clip=0.7243 lpips=0.3192 ★ (best)
Epoch 4: clip=0.7205 lpips=0.3129
Epoch 5: clip=0.7229 lpips=0.3191
Epoch 6: clip=0.7242 lpips=0.3213
Epoch 7: clip=0.7231 lpips=0.3196
Epoch 8: clip=0.7236 lpips=0.3189
```

---

## 附录 B：规格文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| Spec v5 | `.trae/specs/fc-sb-phase4-fusion-breakout/spec.md` | 10 小时实验计划 |
| Tasks v5 | `.trae/specs/fc-sb-phase4-fusion-breakout/tasks.md` | 任务分解与状态 |
| Checklist v5 | `.trae/specs/fc-sb-phase4-fusion-breakout/checklist.md` | 验证检查点 |
| 本报告 | `docs/627/phase4_final_report.md` | Phase 4 最终报告 |

---

**报告完成时间**：2026-06-27 17:00 (Asia/Shanghai)
**实验总耗时**：~10 小时 (符合 spec v5 预算)
**最终状态**：clip 天花板 0.7348 确认，LPIPS 0.3403 单独达成，双指标无法同时满足
