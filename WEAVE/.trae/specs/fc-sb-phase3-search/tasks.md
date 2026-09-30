# FC-SB Phase 3 参数搜索与 hh 排查 — Tasks

## 阶段 1: T 方向 hh 传递链路排查（诊断优先）

### Task 1.1: 读取 model620.py N1 块 hh 代码路径 ✅
**hh 代码路径（无断点）**:
```
L629 hh_adain_scale = _cfg_get('hh_adain_scale', 0.3)
  ↓ L770-771 Haar 分解 content/style fiber
  ↓ L778-779 adain_match_band → mid_matched, hh_matched
  ↓ L781 hh_final = hh_adain_scale * hh_matched + (1-hh_adain_scale) * f_hh_band
  ↓ L785-787 ep_fiber_matched = haar_inv(0, mid_lh, mid_hl, hh_final, ...)
  ↓ L848 endpoint = ep_base + (1-α)*ep_fiber_curr + α*ep_fiber_matched
  ↓ L861-864 v_pred → v_fiber (fiber_proj_ep 再投影)
  ↓ L887 h = h + v_fiber * dt
  ↓ L918-932 BASE LOCKING: h = x_base_lock + (h - lp(h))
  ↓ L933 return h
```
hh 作用于 hh_final → ep_fiber_matched → endpoint → x，链路完整。

### Task 1.2: 加 hh 可观测性 + smoke test ✅
**新增 9 个 hh 可观测性 key**（L678-688 默认值, L793-806 two_level 分支内实际值）:
- n1_hh_input_abs / n1_hh_matched_abs / n1_hh_final_abs
- n1_mid_input_abs / n1_mid_matched_abs / n1_mid_final_abs
- n1_hh_contribution_ratio (hh_final / (mid_final+hh_final))
- n1_hh_adain_scale / n1_mid_adain_scale

**T1 vs T3 smoke test 结果**:
| 指标 | T1(hh=0.3) | T3(hh=0.5) | Δ |
|------|-----------|-----------|---|
| n1_hh_final_abs | 0.1468 | 0.2037 | +38.8% ✅ |
| n1_hh_contribution_ratio | 0.2967 | 0.3633 | +22.4% ✅ |
| clip_style | 0.6675 | 0.6747 | +0.0072 ✅ |
| content_lpips | 0.6673 | 0.6666 | -0.0007 |

**断点定位结论**: **无断点，是设计如此**。hh 实际生效在 clip_style 维度（+0.007），不在 lpips 维度（-0.001）。原因：
1. BASE LOCKING（L918-932）锁死 content lowpass，保 lpips
2. LPIPS 对高频不敏感（HH 是对角高频纹理）
3. N1 设计目标正是"base 锁死保 LPIPS, fiber 获得风格统计提 clip"
**hh 的作用是"提 clip 不损 lpips"**，与 mid/endpoint_adain_scale 职责正交。

## 阶段 2: U/V 参数细化搜索

### Task 2.1: 生成 U/V 新参数变体 checkpoint ✅
新增 7 个变体（从 I7 checkpoint 生成，改 config 字段）:
- U4_alpha01 (α=0.10), U5_alpha015 (α=0.15), U6_alpha025 (α=0.25), U7_alpha03 (α=0.30)
- V4_kernel20 (k=20), V5_kernel24 (k=24), V6_kernel32 (k=32)
全部 endpoint_adain_scale=1.0, endpoint_adain_mode=full，config 验证通过。

### Task 2.2: 评估 U/V 新变体 ✅
全部 n1_adain_executed=1.0（7/7 VALID），评估约 29 分钟。

**U 方向完整结果**:
| 变体 | α | clip_style | lpips | Δclip | Δlpips | 击败I7 |
|------|---|-----------|-------|-------|--------|--------|
| **U4** | 0.10 | 0.7225 | 0.3660 | +0.0208 | +0.0035 | **YES** |
| U5 | 0.15 | 0.7195 | 0.3683 | +0.0178 | +0.0058 | YES |
| U1 | 0.20 | 0.7164 | 0.3735 | +0.0147 | +0.0110 | YES |
| U6 | 0.25 | 0.7131 | 0.3807 | +0.0114 | +0.0182 | YES |
| U7 | 0.30 | 0.7094 | 0.3897 | +0.0077 | +0.0272 | no |
| U2 | 0.50 | 0.6959 | 0.4307 | -0.0058 | +0.0682 | no |
| U3 | 1.00 | 0.6736 | 0.5218 | -0.0281 | +0.1593 | no |
趋势：α 越小越好（外推越温和），U4(α=0.10) 综合最佳。

**V 方向完整结果**:
| 变体 | k | clip_style | lpips | Δclip | Δlpips | 击败I7 |
|------|---|-----------|-------|-------|--------|--------|
| V1 | 4 | 0.7242 | 0.5196 | +0.0225 | +0.1571 | no |
| V2 | 8 | 0.7290 | 0.4497 | +0.0273 | +0.0872 | no |
| V3 | 16 | 0.7295 | 0.3963 | +0.0278 | +0.0338 | no |
| V4 | 20 | 0.6334 | 0.5889 | -0.0683 | +0.2264 | no (崩塌) |
| V5 | 24 | 0.6562 | 0.5330 | -0.0455 | +0.1705 | no (崩塌) |
| **V6** | 32 | 0.7262 | 0.3722 | +0.0245 | +0.0097 | **YES** |
趋势：非单调，仅 2 幂次 kernel（4/8/16/32）工作正常，非 2 幂次（20/24）崩塌（patch 边界伪影）。

### Task 2.3: 分析 U/V 搜索结果 ✅
**找到 5 个击败 I7 的点**（clip>0.7017 且 lpips≤0.3825）:
1. **U4(α0.1)**: clip=0.7225(+2.97%), lpips=0.3660(+0.97%) — **综合最佳**
2. U5(α0.15): clip=0.7195(+2.54%), lpips=0.3683(+1.60%)
3. U1(α0.2): clip=0.7164(+2.10%), lpips=0.3735(+3.03%)
4. U6(α0.25): clip=0.7131(+1.63%), lpips=0.3807(+5.02%)
5. **V6(k32)**: clip=0.7262(+3.49%), lpips=0.3722(+2.67%) — **clip 增益最大**

**帕累托前沿**: I7 → U4(α0.1) → V6(k32) → V3(k16)

## 阶段 3: W 方向 margin 调参

### Task 3.1: 生成 W 调参配置 ✅
W2c(margin=5), W2d(margin=10), W2e(margin=15)，基于 W2b.json，w_anti_input_style=3.0 不变。

### Task 3.2: 训练 W 调参变体 ✅
3 个变体训练成功（每个约 6 分钟，VRAM 5.71GB）。
**[W2-debug] step=1 dist_input/loss**:
| 变体 | margin | step=1 loss | step=51+ loss |
|------|--------|------------|---------------|
| W2c | 5 | 0.4156 | 0.0 |
| W2d | 10 | 1.2793 | 0.0 |
| W2e | 15 | 2.4753 | 0.0 |
关键：step=51 起 loss 全部归 0（模型一步就把 dist_input 推过 margin），hinge loss 仅 step=1 生效。

### Task 3.3: 评估 W 调参变体 ✅
| 变体 | margin | clip_style | lpips | Δlpips vs I7 |
|------|--------|------------|-------|--------------|
| I7 | - | 0.7017 | 0.3625 | 0 |
| W2c | 5 | 0.7123 | 0.3580 | -0.0045 (平凡解) |
| W2d | 10 | 0.7060 | 0.4270 | +0.0645 |
| W2e | 15 | 0.6946 | 0.4652 | +0.1027 |
| W2b | 20 | 0.6947 | 0.4645 | +0.1020 |

**未找到有效折中点**。margin=5 是平凡解（loss 仅 step1 生效，等同无正则化）；margin≥10 导致 lpips 恶化 +0.06~0.10。
**根因**: hinge loss 梯度冲击过大，模型一步就把 dist_input 推过 margin，后续无梯度。
**改进方向**: (1) 降低 w_anti_input_style; (2) 改 soft hinge/KL; (3) 动态 margin 退火。

## Task Dependencies

```
阶段 1 (Task 1.1-1.2) ── T hh 排查（诊断）   ↘
阶段 2 (Task 2.1-2.3) ── U/V 参数搜索（评估）  → 完成
阶段 3 (Task 3.1-3.3) ── W 调参（训练+评估）   ↗
```

## 显存预算

| 阶段 | 类型 | 显存 | 策略 |
|------|------|------|------|
| 阶段 1.2 | smoke test 推理 | ~6-8G | 单样本 |
| 阶段 2.2 | U/V 评估 | ~4G | batch=16, num_steps=12 |
| 阶段 3.2 | W 训练 | ~5.7G | batch=24, 2 epoch |
| 阶段 3.3 | W 评估 | ~4G | batch=16, num_steps=12 |
