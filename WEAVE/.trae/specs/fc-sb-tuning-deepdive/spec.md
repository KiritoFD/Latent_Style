# FC-SB 深度调优攻坚 — 突破训练-推理鸿沟

> **时间预算**: 1 天（~24 小时 GPU 时间）
> **核心目标**: 从 E2 (σ=0.04) 基线出发，系统性调优 FC-SB，实现 clip_style > 0.72 且 LPIPS < 0.45
> **基线对比**: E2 σ=0.04 (clip=0.708, LPIPS=0.540) vs Mystery-SDE (不训练, clip=0.711, LPIPS=0.337)

---

## 第一部分：问题诊断 — 为什么 FC-SB Phase 2 全军覆没？

### 1.1 实验数据铁证

| 实验 | clip_style↑ | LPIPS↓ | 训练epoch | 判定 |
|------|------------|--------|----------|------|
| **E2 σ=0.04 (基线)** | **0.708** 🏆 | **0.540** | 3 | **当前最优平衡** |
| E3 Velocity | 0.701 | 0.547 | 3 | velocity≈endpoint |
| E4 σ=0.02 | 0.703 | 0.554 | 3 | sigma太低更差 |
| Mystery-SDE (不训练) | 0.711 | **0.337** | 0 | **反直觉：不训练更强!** |
| E4 RMSNorm (内容最佳) | 0.672 | **0.373** | 3 | 风格偏弱 |
| E4-long ep5 (风格最高) | **0.727** 🏆 | 0.581 | 5 | 内容差 |
| --- | --- | --- | --- | --- |
| P2 Kernel7 | 0.612 | 0.695 | **1** ❌ | 训练不足 |
| P2 E=0 | 0.611 | 0.695 | **1** ❌ | 训练不足 |
| P2 Curriculum | 0.611 | 0.695 | **1** ❌ | 训练不足 |
| P2 FiberEP | 0.611 | 0.695 | **1** ❌ | 训练不足 |
| P2 Wavelet | 0.612 | 0.695 | **1** ❌ | 训练不足 |

### 1.2 三大失败根因假设

#### 假设 A：训练不充分（主要嫌疑）
- **证据**：Phase 2 全部只训练了1个epoch，auto-eval失败导致中断
- **验证**：重新训练3-5个epoch后再评估
- **预期**：训练充分后应至少追平 E2 基线

#### 假设 B：训练-推理目标不一致（深层矛盾）
- **现象**：Mystery-SDE 不训练反而更强（0.711/0.337 vs 0.708/0.540）
- **推理**：
  - 推理时：从 content 出发 → 加 SDE 噪声 → 风格提升 + LPIPS 低
  - 训练时：x_t 加噪 → 模型学去噪 → 把 SDE 噪声"修平"了 → 风格被去噪掉
- **核心矛盾**：训练目标是"从带噪 x_t 预测干净 target_velocity"，但推理时我们主动加噪声！
- **验证**：对比训练时有无 SDE 噪声的效果

#### 假设 C：Base Locking 过强限制表达
- **推理**：每步强制 `base = base(content)` 可能过于刚性
- **现象**：LPIPS 应该很低才对，但实际不低
- **验证**：soft base locking（加权混合而不是硬替换）

---

## 第二部分：调优战略 — 从 E2 基线出发的增量式探索

### 2.1 核心原则：增量消融，一次只变一个变量

不要像 Phase 2 那样一次把所有 FC-SB 特性全加上。从 E2 基线出发，**每次只改一个配置**，逐步逼近最优解。

### 2.2 四阶段调优路线图

```
阶段 0: 基础设施修复
  ↓ (确保能训练3个epoch + 正确评估)
阶段 1: 基础复现 + sigma 精细扫描
  ↓ (确认 E2 可复现，找最优 sigma)
阶段 2: 训练-推理对齐 + SDE 配方
  ↓ (解决核心矛盾：训练去噪 vs 推理加噪)
阶段 3: FC-SB 增量叠加
  ↓ (fiber projection, base locking 逐个加)
阶段 4: 课程学习 + 长训练
  ↓ (curriculum + 5-10ep)
阶段 5: CFG 外推 + 组合爆破
```

---

## 第三部分：阶段 0 — 基础设施修复

### 3.1 问题：auto-eval 导致训练中断

**根因**：`run.py` 每 epoch 后自动跑评估，CLIP 模型缓存问题导致失败，训练中断。

**解决方案**：
1. 新增 `full_eval_each_epoch: false` 配置（已有？需确认）
2. 或使用 `train_only.sh` 纯训练脚本，训练完统一评估

### 3.2 问题：评估后端不一致

- 本地 vs 远程可能用不同的 CLIP 后端
- 统一使用 OpenAI CLIP 后端做最终评估

---

## 第四部分：阶段 1 — 基础复现 + Sigma 精细扫描

### 4.1 目标
- 复现 E2 (σ=0.04) 结果
- 在 σ ∈ [0.03, 0.06] 区间精细扫描，找最优 sigma

### 4.2 实验矩阵（F1-F5）

| ID | 实验名 | sigma | 其他配置 | 预期 |
|----|--------|-------|---------|------|
| F1 | f1_repro_e2 | 0.04 | =E2 精确复现 | clip≈0.708, LPIPS≈0.540 |
| F2 | f2_sigma_030 | 0.03 | =E2 | LPIPS↓ style↓ |
| F3 | f3_sigma_035 | 0.035 | =E2 | ? |
| F4 | f4_sigma_045 | 0.045 | =E2 | ? |
| F5 | f5_sigma_050 | 0.05 | =E2 | style↑ LPIPS↑ |

**目标**：确定 style-LPIPS tradeoff 的最优 sigma 甜点

---

## 第五部分：阶段 2 — 训练-推理对齐（核心攻坚战）

### 5.1 核心矛盾：为什么"不训练的 SDE"更强？

```
训练过程（当前）:
  x_t + noise → model → predict clean velocity
  模型学会"去噪" → 把噪声修平 → 风格也被修掉了

推理过程（当前）:
  x_t → model → predict velocity → + noise → x_next
  我们主动加噪声 → 风格提升
  但模型预测的 velocity 是"去噪后的" → 两者方向可能矛盾！
```

### 5.2 三种解决方案

#### 方案 A：推理时不加 SDE 噪声（保守）
- 训练有 σ，推理无 σ
- 预期：LPIPS 变好，style 可能下降
- 实验：F6

#### 方案 B：训练也学"加噪后"的目标（激进）
- 训练 target = target_velocity + sigma * noise（和推理一致）
- 模型学习"带噪速度场"而不是"去噪"
- 预期：训练-推理一致，style 提升
- 实验：F7

#### 方案 C：训练用更高噪声，推理用较低噪声（退火）
- 训练 σ=0.08 → 推理 σ=0.04
- 模型过学习去噪 → 推理时少量噪声就能激发风格
- 实验：F8

### 5.3 实验矩阵（F6-F8）

| ID | 实验名 | 训练σ | 推理σ | 核心改动 |
|----|--------|------|------|---------|
| F6 | f6_infer_no_noise | 0.04 | **0.0** | 推理不加噪 |
| F7 | f7_train_additive_noise | 0.04 | 0.04 | 训练 target 加噪 |
| F8 | f8_train_high_sigma | **0.08** | 0.04 | 训练高σ推理低σ |

---

## 第六部分：阶段 3 — FC-SB 增量叠加

在阶段 1-2 找到的最优基础上，**逐个加入** FC-SB 特性：

### 6.1 增量式特性添加顺序

```
Baseline (最优 sigma + 最优 SDE 配方)
  ↓ + Fiber Velocity Projection (推理时剥离 v 的低频)
F9: i2sb_fiber_project_endpoint=true
  ↓ + Highpass Noise (噪声也只加高频)
F10: i2sb_fiber_project_noise=true
  ↓ + Base Locking (硬锁定 content 低频)
F11: bridge_path_mode=vertical
  ↓ + Fiber-Only Endpoint (模型只预测 Fiber)
F12: fiber_only_endpoint=true + training_target_projection=pure_vertical
  ↓ + Wavelet Lowpass (换切割方式)
F13: lowpass_mode=wavelet
```

### 6.2 每个特性的预期影响

| 特性 | 预期 style | 预期 LPIPS | 风险 |
|------|-----------|-----------|------|
| Fiber Vel Proj | ↓ 或 → | ↓↓ | 限制模型表达 |
| Highpass Noise | ↑↑ | ↓ | 理论上双赢 |
| Base Locking | → | ↓↓↓ | LPIPS 有数学保证 |
| Fiber-Only EP | ↑↑ | ↓ | 参数量聚焦 |
| Wavelet Lowpass | ? | ? | 需实验 |

**关键假设**：Highpass Noise + Base Locking 组合应该能实现 LPIPS 大幅下降（接近 Mystery-SDE 的 0.337）。

---

## 第七部分：阶段 4 — 课程学习 + 长训练

### 7.1 课程式 σ 调度

训练过程中逐步提升 σ，让模型先学结构再学风格：

```
Epoch 0-1: σ = 0.0  (学结构)
Epoch 1-3: σ = 0.03 (解耦)
Epoch 3-5: σ = 0.06 (全功率)
```

### 7.2 实验

| ID | 实验名 | 训练方式 | epoch | 说明 |
|----|--------|---------|-------|------|
| F14 | f14_curriculum_5ep | curriculum σ | 5 | 三阶段课程 |
| F15 | f15_constant_5ep | constant σ=0.04 | 5 | 对照：恒常 5ep |

---

## 第八部分：阶段 5 — CFG 外推 + 组合爆破

### 8.1 CFG 外推（无训练成本！）

用已训练好的最佳模型，做推理时 CFG 外推：
```
v = v_cond + 1.5*(v_cond - v_uncond)
```
扫描 `cfg_scale` ∈ [1.0, 3.0]

### 8.2 最佳组合验证

把前面找到的所有正向特性组合在一起，验证叠加效果。

---

## 第九部分：监控指标

### 必看指标

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `total_loss` | 总 Loss | 持续下降 |
| `velocity_std` | 速度场模长 | > 0.8 健康，> 1.2 优秀 |
| `training_target_projection_low_drift` | base 漂移 | < 0.01 (FC-SB模式) |
| `training_target_projection_high_energy_ratio` | fiber 能量比 | > 1.0 健康，> 1.3 优秀 |
| `training_sde_noise_hp_rms / lp_rms` | 高通/低通噪声比 | > 10:1 (纯高通) |

---

## 第十部分：时间预算分配（24小时）

| 阶段 | 任务 | 预计时间 | 累计 |
|------|------|---------|------|
| T+0h | 基础设施修复 + 配置生成 | 1h | 1h |
| T+1h | 阶段1: F1-F5 sigma扫描 (3ep each) | 5h | 6h |
| T+6h | 阶段2: F6-F8 SDE配方 (3ep each) | 3h | 9h |
| T+9h | 阶段3: F9-F13 增量叠加 (3ep each) | 5h | 14h |
| T+14h | 阶段4: F14-F15 长训练 | 6h | 20h |
| T+20h | 阶段5: CFG外推 + 组合验证 | 2h | 22h |
| Buffer | 排障/重跑 | 2h | **24h** |

---

## 第十一部分：功能需求 (FR)

### FR-1: 训练脚本支持无中间评估
系统 SHALL 提供纯训练脚本，训练 N 个 epoch 后退出，不在每个 epoch 后跑完整评估。

### FR-2: 训练 SDE 噪声模式可配置
系统 SHALL 支持两种训练 SDE 噪声模式：
- `subtractive` (默认): x_t 加噪，预测干净 target（去噪）
- `additive`: target_velocity 加噪，预测带噪 target

### FR-3: 课程式 sigma 训练调度
系统 SHALL 支持按 epoch 调度训练 sigma（区别于推理时的 t 调度）。

### FR-4: 统一评估脚本
系统 SHALL 提供统一评估脚本，对指定 checkpoint 用 OpenAI CLIP 后端跑完整评估。

---

## 第十二部分：验收标准

### AC-1: 基础设施可用
- **Given**: 任意实验配置
- **When**: 启动训练
- **Then**: 能完整跑完 3 个 epoch 不中断，生成 epoch_0003.pt
- **Verification**: programmatic

### AC-2: E2 基线可复现
- **Given**: F1 复现实验
- **When**: 训练 3 epoch 后评估
- **Then**: clip_style ∈ [0.69, 0.72], LPIPS ∈ [0.50, 0.58]
- **Verification**: programmatic

### AC-3: 至少一个实验超越 E2 帕累托
- **Given**: 所有实验结果
- **When**: 比较帕累托前沿
- **Then**: 至少存在一个实验满足 (clip > 0.708 AND LPIPS < 0.540) OR (clip > 0.72 AND LPIPS < 0.60)
- **Verification**: programmatic

### AC-4: Base Locking 显著降低 LPIPS
- **Given**: F11 (base locking) vs 其基线
- **When**: 对比评估结果
- **Then**: LPIPS 相对下降 > 10%
- **Verification**: programmatic

### AC-5: CFG 外推提升 style > 5%
- **Given**: 最佳模型 checkpoint
- **When**: cfg_scale=2.0 vs cfg_scale=1.0
- **Then**: clip_style 提升 > 5%
- **Verification**: programmatic

---

## 第十三部分：开放问题

- [ ] 训练 SDE 噪声到底是在帮倒忙还是在帮忙？（阶段2回答）
- [ ] Base Locking 应该多硬？硬替换 vs 软混合？
- [ ] Wavelet vs AvgPool 哪种低频切割方式更好？
- [ ] 5个epoch的自然收敛点在哪里？（E4-long说是epoch5）
