# Phase 4H.1: End-of-trajectory AdaIN (EOTA) — 设计文档

**Date**: 2026-07-01
**Stage**: Phase 4H.1 (加法 - 结构性 pivot, 解决 4G.2b 洞察)
**Goal**: 解耦 ODE 求解与风格注入, 恢复 α 参数有效性, 让 per_subband 路线复活以突破 clip SOTA。

---

## 1. 动机: 4G.2b 的"多步迭代累积"问题

### 1.1 4G.2b 的惊人发现

| 配置 | α | clip | lpips | 判定 |
|------|---|------|-------|------|
| 4G.2 | 1.0 | 0.7361 | 0.3843 | MIXED |
| 4G.2b | 0.5 | 0.7362 | 0.3845 | FAIL |
| Δ | -0.5 | +0.0001 | +0.0002 | — |

**α 减半, 结果几乎零变化!** per-step α 在多步 ODE 中失效。

### 1.2 根因: 多步 Euler 迭代累积

推理用 12 步 Euler (`num_steps=12`), 每步都应用 endpoint AdaIN:

```
每步: sub_new = (1-α)·sub + α·match(sub, s_sub)
n 步后残留: (1-α)^n · sub_original

α=1.0, n=12: (0)^12 = 0% 残留 → 100% 替换
α=0.5, n=12: (0.5)^12 = 0.024% 残留 → ~100% 替换
```

**per-step α 在多步 ODE 中不是有效的注入量控制参数。**

### 1.3 结构性 pivot 的必要性

4G.2b 的 Future Work 列出了 3 个方向:
1. End-of-trajectory AdaIN (只在最后一步应用) — **本实验**
2. α 衰减调度 (α_t = α_0 · (1-t/T))
3. 全局缩放而非 per-step 替换

方向 1 是最干净的结构性 pivot: **改变应用时机, 而非调参数**。

---

## 2. 数学公式

### 2.1 当前 (每步应用 AdaIN)

```
for i in 0..N-1:
    h = euler_step(h, v_pred)          # ODE 求解
    h = endpoint_adain(h, style)        # 每步都注入风格
```

残留 = (1-α)^N → α 失效

### 2.2 Phase 4H.1 (EOTA: 只在最后一步应用)

```
for i in 0..N-2:
    h = euler_step(h, v_pred)          # 前 N-1 步: 纯 ODE 求解
# 最后一步:
h = euler_step(h, v_pred)              # ODE 求解
h = endpoint_adain(h, style)            # 只在最后一步注入风格
```

残留 = (1-α)^1 = (1-α) → α 恢复有效性

### 2.3 理论美感

**解耦**:
- ODE 求解 (前 N-1 步): 纯频域 Euler, LL/LH/HL 独立积分, 无风格干扰
- 风格注入 (最后 1 步): endpoint AdaIN 一次性注入

**物理意义**:
- 前 N-1 步: 网络学习"如何从 content 流形走向 target 流形" (内容理解)
- 最后 1 步: 在到达的流形点上, 一次性注入风格统计

**类比**: 像射箭 — 前 N-1 步是"拉弓瞄准"(ODE 求解轨迹), 最后 1 步是"放箭"(风格注入)。

---

## 3. 实现

### 3.1 配置字段 (config_schema.py)

```python
endpoint_adain_only_last_step: bool = False  # Phase 4H.1: EOTA
```

### 3.2 integrate_transport 修改 (spectral_bridge620.py)

```python
only_last_step = bool(_cfg_get('endpoint_adain_only_last_step', False))

for i in range(steps):
    # ... Euler step ...
    
    # Endpoint AdaIN (EOTA: 只在最后一步)
    apply_adain_this_step = (not only_last_step) or (i == steps - 1)
    if apply_adain_this_step and endpoint_adain_scale > 0.0 ...:
        # 应用 AdaIN
```

### 3.3 实验配置

基于 4G.2 (per_subband α=1.0), 添加 `endpoint_adain_only_last_step: true`:

```json
{
  "_base": "630_phase4g2_per_subband.json",
  "model": {
    "endpoint_adain_only_last_step": true
  }
}
```

---

## 4. 实验设计

### 4.1 实验矩阵

| 编号 | adain_mode | only_last_step | α | 描述 |
|------|-----------|----------------|---|------|
| 4F.1 SOTA | spatial_fiber | false | 1.0 | 当前 SOTA (每步应用) |
| 4G.2 | per_subband | false | 1.0 | MIXED (每步, 9× 注入) |
| **4H.1a** | **per_subband** | **true** | **1.0** | **EOTA + per_subband α=1.0** |
| 4H.1b (条件) | per_subband | true | 0.5 | 若 4H.1a 仍 FAIL, 降 α |

### 4.2 预期

| 指标 | 4G.2 (每步) | 4H.1a (EOTA) 预期 | 原因 |
|------|------------|-------------------|------|
| clip_style | 0.7361 | 0.730-0.738 | 单步注入可能略降 (风格注入量减少) |
| content_lpips | 0.3843 | 0.335-0.345 | 单步注入 (1-α)^1 = 0% (α=1.0), 但只有 1 次注入而非 12 次 |

**关键预测**: EOTA 下 α=1.0 等效于 4G.2 的单步版本。由于 per_subband 在单步内仍做 9 个子带独立匹配, lpips 可能仍略高于 spatial_fiber, 但应大幅优于 4G.2 的 0.3843。

### 4.3 验收

- **PASS**: clip ≥ 0.7243, lpips ≤ 0.3453 → 新 SOTA 候选, 进入 4H.1b 调 α
- **MIXED**: clip > 0.7319 but lpips > 0.3453 → 4H.1b 降 α
- **FAIL**: clip < 0.7243 → EOTA 路线废弃, 记录为 ablation

---

## 5. 理论提升

### 5.1 三层频域解耦的完整实现 (EOTA 成功后)

```
Layer 1: LL velocity (训练 + Euler 应用, 前 N-1 步)
  - 4G.1 证明: LL 携带全局色调, 必须漂移
  - EOTA: 前 N-1 步纯 Euler, 无 AdaIN 干扰

Layer 2: Endpoint AdaIN (最后 1 步, per_subband)
  - 4G.2 证明: per_subband 频域解耦有效 (clip +0.0042)
  - EOTA: 单步应用, α 恢复有效性

Layer 3: Spectral ODE (LH/HL velocity heads, 全程)
  - 3 个独立 velocity heads
  - Euler 积分在频域独立进行
```

### 5.2 论文 Core Story (如果 EOTA 成功)

> "我们通过 Haar DWT 多级分解解耦内容 (LL) 与风格 (HF)。
> 消融矩阵: (1) 减法验证 Content Fidelity Pathway; (2) LL velocity 消融量化 +0.014 clip 贡献;
> (3) 频域 per-subband AdaIN 突破 SOTA 但多步 ODE 使 α 失效 (4G.2b 洞察);
> (4) End-of-trajectory AdaIN 解耦 ODE 求解与风格注入, 恢复 α 有效性, 突破 clip+lpips 双指标。
> 完整故事: 从'损失函数工程'到'频域拓扑空间的直接流形建模', 再到'ODE 求解与风格注入的时序解耦'。"
