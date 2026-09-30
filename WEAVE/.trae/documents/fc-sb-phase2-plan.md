# FC-SB 全量实验 + 自主探索 — 一天实施计划 (Phase 2)

## Summary

基于 [FC.md](docs/622/FC.md) 的全部改造方案，**E1-E5 已完成**，最佳结果为 **E2 (σ=0.04): clip_style=0.708, LPIPS=0.540**。本计划覆盖 **FC.md 中所有未尝试的改造方向** + **自主探索**，目标是在一天内冲击帕累托前沿新高度。

## Current State Analysis

### 已完成的 FC-SB 实验（6 个配置）

| # | 实验名 | 核心变量 | clip_style | LPIPS | 判定 |
|---|--------|---------|-----------|-------|------|
| v1 | fc_sb_v1 | σ=0.08, endpoint, gate=0.5 | 0.704 | 0.559 | 基线 |
| E1 | fc_sb_huber | huber loss, σ=0.08 | 0.695 | 0.572 | 比MSE差 |
| **E2** | **fc_sb_sigma04** | **σ=0.04, endpoint** | **0.708** 🏆 | **0.540** | **当前最优** |
| E3 | fc_sb_velocity | velocity模式, σ=0.08 | 0.701 | 0.47 | ≈endpoint |
| E4 | fc_sb_sigma02 | σ=0.02, endpoint | 0.703 | 0.554 | sigma过低 |
| E5 | fc_sb_long | σ=0.04, 15ep | 0.696 | 0.573 | 过拟合 |

### 已完成的代码改造（4 文件）

| 改造 | 文件 | 状态 |
|------|------|------|
| Base Locking + Fiber SDE 推理 | model620.py L531-595 (`integrate_transport`) | ✅ |
| 训练期高通 SDE 噪声注入 | losses620.py L354-362 | ✅ |
| Huber Loss 支持 | losses620.py L363-370 | ✅ |
| RMSNorm + Gate init=0.5 | blocks620.py L11-28, L83 | ✅ |
| pure_vertical_flow_wavelet 投影 | losses620.py L190-243 | ✅ |
| CFG 推理外推 (已有代码) | model620.py L636-664 | ✅ 未测试 |

### FC.md 提出的但 **尚未尝试** 的改造（6 大方向）

| # | 方向 | 来源 | 当前状态 | 潜力评估 |
|---|------|------|---------|---------|
| A | **三阶段课程训练** (σ: 0→0.03→0.08) | FC.md §第四步 | ❌ 完全未尝试 | 🔥🔥🔥 最高 |
| B | **推理 CFG 外推** (scale>1.0) | model620.py L646-653 已有代码 | ❌ 从未测试过 | 🔥🔥 高 |
| C | **Fiber Endpoint 预测** (仅预测Δf) | FC.md 改造3 | ❌ 需新代码 | 🔥🔥 高 |
| D | **kernel / 低通变体** (k=7, wavelet) | FC.md + plan E6 | ❌ k=7未跑 | 🔥 中 |
| E | **w_style_energy_floor 扫描** (0.0~0.8) | FC.md YAML | ❌ 仅试了0.5 | 🔥 中 |
| F | **coupling_cost / OT 对齐** | FC.md coupling_cost_* | ❌ 字段存在但从未激活 | 🔥 中低 |

---

## Proposed Changes — 实验矩阵（Round 1-4）

### 总览：12 个实验 + 自主探索

| Round | # | 实验名 | 核心变量 | 目的 | 预计时间 | 依赖 |
|-------|---|--------|---------|------|---------|------|
| R1 | F1 | **fc_sb_cfg2** | cfg_target_scale=2.0 | 推理时CFG外推放大风格 | 10min(仅推理) | 无 |
| R1 | F2 | **fc_sb_cfg3** | cfg_target_scale=3.0 | 激进CFG外推 | 10min | 无 |
| R1 | F3 | **fc_sb_kernel7** | fiber_kernel=7 | 温和频率切割 | 50min | 无 |
| R1 | F4 | **fc_sb_floor0** | w_style_energy_floor=0.0 | 移除能量下限约束 | 50min | 无 |
| R2 | F5 | **fc_sb_curriculum** | 动态σ(t)=0.01→0.06 | 三阶段课程近似 | 50min+代码改 | 无 |
| R2 | F6 | **fc_sb_fiber_ep** | Fiber-only Endpoint预测 | 改造3实现 | 50min+代码改 | 无 |
| R2 | F7 | **fc_sb_wavelet** | wavelet低通替代avg_pool | 更干净的Base/Fiber分割 | 50min | 无 |
| R3 | F8 | **fc_sb_combo_a** | σ=0.04+kernel7+floor0 | F1-F4最佳组合 | 50min | R1结果 |
| R3 | F9 | **fc_sb_combo_b** | σ=0.04+curriculum+huber | 新机制组合 | 50min | R2结果 |
| R4 | F10-F12 | **自主探索** | 基于F1-F9动态决策 | 冲击帕累托 | 各50min | 前面结果 |

**总预计训练时间**: ~9小时 + ~2小时eval = **11小时**（留13小时buffer给排障/重跑/额外探索）

---

### Round 1: 快速参数扫描（基于 E2 最优基线 σ=0.04, endpoint）

#### F1: fc_sb_cfg2 — 推理 CFG 外推 (scale=2.0)

**原理**: 利用已有的 CFG 代码（model620.py L646-653），在推理时不改变模型权重，仅调整 `cfg_target_scale`：
```
guided = ep_target + scale * (ep_target - ep_null)
```
当 scale>1 时，风格信号被放大。这相当于"免费"的风格增强——不需要重新训练！

**操作**: 
- 使用 **E2 的 checkpoint**（已训练好的 fc_sb_sigma04 模型）
- 不需要重新训练！仅需修改推理参数 `cfg_target_scale=2.0`
- 运行 full_eval 提取指标

**预期**: clip_style 可能从 0.708 跳到 0.72-0.75，LPIPS 可能微升（因为放大了所有偏差）

---

#### F2: fc_sb_cfg3 — 激进 CFG 外推 (scale=3.0)

同 F1，但 scale=3.0 更激进。

**预期**: clip_style 进一步提升但 LPIPS 可能显著恶化。用于找到 CFG scale 的甜点。

---

#### F3: fc_sb_kernel7 — kernel=7 温和切割

**配置变更**（基于 E2）:
```json
{
  "model": { "i2sb_fiber_project_kernel": 7 }
}
```

**原理**: k=5 的 avg_pool 可能切割过于激进，把一些中频结构信息误判为 fiber。k=7 保留更多中频在 base 中，LPIPS 应更好。

**预期**: LPIPS 下降（可能到 0.48-0.52），clip_style 略降或持平

---

#### F4: fc_sb_floor0 — 移除能量下限

**配置变更**（基于 E2）:
```json
{
  "bridge": { "w_style_energy_floor": 0.0, "style_energy_floor_ratio": 0.0 }
}
```

**原理**: `w_style_energy_floor=0.5` 强迫模型保留高频方差，但这可能与 Base Locking 冲突（Base Locking 已经锁住了低频）。移除这个约束让模型自由学习。

**预期**: 不确定。可能 LPIPS 改善（更少人为干涉），也可能 clip_style 下降（失去高频激励）

---

### Round 2: 新机制实现（FC.md 核心改造）

#### F5: fc_sb_curriculum — 动态 Sigma 课程训练 ⭐最高优先级

**这是 FC.md 第四步"三阶段引爆课程"的核心思想**，但我们用简化版实现：

**代码改动** — `src/model620.py` `integrate_transport()` 方法:

在 Step 4 (SDE噪声注入处, L580-590)，将固定 `sigma_base` 替换为时间相关的动态 sigma：

```python
# 原来:
sigma_t = sigma_base * math.sqrt(max(0.0, t_curr*(1-t_curr))) * math.sqrt(abs(dt))

# 改为（课程式 sigma）:
# 通过配置项 bridge_sigma_schedule 控制
sigma_schedule = str(getattr(cfg, 'bridge_sigma_schedule', 'constant')).lower().strip()
if sigma_schedule == 'curriculum':
    # 三段式: t<0.33 用低σ锚定结构, 0.33<=t<0.66 用中σ解耦, t>=0.66 用高σ引爆
    if t_curr < 0.33:
        sigma_eff = sigma_base * 0.25   # 锚定期: 极低噪声
    elif t_curr < 0.66:
        sigma_eff = sigma_base * 0.6    # 解耦期: 中等噪声
    else:
        sigma_eff = sigma_base * 1.0    # 引爆期: 全功率
elif sigma_schedule == 'linear_ramp':
    sigma_eff = sigma_base * (0.2 + 0.8 * t_curr)  # 线性增长
else:
    sigma_eff = sigma_base  # constant (默认行为不变)
sigma_t = sigma_eff * math.sqrt(max(0.0, t_curr*(1-t_curr))) * math.sqrt(abs(dt))
```

**配置**:
```json
{
  "bridge": {
    "bridge_sigma": 0.06,
    "bridge_sigma_schedule": "curriculum"
  }
}
```

**同时需要在 config_schema.py 添加**: `bridge_sigma_schedule: str = "constant"`

**预期**: 课程训练应缓解冷启动梯度崩溃，让模型先学会结构保持再学风格注入。可能显著改善 LPIPS。

---

#### F6: fc_sb_fiber_ep — Fiber-Only Endpoint 预测 (FC.md 改造3)

**这是 FC.md 改造3的核心思想**：不让网络预测完整的 x_1，只预测 Fiber 差异 Δf。

**代码改动** — `src/model620.py` `integrate_transport()` 方法:

在 Step 1 之后（拿到 endpoint 后），添加 Fiber-only 投影：

```python
endpoint = self.predict_endpoint(...)
# 🆕 Fiber-Only Endpoint Projection
fiber_only_ep = bool(getattr(cfg, 'fiber_only_endpoint', False))
if fiber_only_ep:
    # 只保留 endpoint 的高频分量（fiber），base 强制锁定为 x_t 的 base
    ep_base = lp(endpoint)
    ep_fiber = endpoint - ep_base
    x_base_now = lp(h)  # 当前状态的 base
    endpoint = x_base_now + ep_fiber  # 合成: 当前base + 预测的fiber差异
```

**配置**:
```json
{
  "model": { "fiber_only_endpoint": true }
}
```

**config_schema.py 添加**: `fiber_only_endpoint: bool = False`

**原理**: 网络不再需要学习维护全局结构（Base 由解析几何保证），100% 参数量用来拟合极致笔触差异。

**预期**: clip_style 可能大幅提升（网络专注 fiber），LPIPS 应持平或改善（Base Locking 保证）

---

#### F7: fc_sb_wavelet — Wavelet 低通替代 AvgPool

**代码改动** — `src/model620.py` 和 `src/losses620.py`:

将 `lp()` 函数中的 `F.avg_pool2d` 替换为 wavelet 低通（类似 `_wavelet_lowpass` 已存在于 losses620.py L186-188）：

```python
# 在 integrate_transport() 中:
def lp(y, k=fiber_kernel):
    lowpass_mode = str(getattr(cfg, 'lowpass_mode', 'avg_pool')).lower().strip()
    if lowpass_mode == 'wavelet':
        down = F.avg_pool2d(y.float(), kernel_size=2, stride=2, ceil_mode=False)
        return F.interpolate(down, size=y.shape[-2:], mode='bilinear', align_corners=False).to(dtype=y.dtype)
    else:
        return F.avg_pool2d(y.float(), k, stride=1, padding=k // 2).to(dtype=y.dtype)
```

**配置**:
```json
{
  "model": { "lowpass_mode": "wavelet", "i2sb_fiber_project_kernel": 5 }
}
```

**config_schema.py 添加**: `lowpass_mode: str = "avg_pool"`

**原理**: Wavelet 低通（2×下采样→双线性插值）比 fixed-kernel avg_pool 在频率域有更锐利的截止，Base/Fiber 分割更干净。

**预期**: 更干净的分割 → LPIPS 改善 + clip_style 提升

---

### Round 3: 最优组合（基于 Round 1-2 结果动态选择）

#### F8 & F9: 组合实验

根据 Round 1-2 结果选择最佳组合。预设候选：
- 如果 F5(curriculum) 效果好: σ=0.06+curriculum+huber+kernel7
- 如果 F6(fiber_ep) 效果好: fiber_ep+σ=0.04+wavelet
- 如果 F1/F2(CFG) 效果好: 最佳训练配置 + 推理时 CFG 外推

---

### Round 4: 自主探索 (F10-F12)

基于前面 9 个实验的完整数据：
- 如果接近目标（clip>0.72 且 LPIPS<0.45）→ 微调该方向
- 如果发现新的帕累托点 → 沿该方向继续探索
- 可选: 尝试 coupling_cost 激活、不同 num_steps、多尺度 kernel
- 最终: 更新 Dashboard + 写结论报告

---

## Assumptions & Decisions

1. **基线配置**: 所有实验基于 **E2 (σ=0.04, endpoint, gate=0.5, rms_norm)** 作为起点
2. **batch_size=12**（FC-SB v1 验证安全，VRAM ~3.43GB）
3. **num_epochs=10**（除非特别说明）
4. **远程 GPU**: SSH `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`，WSL 环境 `/mnt/i/Github/Latent_Style/SchrodingerBridge`
5. **F1/F2 是纯推理实验**（使用已有 checkpoint），不需要训练，~10 分钟出结果
6. **代码改动范围**: model620.py（curriculum sigma + fiber ep + wavelet lp）、config_schema.py（3个新字段）
7. **每个实验后运行 full_eval** 并提取 clip_style + LPIPS
8. **Dashboard 在全部实验完成后统一更新**

## Verification Steps

### 每个 experiment 完成后检查:
1. [ ] 训练无 NaN / OOM / 崩溃
2. [ ] full_eval 完成，summary.json 存在
3. [ ] 提取 clip_style 和 LPIPS
4. [ ] 与 E2 基线对比

### 成功标准（按优先级）:
- [ ] **铜牌**: 任一实验达到 clip_style > 0.71 且 LPIPS < 0.50
- [ ] **银牌**: 任一实验达到 clip_style > 0.72 且 LPIPS < 0.45
- [ ] **金牌**: clip_style > 0.73 且 LPIPS < 0.38
- [ ] **钻石**: clip_style > 0.73 且 LPIPS < 0.30（FC-SB 终极目标）
- [ ] Dashboard 更新全部新数据点
- [ ] 最终结论报告

## 时间预算表

| 时段 | 任务 | 累计时间 |
|------|------|---------|
| T+0h | 代码改动 (curriculum/fiber_ep/wavelet/lp_mode) + 配置生成 | 0.5h |
| T+0.5h | **F1/F2**: CFG 外推（纯推理，用E2 checkpoint） | 1h |
| T+1h | **F3**: kernel7 训练 | 2h |
| T+2h | **F4**: floor0 训练 | 3h |
| T+3h | **F5**: curriculum 训练 | 4.5h |
| T+4.5h | **F6**: fiber_ep 训练 | 6h |
| T+6h | **F7**: wavelet 训练 | 7.5h |
| T+7.5h | 分析 R1-R2 → 决定 F8/F9 参数 | 8h |
| T+8h | **F8/F9**: 最优组合训练 × 2 | 11h |
| T+11h | 全部 Eval + 分析数据 | 12h |
| T+12h | **F10-F12**: 自主探索 | 16h |
| Buffer | 排障 / 重跑 / 额外消融 | ≤24h |
