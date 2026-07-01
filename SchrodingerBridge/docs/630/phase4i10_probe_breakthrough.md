# Phase 4I.10: Probe 诊断与结构性突破方案

更新日期：`2026-07-01`

## 1. Probe 诊断结果 (结合 MODEL.md 理论)

Probe 脚本: `tools/probe_spectral_global_bottleneck.py`
Checkpoint: `exp/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt` (当前 SOTA)

### 1.1 瓶颈 A: Velocity Field U 形死亡 (核心瓶颈)

**诊断**: velocity field 在 t=0.5 (轨迹中点) 几乎完全死亡。

| t | LL cos | LH cos | HL cos | LL amp | LH amp | HL amp |
|---|--------|--------|--------|--------|--------|--------|
| 0.0 | 0.60 | 0.63 | 0.63 | 0.56 | 0.49 | 0.50 |
| 0.25 | 0.39 | 0.37 | 0.38 | 0.42 | 0.39 | 0.40 |
| **0.50** | **0.10** | **0.03** | **0.01** | **0.23** | **0.10** | **0.10** |
| 0.75 | 0.36 | 0.41 | 0.41 | 0.25 | 0.40 | 0.40 |
| 1.0 | 0.57 | 0.62 | 0.62 | 0.38 | 0.47 | 0.48 |

**理论解释**: FM 训练目标 v = target - content (t 无关). 但在 t=0.5 时, x_t = 0.5*content + 0.5*target, 模型无法从混合状态中恢复方向. 这是 FM 的固有歧义: 中点方向不确定.

**结论**: velocity field 在中点完全失效, 仅在端点有效. 模型本质是 "端点校正器" 而非 ODE.

### 1.2 瓶颈 B: ODE 轨迹完全无效 (target_reach_ratio = 0.0009)

**诊断**: 8 步 ODE 积分几乎不向目标移动.

| step | t | d_to_src | d_to_tgt | step_Δ |
|------|---|----------|----------|--------|
| 1 | 0.000 | 9.18 | 143.56 | 0.076 |
| 5 | 0.500 | 32.35 | 130.87 | 0.025 |
| 8 | 0.962 | 23.08 | **149.03** | 0.186 |

**关键发现**: d_to_tgt 从 143.56 **增加到** 149.03 — 轨迹远离目标!
**target_reach_ratio = 0.0009** (1.0=完美, 0.0=无移动)

**结论**: ODE 积分完全无效. 所有风格迁移来自最后一步 AdaIN.

### 1.3 瓶颈 C: AdaIN 统计匹配失败

**诊断**: AdaIN 只修复 std, 同时恶化 mean 和协方差.

| 指标 | no_adain | with_adain | target | 修正率 |
|------|----------|------------|--------|--------|
| mean_l1 | 0.3238 | 0.3261 | 0.0000 | **-0.7%** (恶化!) |
| std_l1 | 0.0991 | 0.0673 | 0.0000 | 32.1% |
| cov_frob | 0.5153 | 0.4883 | 0.0000 | 5.2% (极差) |
| cov_offdiag_l1 | 0.0705 | 0.0768 | 0.0000 | **-8.9%** (恶化!) |

**L2 to target**: no_adain=142.6 → with_adain=149.0 (+6.4, **远离目标!**)

**结论**: AdaIN 的 mean+std 匹配破坏了协方差结构, 且整体将输出推离目标.

### 1.4 瓶颈 D: 风格敏感度倒置

**诊断**: LL (应锁内容) 风格敏感度最高, LH/HL (应传风格) 风格敏感度最低.

| 子带 | cos(v, v_shuffled) | |Δv|/|v| | 期望 |
|------|---------------------|----------|------|
| LL | 0.74 | 0.62 | 应 LOW (内容锁) |
| LH | 0.96 | 0.20 | 应 HIGH (风格传) |
| HL | 0.95 | 0.25 | 应 HIGH (风格传) |

**结论**: 风格条件信号泄漏到 LL, 而 LH/HL 几乎不响应风格. 这是反的.

### 1.5 瓶颈 E: 频域能量分布错误

**诊断**: 输出的 LL 占比下降, HH 过冲.

| | LL | LH | HL | HH |
|---|-----|-----|-----|-----|
| content | 63.2% | 12.3% | 13.6% | 10.9% |
| target | 61.6% | 12.6% | 13.8% | 11.9% |
| **output** | **54.4%** | 15.2% | 16.8% | **13.5%** |

**结论**: LL 能量丢失 (63%→54%), HH 过冲 (11%→13.5%). 输出的频谱分布偏离目标.

## 2. 根因分析

模型当前行为可分解为:
1. **ODE 积分** (8步): 几乎无效 (reach_ratio=0.0009), 仅在端点有弱效果
2. **AdaIN** (最后一步): 唯一有效的风格注入, 但只修 std, 同时恶化 mean/cov

**根本原因**: 模型被训练为 FM velocity field, 但 velocity field 在中点死亡 (结构性歧义), 导致 ODE 积分无效. 所有风格依赖 AdaIN, 而 AdaIN 的 mean+std 匹配不足以正确迁移风格.

**与 MODEL.md 理论的一致性**: 理论指出 "模型更像 learned endpoint corrector 而非高精度 ODE". Probe 证实: target_reach_ratio=0.0009, 模型确实不是 ODE.

## 3. 结构性突破方案 (4 个方向)

### 方向 1: Endpoint Prediction Training (EPT) — 拥抱端点校正器本质

**核心思想**: 停止假装是 ODE, 直接训练为端点预测器.

**改动**:
- 训练: 仅在 t=0 采样, loss = MSE(model(content, t=0, style), target)
- 推理: 单步前向, 无 ODE 积分
- 保留: 最后一步 AdaIN/WCT 风格精修

**预期**:
- velocity field 精度从 cos=0.60 提升到 cos>0.8 (聚焦 t=0)
- 训练速度 8x (无 ODE 积分)
- 推理速度 8x (单步 vs 8步)
- 风险: 失去多步精修能力, 但 probe 证明多步本来就无效

### 方向 2: Style-Subband Decoupling (SSD) — 修复风格敏感度倒置

**核心思想**: LL 锁死内容, LH/HL 专传风格.

**改动**:
- LL velocity head: 移除 style cross-attn (内容专用路径)
- LH/HL velocity heads: 增强 style cross-attn (风格专用路径)
- 实现: 在 backbone 后分叉, LL 走无风格路径, LH/HL 走强风格路径

**预期**:
- LL 风格敏感度从 0.62 降到 <0.1
- LH/HL 风格敏感度从 0.20 升到 >0.5
- LPIPS 改善 (LL 内容锁死)

### 方向 3: Covariance-Aware Refinement (CAR) — 修复 AdaIN

**核心思想**: 用 WCT 替代 AdaIN, 修复协方差匹配.

**改动**:
- 推理: spatial_fiber → spatial_fiber_wct
- WCT alpha: 0.3-0.5 (probe 显示 alpha=0.5 给 lpips=0.2971, 远超 SaMam 0.3282)
- Per-subband: LL 不做 WCT (内容锁), LH/HL 做 WCT with moderate alpha

**预期**:
- cov_frob 修正率从 5.2% 提升到 >30%
- mean_l1 不再恶化
- L2 to target 不再增加

**现状**: WCT alpha=0.5 已评估: clip=0.7200, lpips=0.2971
- vs SaMam (0.7222, 0.3282): lpips -9.5% (大幅改善), clip -0.3% (略降)
- 需要结合方向 1/2 提升 clip

### 方向 4: Base-Fiber Realignment (BFR) — 修复 AdaIN 推离目标

**核心思想**: AdaIN 只匹配 fiber 统计, 但 base (lowpass) 错误. 修复 base.

**改动**:
- 当前: h = lp(h) + alpha*matched_fiber + (1-alpha)*fiber
- 新增: lp(h) 也向 lp(target) 靠拢
- h = (1-alpha_base)*lp(h) + alpha_base*lp(target) + fiber_matched

**预期**:
- L2 to target 不再增加
- mean_l1 修正率从 -0.7% 转正

## 4. 推荐实施顺序

| 优先级 | 方向 | 理由 | 预期增益 |
|--------|------|------|----------|
| 1 | CAR (WCT alpha=0.5) | 已验证, 零训练成本 | lpips -9.5% |
| 2 | EPT (端点预测训练) | 直击核心瓶颈 | clip +5-10% |
| 3 | SSD (风格解耦) | 修复结构性缺陷 | lpips -3-5% |
| 4 | BFR (base 修复) | 补充 AdaIN | clip +1-2% |

**组合策略**: CAR + EPT + SSD 应能实现两方面大幅超过 SaMam.

## 5. 实验计划

### 4I.10a: WCT alpha=0.3 扫描 (CAR, 零训练)
- 在 4I.7b checkpoint 上测试 WCT alpha=0.3
- 预期: clip~0.715, lpips~0.30

### 4I.10b: Endpoint Prediction Training (EPT)
- 新配置: t_min=0.0, t_max=0.01 (仅 t≈0 采样)
- 训练 5 epochs, 评估
- 预期: clip>0.73, lpips<0.33

### 4I.10c: EPT + WCT alpha=0.5 (组合)
- 4I.10b checkpoint + WCT alpha=0.5 推理
- 预期: clip>0.73, lpips<0.30 (两方面超 SaMam)

### 4I.10d: Style-Subband Decoupling (SSD)
- LL head 移除 style cross-attn
- 训练 5 epochs, 评估
- 预期: lpips 进一步改善

## 6. 验收标准

- clip_style ≥ 0.7300 (SaMam 0.7222 + 1.1%)
- content_lpips ≤ 0.3100 (SaMam 0.3282 - 5.5%)
- 两方面均超过 SaMam
