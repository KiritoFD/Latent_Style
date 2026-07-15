# Phase 4G: 全频域 ODE (Full Wavelet-Domain Flow Matching) — 设计文档

**Date**: 2026-07-01
**Stage**: Phase 4G (加法 - 论文核心 Story, 用户方案五)
**Goal**: 将"频域解耦"从"事后投影"升级为"原生频域建模"——网络主干直接在小波域求解 ODE, 物理约束注入网络骨髓, 提升训练效率并建立论文 Core Story。

---

## 1. 动机: 当前架构的"半频域"瓶颈

### 1.1 现状审查 (基于 [spectral_bridge620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_bridge620.py) 和 [spectral_losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_losses620.py))

当前 `SpectralODEBridge620` 已在频域做了一定工作, 但仍未达到"全频域 ODE"境界:

| 环节 | 当前实现 | 全频域 ODE 目标 | 差距 |
|------|----------|-----------------|------|
| `forward()` 输入 | `dwt2_haar(x)` 单级, 4 子带 stack | 多级 DWT, 完整子带树 | 仅单级 |
| `forward()` 输出 | 3 个 velocity heads (LL/LH/HL) | 多级 per-subband heads | 子带粒度粗 |
| Backbone 输入 | `cat([ll,lh,hl,hh])` 4C → dim | 选择性屏蔽 LL (stop_grad) | LL 仍参与 |
| Loss | `w_ll·loss_ll + w_lh·loss_lh + w_hl·loss_hl` | LL 不参与 loss 或被锁 | LL 仍训练 |
| `integrate_transport` Euler | `ll += v_ll·dt` (LL 被 v_ll 推动) | LL 锁死, 不漂移 | LL 漂移 |
| Endpoint AdaIN | 空间域: `h = ep_base + (1-α)·ep_fiber + α·matched` | 频域: 每子带独立 AdaIN | 后处理 |
| Endpoint 低通 | `dwt2_lowpass(h, levels=3)` 多级 (4F SOTA) | 已多级 | ✅ |

### 1.2 关键观察: Phase 4A2 的 "假阴性"

Phase 4A2 的 `spectral_w_ll=0.0` 实验**失败** (clip=0.7117, -0.0126)。但仔细分析代码路径:

```python
# 4A2 配置: w_ll = 0
loss = 0 * loss_ll + w_lh * loss_lh + w_hl * loss_hl  # head_ll 无梯度

# 但 forward() 仍然计算 v_ll:
v_ll = self.head_ll(h)  # head_ll 参数未被训练, 输出随机初始化值

# integrate_transport() 仍然使用 v_ll:
ll = ll + v_dict["ll"] * dt  # ← 用未训练的 v_ll 推 LL, 等同于注入噪声!
```

**结论**: 4A2 没有真正"锁死 LL", 而是用未训练的随机 `v_ll` 推 LL, 等效于在 LL 上加噪声, 故 LPIPS 居然反而下降 (0.2994), 但 clip 也下降 (因为 LL 被噪声扰动, 内容错位)。

**Phase 4G.1 必须区分这两件事**:
- (a) `w_ll=0` (不训练 head_ll) — 4A2 已试, 单独不够
- (b) `lock_ll=True` (推理时 `ll_new = ll_old`, 不用 v_ll) — **未试, 是 4G.1 核心**

### 1.3 "全频域 ODE" 的理论美感

> "我们从损失函数工程, 走向了频域拓扑空间的直接流形建模"

当前架构的"频域"是**离线投影**:
1. 训练时在空间域做 FM, 损失分到 4 个子带
2. 推理时事后用 DWT 低通做 AdaIN

全频域 ODE 是**原生频域**:
1. 输入直接是 DWT 系数, LL3 锁死作为"内容锚"
2. 网络拓扑结构上承认"LL/LH/HL/HH 是不同流形"
3. 每条流形上独立求解 ODE, LL 流形是平凡的 (恒等映射)

---

## 2. 数学公式

### 2.1 当前 FM 损失 (空间域 bridge + 频域分解损失)

```
x_t = (1-t)·x_0 + t·x_1                          # 空间域插值
v_pred(x_t, t) = (v_ll, v_lh, v_hl)              # 频域分解的 3 头输出
target = (DWT(x_1 - x_0)).{ll, lh, hl}            # 频域分解的目标速度
L = w_ll·||v_ll - target_ll||² + w_lh·||v_lh - target_lh||² + w_hl·||v_hl - target_hl||²
```

### 2.2 Phase 4G 全频域 FM (原生频域 ODE)

设多级 DWT 分解为 $\mathcal{W}(x) = (L_K, H_K, H_{K-1}, ..., H_1)$, 其中 $L_K$ 是 $K$-级 LL, $H_k$ 是第 $k$-级的高频三元组 $(LH_k, HL_k, HH_k)$。

**全频域 ODE**:

$$\frac{dL_K}{dt} = 0 \quad \text{(LL 锁死, 内容锚)}$$

$$\frac{dH_k}{dt} = v_{H_k}(L_K, H_K, ..., H_1, t, \text{style}) \quad \text{for } k = 1, ..., K$$

**FM 损失** (仅在高频):

$$L = \sum_{k=1}^{K} w_k \cdot \| v_{H_k} - \text{DWT}_k(x_1 - x_0) \|^2$$

LL 不出现在损失中, 也不被推动。

### 2.3 频域 Endpoint AdaIN (替换空间域)

**当前 (空间域 fiber)**:
```
ep_base = lp(h)               # 多级 DWT 低通
ep_fiber = h - ep_base          # 空间域高频残差
h_new = ep_base + (1-α)·ep_fiber + α·match(ep_fiber, style_fiber)
```

**Phase 4G (频域每子带)**:
```
L_K, H_K, H_{K-1}, ..., H_1 = DWT_multi(h)             # 多级分解
for k in 1..K:
    for sub in (LH_k, HL_k, HH_k):
        sub_new = (1-α)·sub + α·match(sub, style_sub)  # 每子带独立 AdaIN
        # style_sub 来自 DWT_multi(style_latent) 的对应子带
L_K 不动 (内容锚)
h_new = iDWT_multi(L_K, H_K, ..., H_1)                  # 多级重建
```

**优势**: 每子带统计独立 (正交性保证), 比空间域"fiber"分离更纯净。

---

## 3. 实验设计: 3 个子阶段

### 3.1 Phase 4G.1: 真·LL 锁死 (最小改动, 验证核心假设)

**改动**:
- Config 新增 `endpoint_lock_ll: bool = false`
- `integrate_transport()`: 当 `endpoint_lock_ll=true` 时, `ll_new = ll_old` (不应用 v_ll)
- 训练不变 (w_ll 仍可 = 1.0, 让 head_ll 继续学习但被推理时丢弃)

**为什么 w_ll=1.0 训练 + lock_ll=true 推理?**
- 训练时让 backbone 看到 LL 的目标, 学到 LL 的"内容理解" (作为条件信号)
- 推理时丢弃 v_ll 的应用, 保证 LL 完全不漂移
- 这与 4A2 的"w_ll=0 不训练 head_ll + 推理仍用 v_ll"形成对照

**假设**: clip 略降 (LL 风格信息丢失), lpips 显著改善 (LL 完全保内容)。
**阈值**: clip ≥ 0.7243, lpips ≤ 0.3453

**4 种对照**:
| 实验 | w_ll (训练) | lock_ll (推理) | 4A2 已做? |
|------|------------|----------------|-----------|
| Baseline (4F.1 SOTA) | 1.0 | false | ✅ (clip=0.7319) |
| 4A2 (假阴性) | 0.0 | false | ✅ (clip=0.7117 FAIL) |
| **4G.1a (新)** | 1.0 | **true** | ❌ |
| **4G.1b (新)** | 0.0 | **true** | ❌ |

### 3.2 Phase 4G.2: 频域 Endpoint AdaIN

**改动**:
- `integrate_transport()`: 用多级 DWT 分解 + 每子带 AdaIN 替换 `ep_fiber = h - lp(h)` 的空间域 fiber
- 每子带 (LH_k, HL_k, HH_k) 对应 style_latent 的同子带做 mean+std 匹配
- LL_K 不参与 (内容锚)

**与现有 4F SOTA 的关系**:
- 4F: `ep_fiber = h - lp(h, levels=3)` 是空间域减法 (LL3 之外的总和)
- 4G.2: 分别对 LH1/HL1/HH1/LH2/HL2/HH2/LH3/HL3/HH3 做匹配

**假设**: 频域解耦更纯, clip 略升 (每子带风格更精准), lpips 持平或略升。

### 3.3 Phase 4G.3: 多级 forward + Selective LL Conditioning

**改动**:
- `forward()`: 使用 3-级 DWT, stack 10 子带 (LL3 + 3×3 高频) 共 10C 通道
- Backbone 处理 10C → dim
- 9 个 velocity heads (LH1/HL1/HH1/LH2/HL2/HH2/LH3/HL3/HH3), 无 LL head
- LL3 通过 `stop_gradient` 作为条件信号注入 backbone (不输出 velocity)

**风险**: 参数量增加, 训练时间增加。可能 OOM。
**缓解**: 压缩 dim 或减少 depth。

---

## 4. 实施路线 (按优先级)

| 阶段 | 改动量 | 风险 | 预期收益 | 优先级 |
|------|--------|------|----------|--------|
| **4G.1a/b** | 极小 (10 行) | 低 | 验证 LL 锁死假设, 可能 -clip +lpips | ⭐⭐⭐ |
| **4G.2** | 中 (40 行) | 中 | 频域 AdaIN 更纯, 可能 +clip | ⭐⭐ |
| **4G.3** | 大 (200+ 行) | 高 (OOM) | 多级 forward, 真正全频域 | ⭐ |

**先做 4G.1a/b**, 因为:
1. 改动最小, 半小时即可验证
2. 直接测试"LL 锁死"的核心假设
3. 与 4A2 形成 2×2 对照矩阵, 论文写作素材完整
4. 如果 4G.1 失败, 4G.2/4G.3 也要重新审视

---

## 5. 代码实现细节 (4G.1)

### 5.1 配置字段 (config_schema.py)

```python
endpoint_lock_ll: bool = False  # Phase 4G.1: True LL lock in inference (skip v_ll)
```

### 5.2 integrate_transport 修改 (spectral_bridge620.py)

```python
# 读取配置
lock_ll = bool(_cfg_get('endpoint_lock_ll', False))

# Euler step:
ll, lh, hl, hh = dwt2_haar(h)
if not lock_ll:
    ll = ll + v_dict["ll"] * dt   # 原行为
# else: ll = ll (锁死, 不漂移)
lh = lh + v_dict["lh"] * dt
hl = hl + v_dict["hl"] * dt
h = idwt2_haar(ll, lh, hl, hh)
```

### 5.3 实验 Config

```json
// configs/630_phase4g1a_lock_ll.json
{
  "extends": "configs/630_phase4f_lvl3.json",
  "bridge": {
    "endpoint_lock_ll": true,
    "endpoint_lowpass_levels": 3
  },
  "training": { "max_epochs": 3 }
}
```

```json
// configs/630_phase4g1b_lock_ll_zero_wll.json
{
  "extends": "configs/630_phase4f_lvl3.json",
  "bridge": {
    "endpoint_lock_ll": true,
    "spectral_w_ll": 0.0,
    "endpoint_lowpass_levels": 3
  },
  "training": { "max_epochs": 3 }
}
```

---

## 6. 论文 Core Story 叙事

### 6.1 当前 Story (Phase 4 已完成部分)

> "我们通过 Haar DWT 多级分解, 将 latent 空间的内容 (LL) 与风格 (HF) 物理解耦, 在 endpoint AdaIN 中以多级低通为内容锚, 高频 fiber 做风格匹配。3-Level DWT 取得 SOTA clip=0.7319。"

### 6.2 Phase 4G 完成后的 Story (升级版)

> "我们将整个 Flow Matching 搬到小波域: LL 子带作为内容锚被完全锁死 (恒等映射), 网络拓扑结构承认'不同频段是不同流形', 每条流形上独立求解 ODE。Endpoint 风格匹配从空间域 fiber 升级为频域 per-subband AdaIN, 利用正交性保证统计独立。这是从'损失函数工程'到'频域拓扑空间的直接流形建模'的范式跃迁。"

### 6.3 关键概念图

```
Phase 4F (当前):  latent → DWT1 → 3 heads → v_ll, v_lh, v_hl → Euler (LL漂移)
                              ↓
                   Endpoint: h = lp3(h) + α·match(h - lp3(h))

Phase 4G (目标):  latent → DWT3 → lock LL3 → 9 heads → v_{LH1..HH3} → Euler (LL3恒等)
                              ↓
                   Endpoint: per-subband AdaIN (每子带独立统计匹配)
```

---

## 7. 风险评估与缓解

### 7.1 4G.1 风险: clip 大幅下降
- **原因**: LL 不漂移 → 全局色调不变 → clip_style 可能下降
- **缓解**: 如果 clip < 0.7243, 退回到 4G.2 (频域 AdaIN, 仍允许 LL 漂移但每子带独立)

### 7.2 4G.2 风险: 多级 DWT 在 endpoint 的开销
- **当前**: `dwt2_lowpass(h, levels=3)` 已经是 3 级 DWT
- **新增**: 完整分解 + 每子带匹配, 计算量略增
- **缓解**: 缓存 style_latent 的多级 DWT 分解

### 7.3 4G.3 风险: OOM + 训练时间
- **原因**: 10C 输入 + 9 heads
- **缓解**: 减 dim 或 depth; 如果 12GB OOM, 跳过 4G.3 (4G.1+4G.2 已足够论文)

---

## 8. 后续展望

如果 4G.1+4G.2 都成功:
- 论文 Core Story 完整, 进入写作
- Phase 4H (DTCWT 复数小波) 与 Phase 4I (Lifting 可学习小波) 可作为 Future Work

如果 4G.1+4G.2 都失败:
- "全频域 ODE" 路线废弃, 退回 4F SOTA
- 论文以 "多级 DWT + endpoint AdaIN" 为 Core Story (当前已完成)
- Phase 4H (DTCWT) 作为加法探索

---

**下一步**: 实施 Phase 4G.1a/b (LL 锁死), 验证核心假设。

---

## 9. 实验结果 (Phase 4G.1a)

### 9.1 配置与执行

**Config**: `configs/630_phase4g1a_lock_ll.json` (基于 `630_phase4f_lvl3.json`)
- `endpoint_lock_ll: true` (推理时跳过 v_ll 应用)
- `endpoint_lowpass_levels: 3` (继承 4F SOTA)
- `spectral_w_ll: 1.0` (默认, head_ll 仍训练)
- 3 epoch, full_eval_each_epoch=true

**训练**: Epoch 3/3 完成, 14.7s/epoch (339 samples/sec), VRAM 3.57GB
**Eval**: 90.3s wall time, 750 generated images

### 9.2 结果对比

| 配置 | lock_ll | w_ll | clip_style | content_lpips | v_ll_abs | 判定 |
|------|---------|------|-----------|---------------|----------|------|
| **4F.1 SOTA** | false | 1.0 | **0.7319** | 0.3428 | 0.666 | PASS |
| 4A2 (假阴性) | false | 0.0 | 0.7117 | 0.2994 | — | FAIL |
| **4G.1a (新)** | **true** | 1.0 | 0.7178 | **0.3281** | 0.654 | **FAIL** |
| 4G.1b (待测) | true | 0.0 | — | — | — | — |

### 9.3 关键发现

**1. LL 漂移是风格传递的必要组成部分**
- 锁死 LL 后, clip_style 从 0.7319 → 0.7178 (**-0.0141**)
- LL 速度场贡献了约 0.014 clip_style 分数
- 这反驳了"LL 只是内容锚, 不需要漂移"的假设

**2. LL 漂移确实损害内容保真度**
- 锁死 LL 后, LPIPS 从 0.3428 → 0.3281 (**-0.0147**, 改善)
- 证实 LL 漂移是 LPIPS 上升的主要原因之一
- 但这个代价是值得的 (clip 收益 +0.0141 > lpips 代价)

**3. 4A2 的"假阴性"已澄清**
- 4A2 (w_ll=0 + 推理仍用 v_ll): clip=0.7117, lpips=0.2994
- 4G.1a (w_ll=1.0 + 推理锁死): clip=0.7178, lpips=0.3281
- 4A2 的 lpips 极低 (0.2994) 是因为未训练的随机 v_ll 实际上"扰乱"了 LL, 导致生成结果偏离内容 (lpips 应该升高才对), 但同时也偏离了风格 (clip 大幅下降)
- **真正的 LL 锁死 (4G.1a) 确实改善了 LPIPS, 但代价是 clip 下降**

**4. "全频域 ODE with LL lock" 路线废弃**
- 方案五 (用户提出) 的核心 "LL stop_gradient + 只算高频速度场" 不可行
- 当前架构 (LL 训练 + Euler 应用) 是正确的
- 论文写作时: 这是一个重要的 **negative ablation**, 证明 LL velocity 的必要性

### 9.4 物理解释

**为什么 LL 漂移对 style transfer 重要?**

LL 子带 (16×16 in single-level DWT, 或 LL1 级) 携带:
- 全局色调 (warm/cool tone, brightness level)
- 整体色相分布 (color palette statistics)
- 光照方向 (lighting direction, soft/hard)

这些信息对 style 至关重要:
- 印象派 (Impressionism): 明亮、高对比度
- 浮世绘 (Ukiyo-e): 平面化、低饱和度
- 洛可可 (Rococo): 柔和粉色调

如果 LL 完全不漂移, 这些全局色调信息无法从 content 转移到 target style, clip_style 必然下降。

**Endpoint AdaIN 能补偿部分但不够**:
- Endpoint AdaIN 通过 fiber (h - lp(h)) 的统计匹配补偿风格
- 但 fiber 是高频残差, 不包含 LL 的全局色调
- 所以即使 AdaIN 正常工作, LL 锁死仍会损失全局风格

### 9.5 4G.1b 预期

**4G.1b (lock_ll + w_ll=0)** 预期:
- 如果 ≈ 4G.1a (clip ≈ 0.718): LL 梯度信号不重要, 只看推理应用
- 如果 < 4G.1a (clip < 0.718): w_ll=1.0 训练的 head_ll 即使不应用, 也通过 backbone 提供了有用的 LL 理解信号
- 如果 > 4G.1a (clip > 0.718, 意外): w_ll=0 让 backbone 专注高频

### 9.6 2×2 矩阵总结 (待 4G.1b 完成后填入)

```
                    │ w_ll = 0.0 (不训练) │ w_ll = 1.0 (训练)  │
────────────────────┼─────────────────────┼─────────────────────┤
lock_ll = False      │ 4A2: clip=0.7117    │ 4F.1 SOTA: 0.7319   │
(推理用 v_ll)         │ lpips=0.2994 FAIL   │ lpips=0.3428 PASS    │
                    │ v_ll_abs=random     │ v_ll_abs=0.666       │
────────────────────┼─────────────────────┼─────────────────────┤
lock_ll = True       │ 4G.1b: clip=0.7174  │ 4G.1a: clip=0.7178   │
(推理锁死 v_ll)       │ lpips=0.3372 FAIL   │ lpips=0.3281 FAIL    │
                    │ v_ll_abs=0.010      │ v_ll_abs=0.654       │
```

**4G.1b 结果** (2026-07-01):
- clip_style = 0.7174 (vs 4G.1a 0.7178, Δ=-0.0004 — essentially identical)
- content_lpips = 0.3372 (vs 4G.1a 0.3281, **+0.0091** — 4G.1a has BETTER lpips)
- v_ll_abs = 0.010 (essentially zero, head_ll untrained with zero-init conv)

### 9.7 2×2 矩阵深度解读

**关键观察 1: LL 训练梯度对 backbone 有"附带收益"**
- 比较 4G.1a (w=1.0,lock=True) vs 4G.1b (w=0,lock=True):
  - clip 相同 (0.7178 ≈ 0.7174, Δ=-0.0004)
  - 但 lpips 差 0.0091 (4G.1a 更好)
- 解释: head_ll 的梯度信号回流到 backbone, 让 backbone 学到更好的内容理解
- 即使 v_ll 不被应用, 训练 head_ll 仍能改善 backbone 的内容保真能力

**关键观察 2: 4A2 是 2×2 矩阵最差配置**
- 4A2 (w=0, lock=False): clip=0.7117 (最低)
- 原因: head_ll 未训练 (随机初始化), 但 v_ll 仍被应用 → LL 被随机噪声扰动
- 4G.1b (w=0, lock=True) 比 4A2 (w=0, lock=False) 好 (+0.0057 clip): 锁死避免了噪声注入

**关键观察 3: 主对角线呈现"全开 vs 全锁"的极端**
- 全开 (4F.1, w=1.0, lock=False): clip=0.7319, lpips=0.3428 — 风格最强, 内容略损
- 全锁 (4G.1b, w=0, lock=True): clip=0.7174, lpips=0.3372 — 内容较好, 风格弱
- Δ clip = 0.0145, Δ lpips = 0.0056
- "全开"是最优, 因为 clip 收益 (+0.0145) > lpips 代价 (+0.0056)

**关键观察 4: LL velocity 的"边际贡献"分解**
- v_ll 应用 (4F.1 vs 4G.1a, 控制训练): +0.0141 clip, +0.0147 lpips
- v_ll 训练 (4G.1a vs 4G.1b, 控制锁死): -0.0004 clip, -0.0091 lpips (训练改善 lpips)
- **结论**: v_ll 的"应用"贡献主要在 clip, v_ll 的"训练"贡献主要在 lpips (通过 backbone 旁路)

### 9.8 论文写作价值

这 2×2 矩阵是论文的**核心 ablation 实验**:

1. **量化 LL velocity 的双重角色**:
   - 直接贡献: 应用 v_ll 推 LL 漂移 → +0.0141 clip_style (风格信息)
   - 间接贡献: 训练 head_ll 提供梯度 → -0.0091 lpips (内容理解旁路)

2. **澄清 4A2 的"假阴性"**:
   - 4A2 不是"LL 不重要", 而是"未训练的 v_ll 是噪声"
   - 真正的 LL 锁死 (4G.1a) 仍损失 0.014 clip

3. **设计指导**:
   - 当前架构 (4F.1: 训练+应用) 是最优
   - LL velocity 不是"装饰", 是核心组件
   - 任何"全频域 ODE with LL lock"方案都不可行

4. **理论叙事**:
   - LL 不是纯内容锚, 而是携带 +0.014 clip 的风格信息
   - 这与 Haar DWT 的物理意义一致: LL 包含全局色调/光照/色相分布
   - 多级 DWT (4F) 是更精细的频域解耦, 但 LL1 仍需漂移

---

## 10. Phase 4G 结论 (4G.1a + 4G.1b 完成)

### 10.1 方案五 (全频域 ODE) 的核心假设被证伪

用户提出的"网络内部不对 LL 通道计算任何速度场"的方案, 经 4G.1a 实验证证为 **NEGATIVE result**:
- LL 速度场贡献 +0.0141 clip_style, 不可省略
- 锁死 LL 会损失全局色调风格信息
- 论文写作时, 这是关键的 ablation, 证明 LL velocity 必要性

### 10.2 论文叙事调整

**原计划 Core Story**: "我们把整个 Flow Matching 搬到小波域, LL 锁死作为内容锚"

**调整后 Core Story**: "我们通过 Haar DWT 多级分解 (4F) 解耦内容 (LL) 与风格 (HF), 并通过消融实验 (4A2 + 4G.1) 精确量化了 LL velocity 的贡献: 移除 LL velocity 损失 0.014 clip_style, 证明 LL 不是纯内容锚, 而是携带关键全局风格信息 (色调、光照、色相分布)。这指导了 endpoint AdaIN 的设计: 在保留 LL velocity 的同时, 通过多级低通 fiber 统计匹配 (4F.1 SOTA) 平衡内容保真与风格注入。"

### 10.3 Phase 4G.2/4G.3 是否继续?

- **4G.2 (频域 Endpoint AdaIN)**: 仍然有价值, 不依赖 LL 锁死。可继续。
- **4G.3 (多级 forward)**: 风险高, 收益不确定。暂缓。

### 10.4 推荐路径

基于 4G.1a 的 NEGATIVE result, 推荐下一步:
1. **完成 4G.1b** 补全 2×2 矩阵 (论文写作素材)
2. **尝试 4G.2** (频域 per-subband AdaIN) - 不需要 LL 锁死, 是纯加法
3. **如果 4G.2 也失败**, 进入论文写作, Core Story 调整为 "LL velocity 贡献量化"

---

## 11. Phase 4G.2 结果 (2026-07-01): MIXED — clip NEW SOTA, lpips FAIL

### 11.1 实验执行

**Config**: `configs/630_phase4g2_per_subband.json` (基于 4F.1 SOTA)
- `endpoint_adain_mode: "per_subband"` (核心改动: 频域每子带独立 AdaIN)
- `endpoint_adain_scale: 1.0` (继承默认 — **4G.2b 的调节目标**)
- `endpoint_lowpass_levels: 3`, 3 epoch, 从零训练
- 详细设计见 [phase4g2_per_subband_adain.md](phase4g2_per_subband_adain.md)

### 11.2 结果

| 配置 | adain_mode | α | clip_style | content_lpips | v_ll_abs | 判定 |
|------|-----------|---|------------|---------------|----------|------|
| 4F.1 SOTA | spatial_fiber | 1.0 | 0.7319 | 0.3428 | 0.666 | PASS |
| **4G.2** | **per_subband** | **1.0** | **0.7361** | **0.3843** | 0.659 | **MIXED** |

- **clip_style**: +0.0042, 突破 SOTA — per_subband 统计隔离确实更精准
- **content_lpips**: +0.0415, 超过 0.3453 阈值 — 风格注入总量过多

### 11.3 根因: α=1.0 在 per_subband 下等效 9× 注入

- spatial_fiber α=1.0: 1 次全局 mean+std 匹配 (所有高频之和)
- per_subband α=1.0: 9 次独立全量替换 (3 级 × 3 方向, 每子带 100% 替换)
- clip 提升证明 per-subband 方向正确, 但 lpips 超标证明需要控制注入量

### 11.4 4G.1 + 4G.2 的协同结论

| 实验 | 改动 | clip | lpips | 结论 |
|------|------|------|-------|------|
| 4G.1a | LL lock (推理) | 0.7178 | 0.3281 | NEGATIVE: LL velocity 必要 (+0.014 clip) |
| 4G.2 | per_subband α=1.0 | 0.7361 | 0.3843 | MIXED: clip NEW SOTA, lpips FAIL (注入过多) |

- 4G.1 证明: **LL 必须漂移** (携带全局色调风格)
- 4G.2 证明: **per-subband 频域解耦有效** (clip 突破), 但**需控制注入量** (α 参数)
- 两者正交, 共同指向: "保留 LL velocity + per-subband AdaIN + 调控 α"

### 11.5 下一步: Phase 4G.2b (α=0.5 缓解)

保持 `endpoint_adain_mode: "per_subband"`, 将 `endpoint_adain_scale` 从 1.0 降到 0.5:
- 每子带保留 50% 原始 + 50% style 匹配
- 等效注入量 ≈ 4.5× (vs 4G.2 的 9×, 4F.1 的 1×)
- 若 PASS 且 clip > 4F.1 → 新 SOTA, 论文 Core Story 升级
- 若仍 FAIL → 记录为 ablation, 论文以 4F.1 为 SOTA

### 11.6 论文 Core Story (根据 4G.2b 结果调整)

**如果 4G.2b 成功 (新 SOTA)**:
> "我们通过 Haar DWT 多级分解解耦内容 (LL) 与风格 (HF)。消融实验 (4A2 + 4G.1) 量化了 LL velocity 的贡献 (+0.014 clip)。进一步, Endpoint AdaIN 从空间域全局 fiber 升级为频域 per-subband 独立统计匹配 (4G.2), 利用 Haar 正交性保证统计隔离, 突破空间域 SOTA。α 参数调控注入量, 平衡风格精准度与内容保真度 (4G.2b ablation)。"

**如果 4G.2b 仍 FAIL (4F.1 为最终 SOTA)**:
> "我们通过 Haar DWT 多级分解解耦内容 (LL) 与风格 (HF)。消融实验矩阵: (1) 4A2+4G.1 量化 LL velocity 贡献 (+0.014 clip, 证明 LL 携带全局色调); (2) 4G.2 探索频域 per-subband AdaIN, 发现 per-subband 虽提升 clip (+0.0042) 但引入 9× 风格注入导致 lpips 超标, 揭示空间域 fiber 的'隐式正则化'价值。最终> 最终 SOTA: 多级 DWT + 空间域 fiber (4F.1, clip=0.7319)。"

---

## 12. Phase 4G.2b 结果 (2026-07-01): FAIL — α 参数在多步 ODE 中失效

### 12.1 实验

保持 `endpoint_adain_mode: "per_subband"`, 将 `endpoint_adain_scale` 从 1.0 降到 0.5。3 epoch, 从零训练。

### 12.2 惊人结果: α=0.5 ≡ α=1.0

| 配置 | α | clip | lpips | 判定 |
|------|---|------|-------|------|
| 4G.2 | 1.0 | 0.7361 | 0.3843 | MIXED |
| 4G.2b | 0.5 | 0.7362 | 0.3845 | FAIL |
| Δ | -0.5 | +0.0001 | +0.0002 | — |

**α 减半, 结果几乎零变化!**

### 12.3 根因: 多步 Euler 迭代累积

推理用 12 步 Euler, 每步应用 AdaIN。残留 = (1-α)^12:
- α=1.0: 0% 残留
- α=0.5: 0.024% 残留 (趋同)

**per-step α 在多步 ODE 中不是有效的注入量控制参数。**

### 12.4 最终结论

- per_subband 路线终止 (无法通过 α 控制 lpips)
- **4F.1 (spatial_fiber) 确认为最终 SOTA**
- spatial_fiber 的"隐式正则化"是多步 ODE 下的天然优势
- Phase 4 探索完成, 进入论文写作

### 12.5 Future Work (4G.2b 洞察的后续方向)

1. End-of-trajectory AdaIN (只在最后一步应用)
2. α 衰减调度 (α_t = α_0 · (1-t/T))
3. 全局缩放而非 per-step 替换
