# Phase 4E: Daubechies 平滑小波基替换 (Smooth Wavelet Basis)

**Date**: 2026-07-01
**Stage**: Phase 4E (加法 - 平滑正交基替换)
**Goal**: 将 Haar 小波 (2-tap, 1 vanishing moment) 升级为 Daubechies-2 (db2, 4-tap, 2 vanishing moments) 用于 endpoint AdaIN 低通路径, 消除 Haar 方块效应带来的棋盘格/锯齿伪影, 提升 clip_style 与画面高级感。

## 1. 理论设计 (用户方案一)

### 1.1 动机: Haar 小波的致命缺陷

**Haar 是最原始的小波** ($2 \times 2$ 阶跃函数), 存在两个致命缺陷:

1. **方块效应 (Blockiness)**: Haar 本质是 $2 \times 2$ 阶跃函数, 不够平滑。在频域做 AdaIN 甚至加噪声后, iDWT 回空间域时极易产生**棋盘格伪影 (Checkerboard Artifacts)** 和锯齿。
2. **消失矩 (Vanishing Moments) 不足**: Haar 的消失矩为 1, 无法表示连续的斜线和曲线。修改高频后极易产生马赛克。

### 1.2 Daubechies-2 (db2) 小波: 平滑正交基

**db2 小波** (Daubechies 4-tap):
- **滤波器长度**: 4 (vs Haar 的 2)
- **消失矩**: 2 (vs Haar 的 1)
- **正交性**: 完美正交, 保证 Perfect Reconstruction (PR)
- **空间域连续**: 滤波器在空间域上是连续且有重叠的 (4×4 vs Haar 的 2×2)
- **效果**: 当高频子带做激进 AdaIN 时, iDWT 逆变换把风格"平滑地"晕染到图像上, **彻底消除锯齿和棋盘格**

### 1.3 db2 滤波器系数 (4-tap)

**分解 (Analysis) 滤波器**:
```
Lo_D = [ 0.482962913145,  0.836516303738,  0.224143868042, -0.129409522551]  (低通)
Hi_D = [-0.129409522551, -0.224143868042,  0.836516303738, -0.482962913145]  (高通)
```

**重建 (Synthesis) 滤波器** (正交小波: 合成 = 分析的转置):
```
Lo_R = [-0.129409522551,  0.224143868042,  0.836516303738,  0.482962913145]  (低通)
Hi_R = [ 0.482962913145, -0.836516303738,  0.224143868042,  0.129409522551]  (高通)
```

### 1.4 周期边界 (Periodic Boundary) 实现

为保证 Perfect Reconstruction (PR), 采用**周期边界 (circular convolution)**:

- **分解**: `y[k] = sum_n filter[n] * x[(2k + n) mod N]`, for `k = 0, 1, ..., N/2 - 1`
- **重建**: `x[k] = sum_j lo_d[(k - 2j) mod N] * low[j] + sum_j hi_d[(k - 2j) mod N] * high[j]`, for `k = 0, ..., N-1`

**实现**: 使用 `torch.einsum` + 预计算的系数矩阵, 避免 `F.conv2d` 的 padding 对齐问题。

### 1.5 设计选择: 只改 lp(), 不改 forward

**与 Phase 4D 一致的策略**:
- `forward()` (训练路径): 继续使用 Haar DWT 分解输入为 4 子带 (LL/LH/HL/HH)
- `integrate_transport.lp()` (推理路径): 切换为 db2 低通, 用于 endpoint AdaIN

**理由**:
1. **零参数增加**: 不改训练架构, 3 个 velocity heads (LL/LH/HL) 保持不变
2. **风险极低**: 训练行为完全不变, 只改推理时的低通滤波器
3. **与 Phase 4D 成功模式一致**: Phase 4D 也是只改 lp() (1-Level → 2-Level), 取得 clip=0.7301 突破
4. **快速验证**: 半天代码, 可快速对比 Haar vs db2

**局限**: forward() 仍用 Haar, 训练时网络看到的子带是 Haar 分解。完整 db2 化需要重训 (留作 Phase 4F)。

## 2. 实现

### 2.1 代码改动

**`src/spectral620.py`** — 新增 db2 函数 + 调度器:
- `_db2_decompose_1d(x, dim)`: 1D db2 DWT (周期边界, einsum 实现)
- `_db2_reconstruct_1d(low, high, dim)`: 1D db2 IDWT
- `dwt2_db2(x)`: 2D db2 DWT, 返回 (LL, LH, HL, HH)
- `idwt2_db2(ll, lh, hl, hh)`: 2D db2 IDWT
- `dwt2_db2_lowpass(x, levels)`: N 级 db2 低通
- `dwt2_lowpass(x, levels, basis)`: 调度器 (haar | db2)

**`src/config_schema.py`** — 新增字段:
- `endpoint_lowpass_basis: str = "haar"` (可选: "haar" | "db2")

**`src/spectral_bridge620.py`** — `integrate_transport.lp()` 使用调度器:
- 读取 `endpoint_lowpass_basis` config
- `lp(y) = dwt2_lowpass(y, levels=lowpass_levels, basis=lowpass_basis)`

### 2.2 Perfect Reconstruction 验证

**Smoke test 结果** (全部 7 项 PASS):
- `[PR random]` max error: 8.88e-16 ✓
- `[PR zeros]` max error: 0.00e+00 ✓
- `[PR ones]` LL mean=2.0, HH mean≈0 ✓
- `[lowpass shape]` levels=1/2/3 输出尺寸 == 输入 ✓
- `[fiber adain]` **db2 fiber TV 比 Haar 低 34.5%** (ratio=0.6550) ✓ — 核心验证: db2 重叠 4-tap 支撑消除了 Haar 的 2×2 棋盘格伪影
- `[dispatcher]` haar/db2/unknown fallback 全部正确 ✓
- `[PR multi-channel float32]` max error: 5.96e-07 ✓

**关键发现**: db2 的低通 LL 本身 TV 比 Haar **高** (因为 db2 有 2 vanishing moments, 保留更多中频结构)。db2 的平滑优势体现在 **fiber (h - lp(h)) 经 AdaIN 修改后的重建** — 这是 endpoint AdaIN 的实际使用场景。

## 6. 实验结果

### 6.1 完整 2×2 消融矩阵

| 配置 | basis | levels | clip_style | content_lpips | v_ll_abs | 判定 |
|------|-------|--------|------------|---------------|----------|------|
| Phase 3 baseline (3ep) | haar | 1 | 0.7261 | 0.3296 | ~0.01 | PASS |
| **Phase 4D.1 (haar lvl2, 3ep)** | haar | 2 | **0.7301** | 0.3402 | — | **SOTA** |
| Phase 4E.1 (db2 lvl1, 3ep) | db2 | 1 | 0.7258 | 0.3288 | 0.666 | PASS |
| Phase 4E.2 (db2 lvl2, 3ep) | db2 | 2 | 0.7298 | 0.3398 | 0.666 | PASS |

### 6.2 关键发现

**db2 与 Haar 在聚合指标上持平**:
- db2 lvl1 vs haar lvl1: Δclip = -0.0003, Δlpips = -0.0008
- db2 lvl2 vs haar lvl2: Δclip = -0.0003, Δlpips = -0.0004
- 两个 level 上的 delta 完全一致 (-0.0003 clip), 说明 db2 基变换的效应是**恒定的**, 不随 level 变化

**多级分解 (lvl1→lvl2) 是主导效应**:
- haar: lvl1→lvl2 带来 +0.0040 clip (0.7261→0.7301)
- db2: lvl1→lvl2 带来 +0.0040 clip (0.7258→0.7298)
- 两种 basis 的多级增益完全相同 (+0.0040), 远大于 basis 切换的效应 (-0.0003)

### 6.3 物理解释

**为什么 db2 的理论平滑优势没有转化为 clip_style 提升?**

1. **CLIP 对纹理平滑度不敏感**: CLIP 特征提取器关注的是高层语义 (风格类别: 笔触类型、色彩分布), 而非像素级平滑度。db2 消除的棋盘格伪影在 CLIP embedding 空间几乎不可见。

2. **LPIPS 同样不敏感**: LPIPS (AlexNet) 衡量的是感知距离, 棋盘格伪影在 VAE latent 空间 (32×32×4) 的影响远小于在像素空间 (512×512×3) 的影响。我们在 latent 空间操作, db2 的平滑优势被 VAE 解码器吸收。

3. **训练 forward() 仍用 Haar**: db2 只改推理 lp() 路径, 训练时网络看到的子带是 Haar 分解。推理时切换 db2 等于引入微小的 distribution shift, 抵消了部分平滑收益。

### 6.4 结论

- **SOTA 不变**: Phase 4D.1 (haar lvl2) 仍是当前最优配置 (clip=0.7301)
- **db2 是有效的理论精炼**: PR 完美, fiber TV 降低 34.5%, 但聚合指标持平
- **保留 db2 代码**: 作为可选 basis (`endpoint_lowpass_basis: "db2"`), 不设为默认值
- **不再追求方案一 (Daubechies 系列升级)**: db2→db4→bior 等更高级 basis 预期同样持平, 因为根本原因是 CLIP/LPIPS 对像素级平滑度不敏感

### 6.5 对 Phase 4 后续方向的指导

| 用户方案 | Phase | 结论 |
|---------|-------|------|
| 方案二: 多级级联 | 4D | ✓ **有效** (+0.0040 clip), 已采用 |
| 方案一: Daubechies 平滑基 | 4E | ✗ 持平 (-0.0003 clip), 代码保留但不采用 |
| 方案三: DTCWT 复数小波 | — | 预期同样持平 (CLIP 不敏感), **暂缓** |
| 方案四: 可学习 Lifting | — | 可能让 LL 更好绑定 LPIPS, 但风险高, **待评估** |
| 方案五: 全频域 ODE | — | 改训练架构, 高风险高回报, **Phase 4F 候选** |

## 3. 实验矩阵

| 编号 | 配置 | lowpass_basis | lowpass_levels | epochs | 描述 |
|------|------|---------------|----------------|--------|------|
| baseline (4D.1) | `630_phase4d_lvl2.json` | haar | 2 | 3 | Phase 4D SOTA: clip=0.7301 |
| **4E.1** | `630_phase4e_db2_lvl1.json` | db2 | 1 | 3 | 隔离 db2 vs Haar 效应 (1-level) |
| **4E.2** | `630_phase4e_db2_lvl2.json` | db2 | 2 | 3 | db2 + 2-Level (最强组合) |

**验收阈值**: clip ≥ 0.7243, lpips ≤ 0.3453

### 3.1 实验设计逻辑

- **4E.1 vs baseline (4D.1)**: 不能直接对比 (basis 和 levels 都不同), 但可对照 4B-3 dwt_a1 (haar lvl1) 的 clip=0.7266
- **4E.2 vs 4D.1**: 隔离 db2 vs Haar 的效应 (两者都用 lvl2)
- **4E.2 vs 4E.1**: 隔离 2-Level vs 1-Level 的效应 (两者都用 db2)

### 3.2 物理意义预期

| 实验 | 预期 clip_style | 预期 lpips | 原因 |
|------|----------------|------------|------|
| baseline (4D.1, haar lvl2) | 0.7301 ⭐ | 0.3402 | 当前 SOTA |
| 4E.1 (db2 lvl1) | 0.728-0.732 | 0.335-0.345 | db2 平滑性提升, 但 1-level 限制中频释放 |
| **4E.2 (db2 lvl2)** | **0.732-0.738** | **0.335-0.345** | **预期最优: db2 平滑 + 2-Level 中频释放** |

**核心理论预测**: db2 消除 Haar 方块效应后, AdaIN 注入的风格更自然, clip_style 有望突破 0.7301。

## 4. 理论分析

### 4.1 Haar vs db2: 频率响应对比

**Haar (2-tap)**:
- 频率响应: $|H(\omega)|^2 = 1 + \cos(\omega)$
- 在 $\omega = \pi$ (Nyquist) 处: $|H(\pi)|^2 = 0$ (低通)
- **问题**: 通带不平坦, 阻带衰减慢 (只有 -20dB/decade)

**db2 (4-tap)**:
- 频率响应更平坦, 阻带衰减更快 (-40dB/decade)
- **优势**: 通带更平坦 → 低频成分更纯净; 阻带衰减更快 → 高频泄漏更少

### 4.2 Content Fidelity Pathway 的平滑升级

现有 (Phase 4D.1, Haar lvl2):
```
2-Level Haar DWT 低通 → Endpoint AdaIN → Spectral ODE → 风格外推
(LL₂ 锁死构图, 方块边界)   (fiber 含中频)   (head_ll)    (scale)
```

升级后 (Phase 4E.2, db2 lvl2):
```
2-Level db2 DWT 低通 → Endpoint AdaIN → Spectral ODE → 风格外推
(LL₂ 锁死构图, 平滑边界)  (fiber 含中频)   (head_ll)    (scale)
```

**关键差异**: db2 的平滑滤波器让 LL₂ 边界过渡自然, ep_fiber 不再含方块伪影, AdaIN 统计匹配更准确。

### 4.3 与用户 5 方案的对应

| 用户方案 | Phase | 状态 |
|---------|-------|------|
| 方案二: 多级级联分解 | 4D | ✓ 实现 (Haar lvl2) |
| 方案一: Daubechies 平滑基 | **4E (本)** | **待实现 (db2)** |
| 方案三: DTCWT 复数小波 | 长期 | 待实现 |
| 方案四: 可学习 Lifting | 长期 | 待实现 |
| 方案五: 全频域 ODE | 长期 (Paper 核心) | 待实现 |

**组合空间**: Phase 4E.2 = 方案一 + 方案二 = 平滑 + 多级 = **理论最强组合**。

## 5. 文件清单

- `src/spectral620.py` — 新增 db2 函数 + 调度器
- `src/config_schema.py` — 新增 `endpoint_lowpass_basis` 字段
- `src/spectral_bridge620.py` — `integrate_transport.lp()` 使用调度器
- `configs/630_phase4e_db2_lvl1.json` — 4E.1 配置
- `configs/630_phase4e_db2_lvl2.json` — 4E.2 配置
- `tools/smoke_db2.py` — db2 PR smoke test
- `docs/630/phase4e_daubechies_wavelet.md` — 本文档
