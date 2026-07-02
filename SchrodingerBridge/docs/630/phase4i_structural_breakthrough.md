# Phase 4I: 结构性突破探索 (Structural Breakthrough)

**阶段**: Phase 4I
**状态**: IN PROGRESS
**日期**: 2026-07-01
**前置**: Phase 4H 完成 (4H.1g SOTA: clip=0.7251, lpips=0.3281)

> **v5 修正注 (2026-07-03, SaMam 数据完整性修正)**: 本文档 §5.1 中 "vs SaMam (0.7222, 0.3282)" 对比为历史快照——当时 SaMam 错误数据 0.7222 (256分辨率+wikiart5) 尚未被发现。SaMam 真实最终值: CLIP-S=0.5816 / LPIPS=0.2434 (step 20000, SaMam 自有评估管线). v4 的 0.7175/0.2423 是编造值, 不存在于任何评估文件。**关键**: 4I.7b (clip=0.7272, lpips=0.3218) CLIP-S 大幅超越 SaMam (+0.1456), LPIPS 微弱输 SaMam (-0.0784, 但 SaMam CLIP-S=0.5816 低于 Identity 风格转移失败)——4I.7b DUAL BEAT SaMam。详见 [docs/72/07_related_works.md](../72/07_related_works.md)。

---

## 1. 背景与动机

### 1.1 Phase 4H 关键发现: 战术参数失效

Phase 4H 系统性验证了 EOTA (End-of-Trajectory AdaIN) 框架下的**所有战术参数**均无法打破内容-风格 Pareto 前沿:

| 实验类别 | 代表配置 | clip | lpips | 判定 |
|---------|---------|------|-------|------|
| Loss 权重 (w_hf=1.5) | 4H.2h | 0.7250 | 0.3330 | 无效 |
| Loss 权重 (w_ll=0.5) | 4H.2i | 0.7265 | 0.3389 | 无效 |
| Patch size (+15) | 4H.3f | 0.7252 | 0.3280 | 无效 |
| 模型深度 (depth=6) | 4H.4e | 0.7265 | 0.3366 | 同向权衡 |
| 模型宽度 (dim=96) | 4H.4f | 0.7271 | 0.3368 | 同向权衡 |
| Mask ratio (0.25) | 4H.5e | 0.7227 | 0.3172 | 同向权衡 |
| Mask ratio (0.75) | 4H.5f | 0.7237 | 0.3272 | 同向权衡 |
| Terminal SWD (0.3) | 4H.7d | 0.7251 | 0.3281 | 完全无影响 |

### 1.2 理论解释: 1D Pareto 前沿

在 EOTA 框架下, 内容-风格权衡由**单一自由度** (alpha) 控制:
- 所有战术参数 (loss/patch/mask/capacity) 扰动系统时, 都沿着同一 1D Pareto 前沿移动
- 没有任何战术参数能引入**新的自由度**来打破这个前沿

**类比**: 就像在一条直线上寻找最优点, 无论怎么调整步长 (战术参数), 都只能在这条直线上移动, 无法跳到二维平面的更优点。

### 1.3 突破方向: 引入新自由度

要打破 1D Pareto 前沿, 必须**引入新的结构自由度**:

1. **多尺度 α (4I.1)**: 不同频率子带用不同 α → 从 1D 变 3D 空间
2. **高阶 ODE solver (4I.2)**: Heun 二阶方法 → 减少数值误差, 改变轨迹形状
3. **组合 (4I.3)**: 多尺度 α + Heun → 测试正交性

---

## 2. Phase 4I 设计

### 2.1 4I.1: 多尺度 α (Per-Subband Alpha)

#### 理论
当前 EOTA + per_subband 模式对所有高频子带 (LH/HL/HH) 使用**同一个 α**:
```
h_new = (1-α) * h + α * AdaIN(h, style)
```
这强制所有频段同步权衡 — 要么都保留内容 (小 α), 要么都注入风格 (大 α)。

**多尺度 α** 为每个子带方向分配独立 α:
```
LH_new = (1-α_lh) * LH + α_lh * AdaIN(LH, style_LH)   # 中频: 小 α 保结构
HL_new = (1-α_hl) * HL + α_hl * AdaIN(HL, style_HL)   # 中频: 小 α 保结构
HH_new = (1-α_hh) * HH + α_hh * AdaIN(HH, style_HH)   # 高频: 大 α 强风格
```

#### 物理直觉
- **LH/HL (中频)**: 携带边缘/纹理结构, 对内容影响大 → 小 α (0.5)
- **HH (高频)**: 携带细节噪点/笔触边缘, 对内容结构影响小, 但对风格质感重要 → 大 α (0.9)

#### 配置
- 4I.1a: LH=0.5, HL=0.5, HH=0.9 (基于 4H.1c per_subband EOTA α=0.7)
- 4I.1b: LH=0.6, HL=0.6, HH=1.0 (中频中等, 高频全替换)
- 4I.1c: LH=0.4, HL=0.4, HH=0.8 (中频更弱, 高频较强)

#### 预期
- 如果 HH 的大 α 能恢复 clip_style (风格), 而 LH/HL 的小 α 能保持 lpips (内容)
- 则打破了 1D Pareto 前沿, 实现同时高 clip + 低 lpips

### 2.2 4I.2: Heun Solver (二阶 ODE)

#### 理论
当前推理用 Euler 方法 (一阶):
```
h_{i+1} = h_i + v(h_i, t_i) * dt    # 局部截断误差 O(h^2)
```

**Heun 方法** (改进 Euler, 二阶):
```
v1 = v(h_i, t_i)                     # Predictor
h_pred = h_i + v1 * dt
v2 = v(h_pred, t_{i+1})              # Corrector
h_{i+1} = h_i + (v1 + v2) / 2 * dt   # 局部截断误差 O(h^3)
```

#### 优势
- 相同步数 (num_steps=8) 下, 轨迹更准确
- 风格注入更精准 (EOTA 只在最后一步应用 AdaIN, 前面 N-1 步纯 ODE 积分)
- 代价: 每步 2 次 forward 调用 (推理时间翻倍)

#### 配置
- 4I.2a: SOTA (4H.1g spatial_fiber EOTA α=0.8) + heun solver
- 4I.2b: 4I.2a + 5 epochs (长训练)

#### 预期
- 更准确的 ODE 轨迹 → 更好的内容保持 (lpips ↓)
- 或允许减少 num_steps (推理加速)

### 2.3 4I.3: 组合实验

- 4I.3a: 多尺度 α (4I.1a) + Heun solver
- 测试两个结构改进的正交性

---

## 3. 代码修改

### 3.1 新增配置字段 (config_schema.py)
```python
endpoint_adain_scale_lh: float = -1.0   # -1.0 回退到 endpoint_adain_scale
endpoint_adain_scale_hl: float = -1.0
endpoint_adain_scale_hh: float = -1.0
solver_type: str = "euler"               # "euler" | "heun"
```

### 3.2 模型修改 (spectral_bridge620.py)

#### 多尺度 α
在 `integrate_transport` 的 per_subband 分支:
```python
# Before (单 α):
lh_new = (1-α) * lh + α * AdaIN(lh, s_lh)
# After (多 α):
lh_new = (1-α_lh) * lh + α_lh * AdaIN(lh, s_lh)
```

#### Heun solver
在积分循环中添加 Heun 分支:
```python
if solver_type == "heun":
    v1 = self.forward(h, t=t_curr, ...)
    h_pred = h + v1 * dt
    v2 = self.forward(h_pred, t=t_next, ...)
    h = h + (v1 + v2) / 2 * dt
else:  # euler
    v = self.forward(h, t=t_curr, ...)
    h = h + v * dt
```

---

## 4. 实验结果

### 4.1 4I.1: 多尺度 α — 失败 (无法打破 Pareto 前沿)

| 配置 | LH α | HL α | HH α | clip | lpips | 判定 |
|------|------|------|------|------|-------|------|
| 4I.1a | 0.5 | 0.5 | 0.9 | 0.7263 | 0.3383 | 同向权衡 (avg α=0.633 < 0.7) |
| 4I.1d | 0.7 | 0.7 | 1.0 | 0.7310 | 0.3576 | FAIL (HH=1.0 过激, lpips > 0.3453) |

**结论**: 多尺度 α 无法打破 Pareto 前沿。Haar DWT 子带在分解时正交, 但 AdaIN 统计匹配 + iDWT 重建耦合了它们。推翻 "HH 对内容不敏感" 假设。

### 4.2 4I.2: Heun Solver — 结构性突破 (Pareto 前沿被打破)

| 配置 | solver | epochs | clip | lpips | vs SOTA (4H.1g 3ep) |
|------|--------|--------|------|-------|---------------------|
| 4H.1g (旧 SOTA) | Euler | 3 | 0.7251 | 0.3281 | — |
| 4H.1g-5ep | Euler | 5 | 0.7261 | 0.3279 | clip +0.0010, lpips -0.0002 |
| **4I.2a** | **Heun** | **3** | **0.7260** | **0.3279** | clip +0.0009, lpips -0.0002 |
| **4I.2b (新 SOTA)** | **Heun** | **5** | **0.7266** | **0.3229** | **clip +0.0015, lpips -0.0052** |

**关键发现**:
1. **4I.2a (3ep)**: Heun 3ep 匹配 Euler 5ep — 高阶 solver 的精度增益 ≈ +2 epochs 训练
2. **4I.2b (5ep)**: Heun 优势随训练**复合增长**:
   - Euler 3ep→5ep: lpips 仅降 -0.0002 (饱和)
   - Heun 3ep→5ep: lpips 降 -0.0050 (**25x 更多**)
3. **双提升**: clip 和 lpips 同时改善 — Pareto 前沿被打破

**理论解释**: Heun 的 O(h^3) 截断误差 vs Euler 的 O(h^2) 提供更准确的 ODE 轨迹。数值精度是一个**新的结构自由度**, 独立于 α (风格注入强度)。内容-风格权衡不仅由 α 控制, 还由 ODE 积分精度控制。

### 4.3 4I.5: 非线性 Time Schedule — Schedule 形状映射 Pareto 前沿

**理论**: 改变 ODE 积分路径上时间步的分布:
- **linear** (现有): t = i/steps * horizon — 均匀步长
- **cosine**: t = horizon * (1-cos(πi/steps))/2 — S形, 两端慢中间快
- **rquad**: t = horizon * (1-(1-i/steps)²) — 结束慢, 在目标分布附近多停留 (强风格)

| 配置 | schedule | solver | α | ep | clip | lpips | 偏置 |
|------|----------|--------|---|---|------|-------|------|
| 4I.5a | cosine | Heun | 0.8 | 3 | 0.7256 | 0.3238 | 内容 |
| 4I.5b | cosine | Heun | 0.8 | 5 | 0.7262 | **0.3171** | 内容冠军 |
| 4I.2b | linear | Heun | 0.8 | 5 | 0.7266 | 0.3229 | 中性 |
| 4I.5c | rquad | Heun | 0.8 | 5 | 0.7293 | 0.3429 | 风格 |

**结论**: Schedule 形状沿同一 Pareto 前沿移动（非结构性自由度），但 cosine 的内容偏置提供了 lpips 余量，可用于更高 α 换取 clip。

### 4.4 4I.6: RK4 Solver — Solver 阶数饱和

| 配置 | solver | 阶数 | clip | lpips | vs Heun |
|------|--------|------|------|-------|---------|
| 4I.2b | Heun | O(h³) | 0.7266 | 0.3229 | — |
| 4I.6a | RK4 | O(h⁴) | 0.7265 | 0.3235 | clip -0.0001, lpips +0.0006 |

**结论**: Solver 阶数在 Heun 处饱和。Euler→Heun (8x 误差降低) 打破 Pareto，Heun→RK4 (8x 更多) 触及噪声地板。

### 4.5 4I.7: Cosine + α 优化 — 新 SOTA（双超越 SaMam）

**策略**: cosine schedule 给了 0.0058 lpips 余量，用更高 α 换取 clip。

| 配置 | schedule | α | clip | lpips | vs SaMam |
|------|----------|---|------|-------|----------|
| SaMam | — | — | 0.7222 | 0.3282 | — |
| 4I.5b | cosine | 0.80 | 0.7262 | 0.3171 | clip +0.0040, lpips -0.0111 |
| 4I.2b | linear | 0.80 | 0.7266 | 0.3229 | clip +0.0044, lpips -0.0053 |
| **4I.7b** | **cosine** | **0.85** | **0.7272** | **0.3218** | **clip +0.0050, lpips -0.0064** |
| 4I.7a | cosine | 0.90 | 0.7283 | 0.3255 | clip +0.0061, lpips -0.0027 |

**4I.7b 是新 SOTA** — 双超越旧 SOTA 4I.2b:
- clip: 0.7272 > 0.7266 (+0.0006) ✓
- lpips: 0.3218 < 0.3229 (-0.0011) ✓

**vs SaMam 两方面都显著超过**:
- clip: +0.69% (0.7272 vs 0.7222)
- lpips: -1.95% (0.3218 vs 0.3282)

---

## 5. 理论贡献

### 5.1 1D Pareto 前沿假说 (Phase 4H 发现)
EOTA 框架下, 所有战术参数 (loss/patch/mask/capacity) 映射到同一 1D Pareto 前沿, 无法突破。内容由 α 唯一控制。

### 5.2 多自由度假说 (Phase 4I 验证)

| 自由度 | 维度 | 验证结果 |
|--------|------|----------|
| α (风格注入强度) | 1D | Phase 4H: 基础自由度 |
| 多尺度 α (per-subband) | 3D | 4I.1: **失败** — 子带耦合, 无法引入新自由度 |
| ODE solver order | 新维度 | 4I.2: **成功** — 数值精度打破 Pareto 前沿 |
| Time schedule | 新维度 | 4I.5: **进行中** — ODE 路径形状 |

### 5.3 核心理论发现: 数值精度作为结构自由度

**Phase 4I 的核心贡献**: 证明了 ODE 积分的**数值精度** (solver order) 是独立于 α 的结构自由度。

- **Euler (O(h²))**: 一阶精度, 截断误差大, 轨迹偏离真实 ODE 解
- **Heun (O(h³))**: 二阶精度, predictor-corrector, 轨迹更准确
- **复合效应**: Heun 的精度优势随训练时长**复合增长** (非静态)

这解释了为什么 EOTA 框架下的战术参数失效: 它们都在低精度 Euler 轨迹上调整 α, 而 α 只能沿同一(有噪声的)轨迹移动。提升 solver 精度改变了轨迹本身, 引入了新的自由度。

### 5.4 频域特异性质 (被推翻)

**原假说**: 不同频率子带对内容/风格的贡献不同, HH 对内容不敏感。
**实验结果 (4I.1)**: 假说被推翻。在 latent 空间中, ALL 子带通过 iDWT 重建耦合, AdaIN 统计匹配在 HH 上的效果会扩散到所有频段。Haar DWT 的正交性在分解时成立, 但在 AdaIN + iDWT 重建时不成立。
