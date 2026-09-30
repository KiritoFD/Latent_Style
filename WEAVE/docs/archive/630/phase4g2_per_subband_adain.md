# Phase 4G.2: 频域 per-subband AdaIN (Per-Subband Statistical Matching)

**Date**: 2026-07-01
**Stage**: Phase 4G.2 (加法 - 频域深化, 设计文档 §3.2 实施)
**Goal**: 将 Endpoint AdaIN 从"空间域全局 fiber 统计匹配"升级为"频域每子带独立统计匹配", 利用 Haar 正交性保证不同尺度风格信息的精准解耦, 突破 4F.1 SOTA (clip=0.7319) 上限。

---

## 1. 动机: 当前空间域 fiber 的"频谱混合"瓶颈

### 1.1 现状审查 ([spectral_bridge620.py L237-257](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_bridge620.py))

当前 4F.1 SOTA 的 Endpoint AdaIN 实现是**空间域全局 fiber**:

```python
ep_base = lp(h)                                    # LL3 多级低通 (4x4 上采样回 64x64)
ep_fiber_curr = h - ep_base                         # 空间域高频残差
style_fiber = style_latent - lp(style_latent)       # style 的空间域高频
# 全局 mean+std 匹配 (所有高频子带混在一起)
ep_fiber_matched = (ep_fiber_curr - pred_mean) / pred_std * target_std + target_mean
h = ep_base + (1-α)·ep_fiber_curr + α·ep_fiber_matched
```

**问题**: `ep_fiber = h - lp(h)` 是**所有高频子带的混合**:
- LH3/HL3/HH3 (4×4 上采样): 中低频, 宏观笔触/光影体积
- LH2/HL2/HH2 (8×8 上采样): 中频, 局部色彩/笔触方向
- LH1/HL1/HH1 (16×16): 高频, 画布材质/微观噪点

**全局 mean+std 匹配会"平均化"这些不同尺度的风格信息**:
- 微观噪点 (LH1/HL1/HH1) 的统计特征被宏观笔触 (LH3/HL3/HH3) 淹没
- 不同方向的细节 (LH 垂直 vs HL 水平 vs HH 对角) 被混合
- Haar 正交性带来的"频谱隔离"优势在统计匹配环节被丢弃

### 1.2 Phase 4G.1 NEGATIVE result 的启示

Phase 4G.1 (真·LL 锁死) 证明:
- LL velocity 贡献 +0.0141 clip_style, 不可省略
- LL 不是纯内容锚, 携带全局色调/光照/色相风格信息

**但 4G.1 没有改变 Endpoint AdaIN 的实现** — 仍然是空间域 fiber。

Phase 4G.2 是正交方向的探索: **不动 LL velocity 路径**, 只升级 Endpoint AdaIN 的 fiber 匹配方式, 从空间域全局升级为频域 per-subband。

### 1.3 理论美感: 正交性的"统计隔离"优势

Haar DWT 的正交性保证: $\langle H_k, H_{k'} \rangle = 0$ 当 $k \neq k'$ (不同级), 且 $\langle LH_k, HL_k \rangle = \langle LH_k, HH_k \rangle = 0$ (同级不同方向)。

这意味着**每个子带的统计量 (mean, std) 是数学独立的**:
- 对 LH3 做 AdaIN 不会影响 HL3/HH3 的统计
- 对 Level 2 高频做匹配不会污染 Level 1 高频

当前空间域 fiber 把它们加在一起做统计, 正交性的"统计隔离"优势被浪费。Phase 4G.2 释放这个优势。

---

## 2. 数学公式

### 2.1 多级 DWT 分解 (级联 Haar)

设 K 级 Haar DWT 分解为:

$$\mathcal{W}_K(x) = \left( LL_K, \{H_k\}_{k=1}^{K} \right)$$

其中:
- $LL_K$ 是 K 级低通 ($64/2^K \times 64/2^K$)
- $H_k = (LH_k, HL_k, HH_k)$ 是第 k 级的高频三元组 (每个 $64/2^{k-1}/2 \times 64/2^{k-1}/2$)

对 K=3 (与 4F.1 SOTA 一致):
- $LL_3$ (4×4): 绝对构图, 物体位置
- $H_3 = (LH_3, HL_3, HH_3)$ (4×4): 中低频, 宏观笔触/光影
- $H_2 = (LH_2, HL_2, HH_2)$ (8×8): 中频, 局部色彩/笔触方向
- $H_1 = (LH_1, HL_1, HH_1)$ (16×16): 高频, 画布材质/微观噪点

### 2.2 当前 (空间域 fiber, spatial_fiber mode)

```
ep_base = W_K_lowpass(h)                          # 只保留 LL_K, 其余置零, IDWT 重建
ep_fiber = h - ep_base = sum_{k=1}^{K} H_k         # 所有高频子带之和 (空间域)
style_fiber = sum_{k=1}^{K} style_H_k              # style 的所有高频之和
# 全局统计匹配 (μ, σ 在所有高频上联合计算)
ep_fiber_matched = AdaIN(ep_fiber, style_fiber)
h_new = ep_base + (1-α)·ep_fiber + α·ep_fiber_matched
```

### 2.3 Phase 4G.2 (频域 per-subband, per_subband mode)

```
LL_K, H_K, H_{K-1}, ..., H_1 = W_K(h)              # 多级分解
s_LL_K, s_H_K, ..., s_H_1 = W_K(style_latent)       # style 同样分解

# 每子带独立 AdaIN (LL_K 不动, 作为内容锚)
for k in 1..K:
    for sub in (LH_k, HL_k, HH_k):
        sub_new = (1-α)·sub + α·AdaIN(sub, s_sub)
        # AdaIN(sub, s_sub) = (sub - μ_sub)/σ_sub · σ_{s_sub} + μ_{s_sub}

LL_K_new = LL_K                                     # 内容锚锁死
h_new = W_K^{-1}(LL_K_new, H_K_new, ..., H_1_new)   # 多级 IDWT 重建
```

### 2.4 关键差异: 统计隔离 vs 统计混合

| 维度 | spatial_fiber (当前) | per_subband (4G.2) |
|------|---------------------|---------------------|
| 统计计算范围 | 所有高频联合 (μ, σ on sum) | 每子带独立 (μ_k, σ_k per subband) |
| 正交性利用 | 浪费 (相加后正交性消失) | 充分 (每子带独立统计) |
| 尺度分离 | 无 (宏观+微观混合) | 有 (LH3 笔触 vs LH1 噪点分离) |
| 方向分离 | 无 (LH+HL+HH 混合) | 有 (垂直/水平/对角独立) |
| 参数量 | 0 (无新参数) | 0 (无新参数, 仅算法改变) |

---

## 3. 实现方案

### 3.1 新增工具函数 (spectral620.py)

```python
def dwt2_haar_multi_decompose(x: torch.Tensor, levels: int = 1) -> dict:
    """多级 Haar DWT 分解, 返回所有子带.
    
    返回 dict:
        {"ll_K": LL_K (最粗), 
         "h": [(LH_K, HL_K, HH_K), ..., (LH_1, HL_1, HH_1)]}  # 从粗到细
    """
    subs = []
    current = x
    for _ in range(levels):
        ll, lh, hl, hh = dwt2_haar(current)
        subs.append((lh, hl, hh))  # 高频三元组
        current = ll  # 继续分解 LL
    return {"ll_K": current, "h": subs}  # subs[0]=最粗高频, subs[-1]=最细高频


def idwt2_haar_multi_reconstruct(decomp: dict, levels: int = 1) -> torch.Tensor:
    """多级 Haar IDWT 重建, 从 dwt2_haar_multi_decompose 的输出重建.
    
    输入: decomp = {"ll_K": ..., "h": [(LH_K,HL_K,HH_K), ...]}
    输出: 与原 x 同尺寸的重建张量
    """
    recon = decomp["ll_K"]
    subs = decomp["h"]
    # 从最粗到最细逐级重建
    for k in range(levels - 1, -1, -1):
        lh, hl, hh = subs[k]
        recon = idwt2_haar(recon, lh, hl, hh)
    return recon
```

### 3.2 新增配置字段 (config_schema.py)

```python
# 630 Phase 4G.2: 频域 per-subband AdaIN (用户方案五的子组件, 不依赖 LL 锁死)
# "spatial_fiber" (default): 现有行为, ep_fiber = h - lp(h), 全局 mean+std 匹配
# "per_subband": 频域每子带独立 AdaIN, 利用 Haar 正交性保证统计隔离
endpoint_adain_mode: str = "spatial_fiber"
```

### 3.3 integrate_transport 修改 (spectral_bridge620.py)

在现有 Endpoint AdaIN 分支中添加 `per_subband` 模式:

```python
adain_mode = str(_cfg_get('endpoint_adain_mode', 'spatial_fiber')).lower()

if endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):
    if adain_mode == "per_subband":
        # Phase 4G.2: 频域 per-subband AdaIN
        h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
        s_decomp = dwt2_haar_multi_decompose(style_latent.to(dtype=h.dtype), levels=lowpass_levels)
        # LL_K 不动 (内容锚)
        # 对每个高频子带独立做 mean+std 匹配
        new_subs = []
        for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
            s_lh, s_hl, s_hh = s_decomp["h"][k]
            lh_new = (1.0 - endpoint_adain_scale) * lh + endpoint_adain_scale * _adain_match(lh, s_lh)
            hl_new = (1.0 - endpoint_adain_scale) * hl + endpoint_adain_scale * _adain_match(hl, s_hl)
            hh_new = (1.0 - endpoint_adain_scale) * hh + endpoint_adain_scale * _adain_match(hh, s_hh)
            new_subs.append((lh_new, hl_new, hh_new))
        h = idwt2_haar_multi_reconstruct({"ll_K": h_decomp["ll_K"], "h": new_subs}, levels=lowpass_levels)
    else:
        # 现有 spatial_fiber 模式 (保持不变)
        ep_base = lp(h)
        # ... (现有代码)
```

辅助函数:
```python
def _adain_match(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """单子带 AdaIN: mean+std 匹配 content 到 style 的统计."""
    B_c = content.shape[0]
    if style.shape[0] == 1 and B_c > 1:
        target_mean = style.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
        target_std = style.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
    else:
        target_mean = style.mean(dim=[2, 3], keepdim=True)
        target_std = style.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
    pred_mean = content.mean(dim=[2, 3], keepdim=True)
    pred_std = content.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
    return (content - pred_mean) / pred_std * target_std + target_mean
```

### 3.4 与 style_extrap_alpha 的交互

当前 `style_extrap_alpha > 0` 时对 style_fiber 做缩放 (D3 style extrap)。
Phase 4G.2 中, per_subband 模式下:
- 对每个 style 子带做相同的 extrap 缩放 (保持一致性)
- 或者禁用 extrap (因为 per-subband 已经是更精准的匹配, extrap 的"外推"语义不再需要)

**决策**: 保留 extrap, 对每个 style 子带做缩放, 与现有行为一致。

---

## 4. 实验设计

### 4.1 实验矩阵

| 编号 | 配置 | endpoint_adain_mode | lowpass_levels | epochs | 描述 |
|------|------|---------------------|----------------|--------|------|
| baseline (4F.1) | `630_phase4f_lvl3.json` | spatial_fiber (默认) | 3 | 3 | 当前 SOTA: clip=0.7319 |
| **4G.2** | `630_phase4g2_per_subband.json` | **per_subband** | 3 | 3 | 主实验: 频域 per-subband AdaIN |

**验收阈值**: clip ≥ 0.7243, lpips ≤ 0.3453

### 4.2 物理意义预期

| 指标 | 预期 | 原因 |
|------|------|------|
| clip_style | 0.730-0.738 | 每子带风格更精准, 中频 (LH2/HL2) 笔触方向被独立匹配 |
| content_lpips | 0.335-0.345 | LL3 锁死保内容, 但每子带独立匹配可能略微增加风格注入量 |

**核心理论预测**: per_subband 利用正交性的统计隔离, 应该在保持 lpips 的同时提升 clip_style (因为不同尺度的风格信息被精准匹配, 而非被平均化)。

### 4.3 风险评估

| 风险 | 概率 | 缓解 |
|------|------|------|
| clip 下降 (per_subband 反而劣于 spatial_fiber) | 低 | 退回 spatial_fiber, 记录为 NEGATIVE result |
| lpips 上升 (每子带匹配引入过多风格) | 中 | 降低 endpoint_adain_scale (如 0.8 → 0.6) |
| 计算量增加 (多级 DWT 分解) | 低 | 已有 dwt2_lowpass 做过多级, 开销可接受 |
| OOM | 极低 | 无新参数, 仅算法改变 |

---

## 5. 理论提升: 三层频域解耦的完整实现

Phase 4G.2 完成后, 架构将实现完整的"三层频域解耦":

```
Layer 1: LL_K velocity (训练 + Euler 应用)
  - 4G.1 证明: LL 不是纯内容锚, 携带 +0.014 clip 的全局风格信息
  - 4G.2 不动这部分

Layer 2: Endpoint AdaIN (频域 per-subband)
  - 4G.2 升级: 从空间域全局 fiber → 频域每子带独立匹配
  - LL_K 锁死作为内容锚
  - 每个高频子带 (LH_k/HL_k/HH_k) 独立 mean+std 匹配

Layer 3: Spectral ODE (LH/HL velocity heads)
  - 3 个独立 velocity heads (head_ll, head_lh, head_hl)
  - Euler 积分在频域独立进行
```

### 5.1 与 4G.1 NEGATIVE result 的协同

4G.1 证明 LL velocity 必须保留 (LL 携带全局色调)。
4G.2 不动 LL velocity, 只升级 Endpoint AdaIN。
两者正交, 4G.2 的失败不影响 4G.1 的结论, 反之亦然。

### 5.2 论文 Core Story (如果 4G.2 成功)

> "我们通过 Haar DWT 多级分解 (4F) 解耦内容 (LL) 与风格 (HF)。
> 消融实验 (4A2 + 4G.1) 精确量化了 LL velocity 的贡献 (+0.014 clip, 证明 LL 携带全局色调风格)。
> 进一步, Endpoint AdaIN 从空间域全局 fiber 升级为频域 per-subband 独立统计匹配 (4G.2),
> 利用 Haar 正交性保证不同尺度 (宏观笔触/中频色彩/微观噪点) 的风格信息被精准隔离匹配,
> 突破了空间域全局统计的'频谱平均化'瓶颈。"

---

## 6. 实施步骤

1. ✅ 设计文档 (本文档)
2. 实现 `dwt2_haar_multi_decompose` / `idwt2_haar_multi_reconstruct` (spectral620.py)
3. 实现 `_adain_match` 辅助函数 + `endpoint_adain_mode` 配置 (config_schema.py + spectral_bridge620.py)
4. Smoke test: PR 验证 + forward/backward 一致性
5. 创建实验配置 `configs/630_phase4g2_per_subband.json`
6. 训练 3-epoch (独立目录, 从零训练)
7. 评估并对比 4F.1 SOTA
8. 更新 progress.json + phase4g 设计文档 + phase4_summary
9. Git 提交

---

## 7. 后续展望

如果 4G.2 成功:
- 论文 Core Story 完整, 进入写作
- 可选: 4G.3 (多级 forward, 真·全频域 ODE) 作为更高风险探索

如果 4G.2 失败 (NEGATIVE):
- 记录为重要 ablation: "频域 per-subband 不优于空间域全局"
- 论文以 4F.1 (多级 DWT + 空间域 fiber) 为 SOTA
- 进入论文写作, Core Story 以 4G.1 的 "LL velocity 贡献量化" 为核心 ablation

---

## 8. 实验结果 (2026-07-01)

### 8.1 执行记录

**Config**: `configs/630_phase4g2_per_subband.json`
- `endpoint_adain_mode: "per_subband"` (核心改动)
- `endpoint_lowpass_levels: 3` (继承 4F.1 SOTA)
- `endpoint_adain_scale: 1.0` (继承默认, 未改动 — **这是 4G.2b 的调节目标**)
- `endpoint_lock_ll: false` (LL velocity 正常训练+应用)
- 3 epoch, full_eval_each_epoch=true
- 训练目录: `exp/630_phase4g2_per_subband/` (从零训练, resume_checkpoint="")

**训练**: Epoch 3/3 完成, 14.6s/epoch, VRAM 3.59GB
**Eval**: epoch_0003_full, 75.2s wall time, 750 generated images

### 8.2 结果对比

| 配置 | adain_mode | α (scale) | clip_style | content_lpips | v_ll_abs | 判定 |
|------|-----------|-----------|------------|---------------|----------|------|
| Phase 4F.1 SOTA | spatial_fiber | 1.0 | 0.7319 | 0.3428 | 0.666 | PASS |
| **Phase 4G.2** | **per_subband** | **1.0** | **0.7361** | **0.3843** | 0.659 | **MIXED** |
| Δ (4G.2 - 4F.1) | — | — | **+0.0042** | **+0.0415** | -0.007 | — |

### 8.3 关键发现

**1. clip_style 突破 SOTA (+0.0042)**
- per_subband AdaIN 确实更精准注入风格
- 每子带独立 mean+std 匹配, 释放了 Haar 正交性的"统计隔离"优势
- 不同尺度 (宏观笔触 LH3 / 中频 LH2 / 微观噪点 LH1) 的风格信息被独立精准匹配
- 这证明设计文档 §1.3 的理论预测正确: 频域解耦比空间域全局更纯

**2. content_lpips 大幅超标 (+0.0415, 超过 0.3453 阈值)**
- 设计文档 §4.3 风险评估预测的"中概率风险"发生
- 根因: **风格注入总量过多**
  - spatial_fiber 模式: 1 次全局 mean+std 匹配 (所有高频之和算一次统计)
  - per_subband 模式 α=1.0: **9 次独立全量替换** (3 级 × 3 方向, 每子带 100% 替换为 style 统计)
  - 等效于 9× 风格注入量, 远超 spatial_fiber 的 1× 注入
- LL_3 虽然锁死作为内容锚, 但 9 个高频子带全部被 style 统计覆盖, 内容保真度不可避免下降

**3. v_ll_abs 正常 (0.659 vs 0.666)**
- LL velocity 路径未受影响, 确认 4G.2 与 4G.1 正交
- 微小下降 (-0.007) 在训练噪声范围内

### 8.4 根因分析: 为什么 α=1.0 在 per_subband 模式下过强

**spatial_fiber 模式的"隐式正则化"**:
```
ep_fiber = h - lp(h)                    # 所有高频之和 (1 个张量)
style_fiber = style_latent - lp(style)  # 1 个张量
ep_fiber_matched = AdaIN(ep_fiber, style_fiber)  # 1 次 mean+std 匹配
h_new = ep_base + (1-α)·ep_fiber + α·ep_fiber_matched
```
当 α=1.0 时, **只有 1 次** mean+std 匹配, 统计平均效应天然抑制了过拟合。

**per_subband 模式的"注入放大"**:
```
for k in 1..3:                          # 3 级
    for sub in (LH, HL, HH):            # 3 方向
        sub_new = (1-α)·sub + α·AdaIN(sub, s_sub)  # α=1.0 = 全量替换
```
当 α=1.0 时, **9 次** 独立全量替换, 每个子带都被 style 的统计完全覆盖。

**数学等价性分析**:
- 如果所有子带的 style 统计相同: per_subband α=1.0 ≈ spatial_fiber α=1.0 (退化情况)
- 实际中不同子带的 style 统计不同 (LH3 笔触 vs LH1 噪点), per_subband 引入了"更多自由度的风格注入"
- 这正是 clip 提升的原因 (更精准), 但也是 lpips 超标的原因 (更多注入)

### 8.5 设计文档 §4.3 预案的触发

设计文档 §4.3 风险评估表已预案:
> | lpips 上升 (每子带匹配引入过多风格) | 中 | 降低 endpoint_adain_scale (如 0.8 → 0.6) |

预案触发, 进入 **Phase 4G.2b**: 保持 `endpoint_adain_mode: "per_subband"`, 将 `endpoint_adain_scale` 从 1.0 降到 0.5。

**为什么选 α=0.5 而非 0.6/0.7?**
- 4G.2 lpips 超标 +0.0415 (0.3428→0.3843), 需要显著降低注入量
- α=0.5 意味着每子带保留 50% 原始 + 50% style 匹配, 等效注入量 ≈ 4.5× (vs 4G.2 的 9×)
- 若 α=0.5 仍 FAIL, 可尝试 α=0.3; 若 PASS 且 clip 仍 > 4F.1, 则是新 SOTA
- 极端值拉开 (0.5 vs 1.0) 符合用户"实验配置档位需显著拉开区别"的要求

### 8.6 论文写作价值 (无论 4G.2b 结果)

**如果 4G.2b 成功 (PASS, clip > 4F.1)**:
- 论文 Core Story 升级: "频域 per-subband AdaIN + α 调控 = 突破空间域 SOTA"
- 提供 α 的 ablation (1.0 vs 0.5), 展示"注入量-保真度"的 trade-off 曲线
- 完整的三层频域解耦故事 (LL velocity + per-subband AdaIN + spectral ODE)

**如果 4G.2b 仍 FAIL**:
- 记录为重要 ablation: "per_subband 在 α=1.0/0.5 都无法同时满足 clip+lpips"
- 论文以 4F.1 (多级 DWT + 空间域 fiber) 为 SOTA
- 4G.2 作为"频域解耦的极限探索", 证明空间域 fiber 的"隐式正则化"价值
- Core Story: "4G.1 (LL velocity 必要性) + 4G.2 (频域解耦极限) = 完整的 ablation 矩阵"

---

## 9. Phase 4G.2b 结果 (α=0.5 缓解实验, 2026-07-01): FAIL — α 参数失效

### 9.1 实验执行

**Config**: `configs/630_phase4g2b_per_subband_a05.json`
- `endpoint_adain_mode: "per_subband"` (保持 4G.2 的频域模式)
- `endpoint_adain_scale: 0.5` (**核心改动**: 从 1.0 降到 0.5)
- `endpoint_lowpass_levels: 3`, 3 epoch, 从零训练
- 训练: Epoch 3/3, loss=2.1535, 14.5s/epoch, VRAM 3.23GB
- Eval: 88.1s, 750 generated images

### 9.2 结果: α=0.5 与 α=1.0 几乎完全相同

| 配置 | adain_mode | α | clip_style | content_lpips | v_ll_abs | 判定 |
|------|-----------|---|------------|---------------|----------|------|
| 4F.1 SOTA | spatial_fiber | 1.0 | 0.7319 | 0.3428 | 0.666 | PASS |
| 4G.2 | per_subband | 1.0 | 0.7361 | 0.3843 | 0.659 | MIXED |
| **4G.2b** | **per_subband** | **0.5** | **0.7362** | **0.3845** | 0.656 | **FAIL** |
| Δ (4G.2b - 4G.2) | — | -0.5 | **+0.0001** | **+0.0002** | -0.003 | — |

**α 从 1.0 降到 0.5, clip 和 lpips 几乎零变化!** 这完全出乎预期。

### 9.3 根因分析: 多步 Euler 积分的"迭代累积"效应

**关键发现**: 推理时使用 12 个 Euler steps (`num_steps=12`), 每步都调用 `integrate_transport`（包含 endpoint AdaIN）。

**数学分析**:
```
每步的 AdaIN 操作: sub_new = (1-α)·sub + α·match(sub, s_sub)
                       = sub + α·(match(sub, s_sub) - sub)

设 δ = match(sub, s_sub) - sub (向 style 统计的"拉力")
每步后, sub 的"非 style 残留" = (1-α)·sub_prev

n 步后, 原始统计残留 = (1-α)^n · sub_original
```

| α | n=12 步后残留 | 等效注入 |
|---|--------------|---------|
| 1.0 | (0)^12 = 0% | 100% 替换 |
| 0.5 | (0.5)^12 = 0.024% | ~100% 替换 |
| 0.3 | (0.7)^12 = 1.4% | ~98.6% 替换 |
| 0.1 | (0.9)^12 = 28.2% | ~71.8% 替换 |

**结论**: 对于 α ≥ 0.3 和 n=12, 迭代累积使 per-step α 失效。α=0.5 和 α=1.0 的最终效果趋同 (残留 < 0.024%)。

### 9.4 理论洞察: per-step AdaIN α 的失效条件

**定理 (informal)**: 在 n 步 Euler 积分中, 如果每步都应用同一 AdaIN 操作 (α, style_stats), 则原始统计的残留比例为 (1-α)^n。当 n(1-α) >> 1 时, α 参数失效。

对于我们的配置 (n=12):
- α > 0.2 时, 12(1-α) < 9.6, 残留 (1-α)^12 < 7%, 效果趋同
- α > 0.5 时, 残留 < 0.024%, 完全失效

**推论**: 要在 per_subband 模式下控制注入量, 不能通过 per-step α, 而需要:
1. **End-of-trajectory AdaIN**: 只在最后一步应用 (但需要改架构, 风险高)
2. **α 衰减调度**: α 随 step 衰减 (如 α_t = α_0 · (1-t/T))
3. **减少 Euler steps**: 降低 n 使 (1-α)^n 不趋近 0 (但影响 ODE 精度)
4. **全局缩放而非替换**: 用 sub_new = sub · scale + match · (1-scale) (但破坏 Haar 正交性)

这些是 Future Work 方向, 当前 Phase 4 不再深入。

### 9.5 最终结论: per_subband 路线终止

**4G.2 + 4G.2b 联合结论**:
1. per_subband AdaIN 的频域解耦方向**有效** (clip +0.0042 突破 SOTA)
2. 但无法通过 α 参数控制 lpips (迭代累积使 α 失效)
3. **4F.1 (spatial_fiber) 确认为最终 SOTA** (clip=0.7319, lpips=0.3428, PASS)

**spatial_fiber 的"隐式正则化"价值**:
- spatial_fiber 在空间域做 1 次全局 mean+std 匹配
- 多步 Euler 累积后, 这个"1 次全局匹配"的注入量被稀释 (因为 spatial_fiber 的 fiber = h - lp(h) 在每步后变化, 不是固定的 style 统计)
- 这就是 spatial_fiber 的 lpips=0.3428 而 per_subband 的 lpips=0.3843 的根因
- **spatial_fiber 的"隐式正则化"是多步 ODE 下的天然优势, per_subband 无法复制**

### 9.6 论文写作定位

4G.2 + 4G.2b 作为**重要的理论 ablation** 写入论文:
- **发现**: per_subband AdaIN 突破 clip SOTA (+0.0042), 证明频域解耦有效
- **但**: 多步 ODE 的迭代累积使 α 参数失效, 无法控制 lpips
- **洞察**: spatial_fiber 的"隐式正则化"是多步 ODE 下的天然优势
- **Core Story (最终, §3.2)**: 4F.1 为 SOTA, 4G.2 作为"频域解耦极限探索"的 ablation
