# Semantic SWD 理论分析

## 1. 背景与问题定义

### 1.1 SWD 在 FC-SB 中的角色

Sliced Wasserstein Distance (SWD) 是 FC-SB (Flow-matching Conditional Schrödinger Bridge) 中的端点分布约束损失。其作用是让模型预测的端点 `z_hat1` 的分布匹配目标 `projected_target` 的分布，而非仅做逐点匹配。

```python
# 基础 SWD: 全局边缘分布匹配
proj_a = a_spatial @ dirs.t()          # [B, N, P] 随机投影
proj_b = b_spatial @ dirs.t()
swd = (sort(proj_a) - sort(proj_b)).abs().mean()  # 分位数 L1 距离
```

**关键洞察**: MUSIQ（无参考图像质量指标）奖励纹理自然度和锐度。SWD 通过匹配 reference artwork 的高频统计来驱动 MUSIQ — 丢弃 SWD 会导致 MUSIQ 从 41.11 降到 35.31（详见 `results_musiq_sweep.md`）。

### 1.2 "Semantic" SWD 的目标

用户的核心诉求: **"让内容相近的块匹配"** — 即语义 coherent 的区域内部做分布匹配，而非全局混合。

直觉上这很合理: 一张风景画的天空（平滑）和树冠（纹理丰富）如果共享一个全局边缘分布匹配，天空会被迫"沾染"树冠的高频统计，产生 muddy blend。

## 2. k-means 区域划分路线的失败（S1-S4）

### 2.1 实验结果

| 方向 | 机制 | MUSIQ | CLIP-S | LPIPS | 结论 |
|------|------|-------|--------|-------|------|
| Baseline | global + attention-weighted | 41.11 | 0.7275 | 0.4347 | 基线 |
| S1 | k-means region (8区域, blend=0.7) | 41.59 | 0.7245 | 0.5067 | MUSIQ微升, LPIPS恶化 |
| S2 | k-means + multi-patch | 38.89 | 0.7047 | 0.5311 | MUSIQ下降, 全面恶化 |
| S3 | k-means per DWT subband | 40.92 | 0.6976 | 0.4240 | CLIP-S下降 |
| S4 | cross-attn guided + k-means | 42.01 | 0.7095 | 0.5320 | MUSIQ最高但LPIPS失控 |

### 2.2 根因分析: k-means 路线为何失败

**根因 1: content latent 不是语义特征**

k-means 在 `content`（VAE 编码的内容图 latent）上聚类。但 VAE latent 编码的是低级像素信息，不是真正的语义。同一张图的天空区域和水面区域在 latent 空间可能距离很近（都是平滑区域），但它们的"正确"风格转移方式完全不同。

**根因 2: 区域匹配破坏全局分布约束**

global SWD 已经在驱动 MUSIQ — 它让 `z_hat1` 的全局边缘分布匹配 `projected_target`。这个全局匹配包含了 reference artwork 的纹理统计，正是 MUSIQ 奖励的。

k-means 区域内匹配将这个全局约束拆成 K 个局部约束，每个局部约束的样本量只有全局的 1/K。这:
- 削弱了全局统计信号（MUSIQ 的驱动力）
- 降低了 SWD 的统计效力（样本量减少 → 分位数估计噪声增大）
- 引入区域间的不一致（每个区域独立匹配，边界处产生伪影）

**根因 3: 训练不稳定**

k-means 每次 forward 重新聚类，聚类结果对初始化敏感。即使用了 deterministic seeding（按 norm 排序），content latent 的微小变化也会导致区域划分跳变，使训练信号噪声化。

**根因 4: 区域对齐的伪问题**

`_semantic_region_swd` 尝试将 gen 的区域和 target 的区域"对齐"（按 centroid mean-projection 排序）。但 gen 和 target 是不同的图，它们的区域语义未必对应。强行对齐会扭曲内容。

### 2.3 S4 的启示: guidance signal 有效，区域划分无效

S4（cross-attn guided + k-means）取得了最高 MUSIQ (42.01)，证明 **cross-attn entropy 作为 guidance signal 有效** — 它告诉 SWD "哪里需要风格转移"。

但 S4 的 LPIPS (0.5320) 远超基线 (0.4347) 和目标 (0.48)，证明 **k-means 区域划分仍然在伤害内容保真度**。

结论: 正确的方向是 **保留 guidance signal，去掉区域划分**。

## 3. Semantic SWD 的正确机制: Guidance-based 而非 Region-based

### 3.1 核心区分

| 路线 | 机制 | 问题 |
|------|------|------|
| Region-based (S1-S4) | 划分区域 → 区域内独立匹配 | 破坏全局约束, 样本不足, 训练不稳定 |
| **Guidance-based (S5-S8)** | 全局匹配 → guidance 调整采样权重 | 保留全局约束, 聚焦关键区域 |

### 3.2 Guidance-based Semantic SWD 的数学形式

标准 SWD 对所有空间位置均匀采样:
```
SWD_global = E_dir [ (1/N) Σ_i | sort(proj_a)_i - sort(proj_b)_i | ]
```

Guidance-based SWD 用 semantic signal `w(x,y)` 调整采样概率:
```
p(x,y) = w(x,y) / Σ w(x,y)          # 归一化为概率分布
SWD_guided = E_dir [ (1/M) Σ_j | sort(proj_a[~p])_j - sort(proj_b[~p])_j | ]
```

其中 `~p` 表示按概率 `p` 采样 M 个位置。

**关键区别**: 这不是区域划分后独立匹配，而是全局匹配中的重要性采样。全局分布约束保留，但 SWD 的"注意力"集中在 semantic 重要的区域。

### 3.3 三种 Guidance Signal

#### 3.3.1 Cross-attention Entropy (S4, S5, S6)

```python
# model.last_pixel_entropy: [B, 1, H, W]
# 交叉注意力路由模块的像素熵 — 哪里在"编辑"内容
weight = model.last_pixel_entropy
weight = weight / weight.mean()  # 归一化到均值1
```

**语义**: 交叉注意力熵高的区域 = 模型正在做风格转移的区域 = 需要 SWD 约束的区域。

**验证**: S4 MUSIQ=42.01 > baseline 41.11，证明此 signal 有效。

#### 3.3.2 DWT High-frequency Energy (S7)

```python
# content 的 DWT 高频能量: |LH| + |HL| + |HH|
_, lh, hl, hh = dwt2_haar(content)
energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)
weight = energy / energy.mean()  # 归一化
```

**语义**: 高频能量高的区域 = 纹理丰富的区域 = MUSIQ 直接奖励的区域。

**假设**: MUSIQ 奖励纹理自然度，因此 SWD 应该在纹理丰富的区域更严格地匹配。平滑区域（天空、墙面）的低频统计已经由 FM loss 约束，不需要 SWD 额外约束。

**优势**: 完全 content-adaptive，不依赖模型内部状态，训练初期就有效（cross-attn entropy 需要模型训练后才有意义）。

#### 3.3.3 Combined: Cross-attn × DWT (S8)

```python
weight = (dwt_weight * attn_weight)  # 逐元素乘积
weight = weight / weight.mean()       # 重归一化
```

**语义**: 既需要风格转移（cross-attn）又有纹理细节（DWT）的区域 = 风格转移的"关键战场"。

**假设**: 两个 signal 互补 — cross-attn 标记"哪里转移"，DWT 标记"哪里有纹理"。乘积聚焦在两者的交集。

### 3.4 频段分解 + Guidance 的协同 (S5)

S5 将 DWT 频段分解与 cross-attn guidance 结合:

```python
for band, weight_b in [(LL, 0.25), (LH, 1.0), (HL, 1.0), (HH, 2.0)]:
    swd += weight_b * SWD(gen_band, target_band, sample_weight=guidance)
```

**理论**: 
- LL (低频): 结构/颜色，FM loss 已约束，SWD 权重低 (0.25)
- LH/HL (中频): 纹理方向，SWD 权重高 (1.0)
- HH (高频): 细节/锐度，MUSIQ 最敏感，SWD 权重最高 (2.0)
- Guidance: 在每个频段内，cross-attn guidance 聚焦在编辑区域

### 3.5 多尺度 Patch + Guidance 的协同 (S6)

S6 将多尺度 patch 纹理匹配与 cross-attn guidance 结合:

```python
for patch_size, weight_p in [(1, 0.3), (3, 0.4), (5, 0.3)]:
    swd += weight_p * patch_SWD(gen, target, patch=patch_size, sample_weight=guidance)
```

**理论**:
- patch=1: 像素级颜色边缘分布（baseline SWD）
- patch=3: 细粒度纹理（笔触、颗粒）
- patch=5: 粗粒度纹理（色块、结构）
- Guidance: 在每个尺度内，聚焦在编辑区域的局部纹理匹配

## 4. 实验设计

### 4.1 控制变量

所有实验固定:
- batch_size=48, num_epochs=10, lr=2e-4
- cross_attn_dwt_route=true (DWT 路由架构)
- single_step_swd_weight=8.0
- 5-style distinct5 数据集
- RTX 3060 12GB, VRAM < 11GB

### 4.2 变量

| 实验 | swd_scale_mode | swd_guidance_source | swd_band_mode | swd_patch_mode |
|------|---------------|--------------------|--------------|---------------|
| S5 | cross-attn-guided | style_delta (default) | split | off |
| S6 | cross-attn-guided | style_delta (default) | off | multi |
| S7 | cross-attn-guided | dwt_energy | off | off |
| S8 | cross-attn-guided | cross_attn_plus_dwt | off | off |

### 4.3 评估指标

- **MUSIQ**: 主要目标（推高）
- **CLIP-S**: 风格相似度（保持 > 0.70）
- **LPIPS**: 内容保真度（控制 < 0.48，参考 Seedream 0.4767）

### 4.4 Trade-off 参考

Seedream 4.5 在 D5-512: CLIP-S=0.7198, LPIPS=0.4767, MUSIQ=69.51
→ 接受 LPIPS 到 ~0.48，换取 MUSIQ 提升

## 5. 预期与风险

### 5.1 预期

- S5 (band+guidance): MUSIQ 应高于 baseline，因为 HF 频段被重点约束
- S6 (patch+guidance): MUSIQ 应高于 baseline，因为多尺度纹理匹配
- S7 (DWT-energy): 如果 MUSIQ 提升，证明 content-adaptive guidance 不依赖模型内部状态
- S8 (combined): 如果两个 signal 互补，应该有最高的 MUSIQ

### 5.2 风险

- DWT-energy guidance 可能在内容图本身纹理丰富时退化（所有区域权重相近 → 退化为 global SWD）
- Combined guidance 的乘积可能过度聚焦，导致大部分区域无约束
- band-split 和 multi-patch 的计算开销可能影响训练速度

## 6. 代码路径

- 理论实现: `src/spectral_losses620.py`
  - `_sliced_wasserstein()`: 基础 SWD + 加权采样
  - `_patch_swd()`: 多尺度 patch SWD
  - `_dwt_energy_weight()`: DWT 能量 guidance (新增)
  - `_cross_attn_swd_weight()`: 统一 guidance 入口 (扩展)
  - `_compute_swd()`: SWD 分发器
- 配置: `configs/musiq_s{5,6,7,8}_*.json`
- 评估: `task_musiq/eval_only.py` + `scripts/_compute_musiq_batch.py`
