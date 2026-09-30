# FC-SB Phase 3: 风格瓶颈突破的多方向探索 Spec

## Why

**当前状态（客观，5-style 主指标）**：
- H5 baseline: t_clip=0.7026, t_lpips=0.4936
- P3 (N1+eds=0, LPIPS 极致): t_clip=0.6638, t_lpips=0.2658
- 核心矛盾：CLIP vs LPIPS 存在硬 trade-off，eds↓ → LPIPS↓ 但 CLIP↓
- Q 系列 WCT 在 VAE latent (C=4) 退化为 AdaIN（协方差接近对角，已验证）

**问题**：N1 AdaIN 的 per-channel 一阶统计匹配丢失空间信息，无法捕捉 CLIP 关心的笔触/纹理/构图。需要从多个理论角度探索突破 CLIP 瓶颈的方向，而非继续在 eds 参数上做 trade-off。

**目标**：在保持 P3 的 LPIPS 优势（t_lpips < 0.30）的同时，恢复或提升 5-style CLIP（t_clip ≥ 0.70，理想 > 0.72）。

## What Changes

### 工程约束（硬性）
- **显存控制**：训练/推理评估显存控制在 9-11G（RTX 3060 12GB，留 1-2G 安全边际）
- **算力复用**：style 统计 cache、fiber lowpass 结果复用、避免重复 forward
- **统一评估协议（三阶段，避免参数设置不当干扰方法评定）**：
  - **阶段 A：参数搜索** — 每个方向的关键参数做 4-6 候选值的小规模快速训练（1-2 epoch），5-style 评估，找最佳参数点
  - **阶段 B：最佳点训练** — 每个方向用搜索到的最佳参数（1-2 个）训练到收敛（Patience=2, max_epochs=10, 至少 5 epoch）
  - **阶段 C：最终评估** — 训练后 best checkpoint 做 5-style 完整评估，给该方法结论
  - 训练数据集：5-style 标准训练集（Early_Renaissance/Impressionism/Minimalism/Rococo/Ukiyo_e）
  - 评估数据集：5-style 标准测试集（wikiart_distinct5_samam_512_classview/test）
  - **不使用单 style 筛选**：单 style 与 5-style 行为不一致，已验证
- **配置生成**：从 base_focused.json 脚本生成，避免碎文件传输

### 前置方向 I: 初始化策略探索（必先于 R/T/U/V/W 执行）

**为什么必须前置**：
- 用户指出"我们之前做过0初始化"，初始化策略直接影响 style 信号路径的起始强度
- 当前代码 [blocks620.py:170](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py#L170) 中 `film_init_std` 硬编码为 0.02（死代码 `getattr(self, "film_init_std", 0.02)` 永远返回 0.02），无法通过 config 探索
- 在初始化策略未定的情况下跑 R/T/U/V/W，等于在不确定的基线上做方向对比，结论不可靠

**理论分析**（FiLM: `x' = (1+γ(s))·x + β(s)`，`γ = W_γ @ s`）：
- **zero-init (std=0.0)**: `W_γ=W_β=0` → `γ=β=0` → FiLM=identity。梯度 `dL/dW_γ = (dL/dx')·s` 非零（链式法则），模型**可以学习**，但初始时所有 style 产生相同 `x'`，模型看不到 style 差异 → 学习慢，可能陷入"忽略 style"吸引子
- **small random (std=0.02)**: `γ(s_1) ≠ γ(s_2)` → 模型立即看到 style 差异 → 打破"条件期望坍缩"平衡 → 当前默认
- **strong random (std=0.1+)**: 更强 style 信号，但可能引入噪声干扰 content

**代码改动**（已完成）:
- [config_schema.py:315](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py#L315): 新增 `style_film_init_std: float = 0.02` 字段
- [blocks620.py:97](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py#L97): `__init__` 添加 `film_init_std: float = 0.02` 参数，0.0 分支走 zero-init
- [model620.py:188](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L188): 从 `model_cfg` 读取 `style_film_init_std` 传入 block

**L9 正交设计（3 维 × 3 水平 = 9 组）**:

| 维度 | 水平 1 | 水平 2 | 水平 3 |
|------|--------|--------|--------|
| `style_film_init_std` | 0.0 (zero-init) | 0.02 (当前) | 0.1 (强打破) |
| `style_embed_scale` (M3) | 1.0 | 2.0 | 4.0 |
| `endpoint_delta_scale` (M4) | 0.5 | 1.0 | 2.0 |

固定: `gate_mode=fixed_one`、`lr=5e-5`、`gate_warmup=0`（去除干扰变量）

L9 正交表（每组只改 3 个变量到指定水平，其他保持 H5_BASE）:
| 组 | film_init_std | embed_scale | delta_scale |
|----|---------------|-------------|-------------|
| I1 | 0.0 | 1.0 | 0.5 |
| I2 | 0.0 | 2.0 | 1.0 |
| I3 | 0.0 | 4.0 | 2.0 |
| I4 | 0.02 | 1.0 | 1.0 |
| I5 | 0.02 | 2.0 | 2.0 |
| I6 | 0.02 | 4.0 | 0.5 |
| I7 | 0.1 | 1.0 | 2.0 |
| I8 | 0.1 | 2.0 | 0.5 |
| I9 | 0.1 | 4.0 | 1.0 |

**评估加 probe 指标**: 除 t_clip / t_lpips 外，提取
- `style_gate_value`（gate 强度）
- `cross_attn_delta_abs`（cross-attn 输出强度）
- `film_gamma_abs / film_beta_abs`（FiLM 调制强度）
- `cos_sim(v(style_1), v(style_2))`（条件期望坍缩指标）

**输出**: 最佳初始化策略（film_init_std, embed_scale, delta_scale 组合），用于后续 R/T/U/V/W 各方向的 config 基线

### 四个探索方向（理论驱动）

#### 方向 R: Fiber-CFG（N1 基础上的纤维空间 CFG 外推）
**数学理论**：
标准 CFG: `v_cfg = v_uncond + (1+s)·(v_cond - v_uncond)`
Fiber-CFG: 仅在 fiber 分量外推
`v_fiber_cfg = v_fiber_uncond + (1+s)·(v_fiber_cond - v_fiber_uncond)`

**为什么之前 K1 失败但现在可能成功**：
- K1 时期（无 N1）：fiber 无风格方向（runtime_observability: gate=0.05, delta=0.038），CFG 外推无方向信号 = 放大噪声
- 现在（N1 之后）：AdaIN 注入了风格统计到 fiber，fiber 获得方向，CFG 外推有方向信号 = 放大风格方向
- **关键假设**：N1 + CFG 组合下，CFG 能放大 N1 注入的风格方向

**显存策略**：双 forward（cond + null_style）峰值显存高，采用**分两次 forward**（非并行 batch），避免 OOM。style_latent 统计 cache 复用。

#### 方向 T: Multi-band Per-frequency AdaIN（多频段独立统计匹配）
**数学理论**：
当前：wavelet 一级分解 → LL (base, 锁死) + HH (fiber, 统一 AdaIN)
改进：Haar 二级分解 → LL (结构) + Mid (LH+HL, 粗纹理) + HH (细纹理)
- LL: base locking（不动）
- Mid: 中等 adain_scale（匹配粗笔触统计）
- HH: 强 adain_scale（匹配细纹理统计）

**为什么能突破 CLIP**：
- CLIP 关注多尺度纹理（笔触有粗有细）
- 不同频段的风格统计分布不同，统一 adain 会过度匹配某频段、欠匹配另一频段
- 分频段匹配保留多尺度风格特征 → 多尺度纹理一致性 → CLIP↑

**显存策略**：wavelet 分解轻量（O(HW)），无额外 forward，显存友好。

#### 方向 U: Style Latent Extrapolation（风格潜变量外推）
**数学理论**：
当前：`style_fiber = style_latent - lp(style_latent)` 直接用
改进：外推 style_latent 到更极端
`style_ext = style_latent + α·(style_latent - μ_dataset_fiber)`
`style_fiber_ext = style_ext - lp(style_ext)`

**理论依据**：
- StyleGAN truncation trick 的反向应用：推向更极端风格（而非更保守）
- 类似 CFG 但在 style 空间而非速度空间，不需要双 forward
- 数学等价于在 style 潜空间沿"风格方向"外推

**显存策略**：仅需预先计算 μ_dataset_fiber（一次性 cache），无额外 forward，显存最友好。

**风险**：外推过度可能产生伪影；需探测 α 的合理范围。

#### 方向 V: Spatial Patch AdaIN（空间分块统计匹配）
**数学理论**：
当前 AdaIN: per-channel 全局 (μ, σ)，空间不变 → 丢失空间局部风格特征
Patch AdaIN: 将 fiber 分 patch，per-patch 独立匹配统计
`unfold(fiber, k=8) → per-patch (μ_p, σ_p) → match to style patch stats → fold`

**为什么能突破 CLIP**：
- CLIP 是 ViT，在 patch 级别提取特征，衡量空间分布相似度
- 全局 AdaIN 使空间分布均匀化，丢失局部笔触方向
- Patch AdaIN 保留空间局部风格特征（笔触方向、局部纹理变化）

**显存策略**：unfold 会增加内存（patch 数 = (H/k)·(W/k)），k=8 时 patch 数=64，可控。style patch 统计预先 cache。

#### 方向 W: 风格排斥 Loss（训练侧，跨风格对抗）
**数学理论**：
当前训练 loss 只含正向风格匹配（SWD: 让生成 z_hat 接近 target style 的统计）。缺乏**负向约束**（让生成 z_hat 远离其他 style 的统计）。

**FC-SB 理论依据**：FC.md 核心命题"底流形死寂，纤维狂热扩散"要求 fiber 携带**风格判别信息**。但当前 fiber 速度无方向（runtime_observability 已证明），模型学到的 fiber 扩散对所有 style 几乎相同（style 门 0.05 关闭）。需通过 loss 强制 fiber 编码 style 判别信息。

**已有基础**：losses620.py L125-127, L600-626 已实现 `w_style_contrastive` — batch 内 pairwise 余弦相似度 margin loss。但这是**隐式排斥**（只让 batch 内样本互相远离，不针对具体 style）。

**W 方向新增三种排斥机制**：

**W1: Cross-style Fiber Repulsion（跨风格 fiber 排斥）**
对 batch 内不同 style 的 fiber 分量做排斥：
$$L_{repel} = \frac{1}{B(B-1)} \sum_{i \neq j} \text{relu}(m - \|f_i^{style} - f_j^{style}\|_2)$$
其中 $f_i^{style}$ 是样本 i 的 fiber 风格特征。强制不同 style 的 fiber 在特征空间分离。

**W2: Anti-input-style Repulsion（输入风格排斥）**
让生成结果远离**输入内容自身的风格**（避免模型懒惰地保留输入风格）：
$$L_{anti\_input} = \text{relu}(m - \|f^{gen} - f^{content\_input}\|_2)$$
强制生成 fiber 与输入 content 的 fiber 统计有距离。

**W3: Style Discriminative Loss（风格判别损失）**
在 fiber 上加一个轻量 style classifier（线性头），强制 fiber 能判别 style：
$$L_{disc} = \text{CrossEntropy}(classifier(f^{gen}), style\_id)$$
这迫使 fiber 编码 style 判别信息，而非无方向噪声。

**为什么能突破 CLIP**：
- CLIP 衡量"生成图与风格参考图的语义相似度"
- 当前模型 fiber 无 style 判别力 → 生成图的"风格特征"是模糊的统计匹配，非真正风格
- W 方向强制 fiber 编码 style 判别信息 → 生成图具有明确风格身份 → CLIP↑

**训练显存策略**：
- W1/W2: 仅增加 pairwise 距离计算（O(B²)），batch=24 时 576 pair，可忽略
- W3: 轻量 classifier（C→num_styles 线性层），参数量 < 100，可忽略
- **训练配置**：batch_size=24（12GB VRAM 安全，已验证），2-3 epoch，LR=5e-5
- **算力复用**：fiber 分量已在 forward 中计算，loss 复用，无需额外 forward

**风险**：
- W1/W2 margin 过大会导致生成伪影（过度远离导致偏离合理风格空间）
- W3 classifier 过强会占用模型容量，需小 weight（0.1-0.5）
- 需要探测 weight 的合理范围

## Impact

- **Affected specs**: `fc-sb-breakthrough`（已完成，本 spec 是其 Phase 3 延续）
- **Affected code**:
  - `src/model620.py` — `i2sb_inference()` 中 N1 AdaIN 块（L635-717），新增 R/T/U/V 模式分支
  - `src/config_schema.py` — 新增配置字段（r_cfg_scale, t_multiband, u_style_extrap_alpha, v_patch_kernel 等）
  - `exp/625_fc_sb/` — 新增 R/T/U/V 系列变体生成与评估脚本
- **Affected docs**: `docs/625_fc_sb/EXPERIMENT_LOG.md` — 记录四方向探索结果

## ADDED Requirements

### Requirement: 显存预算与算力复用
系统 SHALL 在所有新方向（R/T/U/V）的推理评估中控制峰值显存 ≤ 11GB。
- **WHEN** 执行 5-style 评估（batch_size=1, num_steps=12）
- **THEN** `torch.cuda.max_memory_allocated() < 11GB`
- 方向 R（双 forward）SHALL 采用分次 forward 而非并行 batch
- 方向 V（patch unfold）SHALL 限制 patch 数 ≤ 256（k ≥ 8 时满足）
- 所有方向 SHALL 复用 style 统计 cache，避免重复计算

### Requirement: 5-Style 三阶段统一评估（参数搜索 → 最佳点训练 → 最终评估）
系统 SHALL 对每个方向（R/T/U/V/W）执行三阶段评估，避免参数设置不当干扰方法评定。
- **阶段 A（参数搜索）**：每个方向的关键参数做 4-6 候选值，每个候选做小规模快速训练（1-2 epoch）+ 5-style 评估，找最佳参数点
- **阶段 B（最佳点训练）**：每个方向用搜索到的最佳参数（1-2 个）训练到收敛（Patience=2, max_epochs=10, 至少 5 epoch）
- **阶段 C（最终评估）**：训练后 best checkpoint 做 5-style 完整评估，给该方法结论
- **WHEN** 任何方向需要给出性能结论
- **THEN** MUST 完成三阶段流程后才可写入 EXPERIMENT_LOG.md 结论
- 训练数据集：5-style 标准训练集
- 评估数据集：wikiart_distinct5_samam_512_classview/test
- 禁止单 style 筛选后直接给结论
- 禁止跳过参数搜索直接用任意参数给结论

### Requirement: 方向 R — Fiber-CFG
系统 SHALL 支持 `fiber_cfg_scale` 配置，在 N1 AdaIN 激活后对 fiber 速度做 CFG 外推。
- **WHEN** `fiber_cfg_scale > 0` 且 `endpoint_adain_scale > 0`
- **THEN** 推理时执行双 forward（cond + null_style），在 fiber 速度上做 `(1+s)·v_fiber_cond - s·v_fiber_uncond`
- **验收**: fiber_cfg_scale=0 时行为与 N1 完全一致（向后兼容）

### Requirement: 方向 T — Multi-band AdaIN
系统 SHALL 支持 `multiband_adain_mode` 配置，对 fiber 做 Haar 二级分解后分频段 AdaIN。
- **WHEN** `multiband_adain_mode = "two_level"`
- **THEN** fiber 分解为 Mid (LH+HL) 和 HH，各自独立匹配 style 对应频段统计
- **验收**: mid_scale=hh_scale 时行为与单 band AdaIN 一致（退化验证）

### Requirement: 方向 U — Style Latent Extrapolation
系统 SHALL 支持 `style_extrap_alpha` 配置，对 style_latent 做外推后再提取 fiber 统计。
- **WHEN** `style_extrap_alpha > 0`
- **THEN** `style_ext = style_latent + α·(style_latent - μ_dataset_fiber_cache)`
- **验收**: α=0 时行为与 N1 完全一致

### Requirement: 方向 V — Spatial Patch AdaIN
系统 SHALL 支持 `patch_adain_kernel` 配置，对 fiber 做空间分块统计匹配。
- **WHEN** `patch_adain_kernel > 0`
- **THEN** fiber 按 k×k patch unfold，per-patch 匹配 style patch 统计
- **验收**: kernel ≥ latent_H 时退化为全局 AdaIN（一致性验证）

## MODIFIED Requirements

### Requirement: N1 Endpoint AdaIN（扩展模式）
原 N1 支持 full/mean_only/std_only/wct/wct_diag 模式。新增 R/T/U/V 作为**正交增强**（可与任意 base 模式组合）：
- R 是 fiber 速度层增强（独立于 AdaIN 模式）
- T 是 AdaIN 分频段增强（替换原统一 AdaIN）
- U 是 style 源信号增强（在 AdaIN 之前）
- V 是 AdaIN 空间分块增强（替换原全局 AdaIN）

## REMOVED Requirements
无（所有新方向向后兼容，默认配置行为不变）
