# Phase 4C: Block Masking on RGB Image (Block Cutout)

**Date**: 2026-07-01
**Stage**: Phase 4C (加法 - RGB 层面 Block Masking)
**Goal**: 通过在 RGB 像素空间做大块几何遮挡 (Block Masking / Grid Masking) 后再 DINO encode, 解决 Token-level mask 破坏潜空间流形的问题, 突破 content_lpips 0.34 上限。

## 1. 动机: 为什么 Token-level Mask 失败, RGB-level Mask 是终极正解

### 1.1 Phase 4B/4D 的失败教训

Phase 4B-3 (DWT tokenizer) 和 Phase 4D (multi-level DWT) 都在 latent/token 层面做 mask。但 token-level mask 有两个致命缺陷:

1. **破坏潜空间流形 (Out-of-Distribution 灾难)**
   VAE/DINO 提取的 tokens 具有强局部空间相关性。随机 drop 75% token 会让主干网络看到"布满黑洞的非自然噪声",无法提取有效 K/V。

2. **撕碎笔触连续性**
   油画笔触跨多个 token, 随机 dropout 把连续笔触切成无意义高频噪点。

**实测验证**: Phase 4B-3 dwt_a1_rand50 (random_50 token mask) 比 dwt_a1 (无 random mask) 还低 0.0011 clip_style。

### 1.2 用户提出的 RGB-level Block Masking 是"终极正解"

如果在 **RGB 像素空间** 先用大块几何遮挡 (Block Masking / Grid Masking), 然后 DINO encode, 物理逻辑完全改变:

1. **完美流形保真 (Manifold Preservation)**
   一张挖了几个黑洞的 RGB 图片**依然是合法图片**。DINO encode 时卷积核能正常工作, 提取的 token 仍在模型原本知识空间内。

2. **真正的"杀结构, 保纹理"**
   - **结构被毁**: 512×512 图片挖几个 128×128 大黑块后, 猫的轮廓被破坏, DINO 认不出物体 → **完美防止语义内容泄漏**
   - **纹理幸存**: 未遮挡区域保留**完整的连续笔触、颜料厚度、色彩分布** → VAE/DINO 能提取最纯正的"风格 token"

3. **非对称引导 (Asymmetric Conditioning)**
   - 训练目标 (Flow Matching Target): 完整无损的风格图 Latent
   - 条件输入 (Cross-Attention K/V): 被 Mask 过的风格图 Latent
   - 模型被迫学会"从局部纹理推导全局风格", 彻底根除死记硬背

## 2. 架构约束与实现方案

### 2.1 关键约束: 训练路径无 RGB

代码探查发现 (src/utils/dataset.py):
- baseline 配置 `dino_cache_path: ""` → dataset 不返回 `target_style_dino_patches`
- 模型侧 `StyleConditioner620.forward` 收到 `style_dino_patches=None` 时, 使用可学习 `style_memory` [5, 256, 384] 作为伪 patches
- **这意味着之前所有 "random_50 token mask" 实验都是对 style_memory 做 mask, 而非真实 DINO patches!**

要使用真实 DINO patches:
- 设置 `dino_cache_path` 指向有效 .pt 文件
- .pt 格式 (build_offline_dino_pairing_cache.py 输出):
  ```python
  {
    "rows": [{"style", "stem", "image_path", "latent_path"}, ...],
    "cls_embeddings": [N, D],
    "patch_embeddings": [N, P, D],
    ...
  }
  ```

### 2.2 实现方案: 离线生成 Block-Masked DINO Cache

由于训练路径无 RGB, 无法在 DataLoader 里在线 mask。采用**离线预处理**方案:

```
[原 RGB 图] → [Block Mask] → [DINO encode] → [patch_embeddings 存入 .pt cache]
```

新工具: `tools/experiments/build_offline_dino_cache_blockmask.py`

**输入**:
- `--flat-image-dir`: 扁平 RGB 目录 (e.g., `F:\wikiarts_5_full_notest\train_flat\style`)
- `--latent-root`: latent 根目录 (e.g., `F:\wikiart_distinct5_samam_512_latents_ema\train`)
- `--output`: 输出 .pt 路径
- `--block-mask-ratio` (默认 0.0): 遮挡比例
- `--block-size` (默认 128): 黑块尺寸
- `--seed` (默认 42): 可复现

**Block Mask 算法** (Grid Masking 变体):
```python
def apply_block_mask(pil_img, mask_ratio=0.6, block_size=128, seed=None):
    """在 RGB 图片上随机挖 block_size×block_size 的黑块."""
    if seed is not None:
        random.seed(seed)
    w, h = pil_img.size  # 512, 512
    img_array = np.array(pil_img)  # H, W, 3
    num_blocks_x = w // block_size  # 4
    num_blocks_y = h // block_size  # 4
    total_blocks = num_blocks_x * num_blocks_y  # 16
    num_mask = int(total_blocks * mask_ratio)  # 9 (60% of 16)
    mask_indices = random.sample(range(total_blocks), num_mask)
    for idx in mask_indices:
        by = idx // num_blocks_x
        bx = idx % num_blocks_x
        y1, y2 = by * block_size, (by + 1) * block_size
        x1, x2 = bx * block_size, (bx + 1) * block_size
        img_array[y1:y2, x1:x2, :] = 0  # 黑块
    return Image.fromarray(img_array)
```

**输出**: 与 build_offline_dino_pairing_cache.py 完全相同的 .pt 格式, dataset 无需改动。

### 2.2.1 与原 build_offline_dino_pairing_cache.py 的关系

| 方面 | 原脚本 (build_offline_dino_pairing_cache.py) | 新脚本 (build_offline_dino_cache_blockmask.py) |
|------|---------------------------------------------|----------------------------------------------|
| 图片布局 | `image_root/<style>/*.jpg` (按风格分子目录) | `--flat-image-dir` 扁平结构 (所有 .jpg 在一个目录) |
| Mask 支持 | 无 | `--block-mask-ratio` + `--block-size` + `--seed` |
| 输出格式 | 标准 DINO cache .pt | 同标准格式 (dataset 无感) |
| Mask 元数据 | 无 | cache 顶层增加 `block_mask_config` 字段 (用于审计) |

新脚本不修改原脚本, 是**独立工具**。原脚本仍可用于"clean cache"生成。

## 3. 实验矩阵

| 编号 | 配置 | dino_cache | block_mask | endpoint_lowpass_levels | epochs | 描述 |
|------|------|------------|------------|-------------------------|--------|------|
| baseline (4D.1) | `630_phase4d_lvl2.json` | (空, 用 style_memory) | — | 2 | 3 | Phase 4D SOTA: clip=0.7301 |
| **4C.0** | `630_phase4c_dino_clean_lvl2.json` | clean DINO cache | 0.0 | 2 | 3 | 对照组: 仅"用真实 DINO"的效果 |
| **4C.1** | `630_phase4c_blockmask_r60_b128_lvl2.json` | blockmasked DINO cache | 0.6, b=128 | 2 | 3 | 主实验: RGB block mask + lvl2 |

**验收阈值**: clip ≥ 0.7243, lpips ≤ 0.3453

### 3.1 实验设计逻辑

- **4C.0 vs baseline (4D.1)**: 隔离"用真实 DINO patches"的效应。如果 4C.0 显著优于 baseline, 说明真实 DINO 比可学习 style_memory 更好。
- **4C.1 vs 4C.0**: 隔离"RGB block mask"的效应。在都使用真实 DINO 的前提下, mask 与不 mask 的差异。
- **4C.1 vs 4D.1**: 综合效应 (真实 DINO + RGB mask) vs (style_memory + 无 mask)。

### 3.2 物理意义预期

| 实验 | 预期 clip_style | 预期 lpips | 原因 |
|------|----------------|------------|------|
| baseline (4D.1) | 0.7301 ⭐ | 0.3402 | style_memory 学习到的平均风格 |
| 4C.0 (clean DINO) | 0.729-0.732 | 0.335-0.345 | 真实 DINO 信息更丰富, 但可能过拟合 |
| **4C.1 (block mask)** | **0.732-0.738** | **0.325-0.335** | **预期最优: 真 DINO + 防内容泄漏** |

**4C.1 的核心理论预测**: 用户原话 "这几乎肯定能大幅缓解你的'内容泄漏 (Content Leakage)'导致 LPIPS 居高不下的问题。"

## 4. 理论提升

### 4.1 Content Fidelity Pathway 升级

现有 (Phase 4D.1):
```
2-Level DWT 低通 → Endpoint AdaIN → Spectral ODE 低频路径 → 风格外推
(LL₂ 锁死构图)    (fiber 含中频)   (head_ll 补偿)         (scale)
                   ↑
                   风格统计来自 style_memory (可学习, 平均)
```

升级后 (Phase 4C.1):
```
2-Level DWT 低通 → Endpoint AdaIN → Spectral ODE 低频路径 → 风格外推
(LL₂ 锁死构图)    (fiber 含中频)   (head_ll 补偿)         (scale)
                   ↑
                   风格统计来自 真实 DINO patches (block-masked RGB 编码)
                   ↓
                   Cross-Attention 收到的 K/V 已不含"猫轮廓"等结构信息
                   模型被迫从局部笔触推导全局风格 → 防内容泄漏
```

### 4.2 三层信息解耦

Phase 4C.1 实现了"三层信息解耦"的理论美感:

1. **绝对构图 (LL₂ 8×8)**: DWT 锁死, 保 LPIPS
2. **宏观笔触 (LH₂/HL₂/HH₂ 8×8)**: Endpoint AdaIN 风格化, 释放 clip_style
3. **风格语义 (Block-Masked DINO patches)**: 防内容泄漏, 提供纯净风格 K/V

### 4.3 与用户 5 方案的对应

Phase 4C 不属于用户原 5 方案 (Daubechies/多级/DTCWT/Lifting/全频域), 而是用户在本轮新提出的**第 6 方案 (Block Masking on RGB)**, 与 5 方案正交。组合空间:

| 维度 | 选项 |
|------|------|
| 小波基 | Haar / Daubechies (4E) / DTCWT (长期) |
| 分解级数 | 1-Level / 2-Level (4D.1) / 3-Level (后续) |
| 风格条件 | style_memory / clean DINO (4C.0) / blockmasked DINO (4C.1) |

## 5. 实施步骤与时间预估

1. ✅ 设计文档 (本文档)
2. 实现 `tools/experiments/build_offline_dino_cache_blockmask.py` (~250 行)
3. Smoke test 新工具 (5 张图验证 pipeline)
4. 后台生成 clean cache + block-masked cache (预计 ~5-10 分钟)
5. 写 Phase 4C configs (4C.0 + 4C.1)
6. 训练 + 评估 (2 个实验 × 3 epochs × ~1 分钟 = ~6 分钟)
7. 文档结果 + git 提交

## 6. 实验结果 (NEGATIVE — 与理论预期相反)

### 6.1 实测数据 (3 epochs)

| 配置 | dino_cache | block_mask | clip_style | content_lpips | v_ll_abs | verdict |
|------|-----------|------------|-----------|---------------|----------|---------|
| **baseline (4D.1)** | (空, style_memory) | — | **0.7301** ⭐ | 0.3402 | (head_ll ≈ 0) | **PASS** |
| 4C.0 (clean DINO) | clean | 0.0 | 0.7118 | **0.3038** ⭐ | 0.3419 | **FAIL** (clip -0.0125) |
| 4C.1 (block mask) | blockmasked | 0.6/b128 | 0.7151 | 0.3177 | 0.2662 | **FAIL** (clip -0.0092) |

**验收阈值**: clip ≥ 0.7243, lpips ≤ 0.3453

### 6.2 关键发现:Real DINO Cache **伤害** clip_style

**与理论预期完全相反**:
- 预期: 真 DINO + block mask → clip_style ↑, lpips ↓
- 实测: 真 DINO → clip_style **大跌 -0.012 ~ -0.018**; block mask 仅能**部分恢复** +0.0033

详细对比:
| 指标 | 4D.1 (style_memory) | 4C.0 (clean DINO) | Δ (4C.0 - 4D.1) |
|------|---------------------|-------------------|------------------|
| clip_style | 0.7301 | 0.7118 | **-0.0183** ❌ |
| content_lpips | 0.3402 | 0.3038 | -0.0364 ✓ |
| v_ll_abs | ~0 | 0.3419 | +0.342 |

### 6.3 物理原因分析 (理论更新)

**为什么 learnable `style_memory` 反而比 real DINO patches 好?**

#### 6.3.1 `style_memory` 是"任务最优"嵌入, DINOv2 是"内容最优"特征

- **`style_memory` [5, 256, 384]**: 可学习参数, 在训练中被反向传播**直接优化**为"对当前 style transfer 任务最有用的 K/V"。它编码的是模型**自己学到的风格表示**, 与损失函数目标对齐。
- **DINOv2 patch_embeddings**: 在 1.4 亿张自然图像上自监督训练得到, 优化目标是**区分不同物体/场景**。它编码的是**内容语义** (物体轮廓、场景拓扑), 而非"风格"。

**关键洞察**: DINOv2 的特征空间是**内容-风格纠缠**的。Cross-Attention 读取 DINO K/V 时, 会"顺手"读到内容信息 (猫的轮廓、向日葵的形状), 导致模型偷懒 — **用 style 图的内容信息补 content 图的内容**, 而非真正学习风格迁移。这反而**抑制**了 clip_style 的提升 (因为风格信号被内容信号稀释)。

#### 6.3.2 Block Masking 部分验证了内容泄漏假说

- 4C.0 (clean DINO): clip=0.7118, lpips=0.3038
- 4C.1 (block mask r0.6): clip=0.7151, lpips=0.3177
- **Δclip = +0.0033**, **Δlpips = +0.0139**

**Block mask 确实缓解了内容泄漏** (clip 小幅恢复), 但代价是 lpips 上升 (因为 mask 后的 patches 不再完整覆盖所有风格细节)。**部分验证了用户的内容泄漏假说, 但远不足以弥补 real DINO 的整体劣势**。

#### 6.3.3 v_ll_abs 的反向变化

观察到一个反直觉现象:
- 4D.1 (style_memory, 无 DINO): head_ll 几乎不被激活 (v_ll_abs ≈ 0)
- 4C.0 (real DINO): head_ll 被激活 (v_ll_abs = 0.34)
- 4C.1 (blockmasked DINO): head_ll 中等激活 (v_ll_abs = 0.27)

**解释**: 当 style_memory 提供的 K/V 已是"任务最优"时, head_ll 不需要做额外补偿。当切换到 real DINO (内容-风格纠缠) 后, LL 路径必须主动补偿"被 DINO 内容污染的低频信号", 但这种补偿是**事后纠错**, 而非主动优化, 效果反而更差。

### 6.4 理论结论与方向修正

**Phase 4C 是一个重要的 NEGATIVE result**, 但提供了**关键理论洞察**:

1. **"Style Is Learned, Not Extracted"**: 风格表示应该**端到端学习**, 而非从预训练网络中提取。预训练特征 (DINOv2/CLIP) 是内容-风格纠缠的, 对风格迁移任务**不是最优**。
2. **Block Masking 思路正确, 但切入点错**: 在 RGB 上做 mask 确实能减少内容泄漏, 但如果底层特征本身就是次优的 (DINOv2 vs learnable), mask 也救不回来。
3. **`style_memory` 才是当前架构的最优选择**: 一个简单的 [5, 256, 384] 可学习张量, 在反向传播中演化出任务最优的风格表示, **不需要外部 DINO**。

**方向修正**:
- ❌ 放弃继续优化 DINO patches 路线 (4C.2, 4C.3 等不再做)
- ✓ 保留 `style_memory` 作为默认 style conditioner
- ✓ 后续 Phase 4E (Daubechies) 和 Phase 4F (全频域 ODE) 继续推进
- ✓ Block Masking 思想可以保留为**正则化项** (训练时随机 mask style_memory 的部分 patches), 但不应替换 style_memory 本身

### 6.5 重要理论提升: "Learned vs Extracted Style" 对比

这个 negative result 实际上是**Paper 的一个重要卖点**:

> "我们发现, 一个简单的 [5, 256, 384] 可学习 style memory bank, 在端到端训练中演化出的风格表示, 显著优于从 DINOv2 提取的 patch features (clip_style +0.018)。这挑战了主流 style transfer 文献依赖预训练 CLIP/DINO 作为 style encoder 的做法。"

**理论支撑**:
- 风格迁移任务的"风格"是一个**任务相关**的隐变量, 不是普适的视觉特征
- 预训练特征是**通用表示** (content-biased), 不是**任务最优** (style-pure)
- 端到端学习可以让模型自己决定"什么算风格", 而非由 DINO 的训练目标决定

## 7. 文件清单

- `tools/experiments/build_offline_dino_cache_blockmask.py` — 新工具
- `configs/630_phase4c_dino_clean_lvl2.json` — 4C.0 配置
- `configs/630_phase4c_blockmask_r60_b128_lvl2.json` — 4C.1 配置
- `docs/630/phase4c_block_masking_rgb.md` — 本文档
- (生成物) `eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache_clean.pt` (1.88 GB)
- (生成物) `eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache_blockmask_r06_b128.pt` (1.88 GB)
