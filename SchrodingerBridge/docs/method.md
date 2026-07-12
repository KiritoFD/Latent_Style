# WEAVE 方法文档

本文档是对 WEAVE (Wavelet-decoupled AdaIN Velocity Estimator) 方法的完整描述，作为论文 Method 部分的支撑材料，并给出架构图的绘制规范。

所有内容已通过 Stage1-16 消融实验验证，仅保留有效组件，无效组件（SWD/cross-attn/ASG/edge loss/统计矩损失）已剔除或标注为辅助。

---

## 1. 核心思想

风格迁移的"恒等捷径"（identity shortcut）不是优化失败，而是几何必然：当内容和风格共享同一坐标基时，自然图像的 $1/f$ 频谱能量分布使低频内容误差主导梯度，风格信号被饿死。

WEAVE 通过**坐标变换**解决：用 Haar 小波分解将 latent 分到正交子带，内容锁在 LL，风格只走高频通路。这样优化器无法用内容换风格，也无法用风格换内容。

**仅有 3 个有效组件**（经消融验证）：
1. **Haar 小波分解** — 结构-风格正交分离的几何基础
2. **Flow Matching** — 内容→风格的可学习传输引擎（占梯度 92%）
3. **Endpoint AdaIN** — 主要风格注入通道（移除后 CLIP-S -0.019，输出回退到内容保留）

其余组件（SWD、cross-attention、ASG、edge loss、统计矩损失）经消融验证贡献 <0.002 或 0，已在描述中降级或剔除。

---

## 2. 框架设置

### 2.1 工作空间
- **VAE latent 空间**：Stable Diffusion v1.5 EMA VAE，$z_0 = \mathcal{E}(x_c) \in \mathbb{R}^{4 \times 32 \times 32}$
- 训练时 $z_1 \sim p_s$ 为目标风格域采样的 latent

### 2.2 Rectified Flow
直线传输路径：
$$z_t = (1-t)z_0 + tz_1, \quad u_t = z_1 - z_0, \quad t \sim \mathcal{U}([0,1])$$

选择 Rectified Flow 而非 DDPM：直线轨迹用少量 Euler 步即可逼近（8 步收敛），避免扩散的数千次去噪迭代，使训练和推理都能在单消费级 GPU 上完成。

推理：从 $z_0$ 积分到 $\hat{z}_1$，8 步 Euler 求解器，解码 $\hat{x}_s = \mathcal{D}(\hat{z}_1)$。

### 2.3 IDT 校准
**Identical-Image Transfer (IDT) Floor**：未改变的源图定义的 no-op 基线。有效迁移必须满足 $\mathrm{CLIP}(x_{\mathrm{out}}, s_{\mathrm{tgt}}) \geq \mathrm{CLIP}(x_{\mathrm{src}}, s_{\mathrm{tgt}})$。低于 floor 表示负迁移。

---

## 3. 小波分解与频率路由

### 3.1 Haar 小波分解
单级 Haar 变换将 $4 \times 32 \times 32$ latent 分解为 4 个正交子带：
- **LL**（低频，$4 \times 16 \times 16$）— 结构
- **LH / HL / HH**（高频，各 $4 \times 16 \times 16$）— 风格

记 $\mathcal{W}(z_t) = (\ell_t, h_{1,t}, h_{2,t}, h_{3,t})$。变换正交，Frobenius 范数可加性分解：
$$\|z\|_F^2 = \|\ell\|_F^2 + \|h_1\|_F^2 + \|h_2\|_F^2 + \|h_3\|_F^2$$

**关键性质**：只修改高频子带，LL 坐标不变 → 结构保持是构造性的，不依赖损失权衡。

选择 Haar 的理由：精确正交、紧支撑、极低计算开销。单级分解对紧凑 latent 分辨率已足够。

### 3.2 频率探测（动机）
跨 5000 个随机风格对：
- LL 包含 69.5% 的目标传输能量平方
- 归一化风格分离度在 HH 最强

→ 动机：弱化 LL 监督，强化高频外观统计。

### 3.3 Structure-Aligned Target (SAT)
训练目标采用结构对齐构造：
$$\mathrm{target} = \mathrm{IDWT}(\ell_{\mathrm{content}}, h_{1,\mathrm{style}}, h_{2,\mathrm{style}}, h_{3,\mathrm{style}})$$

LL 锁死为 content，高频子带用 style。这是 DINO-S 0.48 天花板的根因（见 §6）。

---

## 4. 架构

### 4.1 总览
速度网络由 **4 个残差块**（width=64，约 903K 可训练参数）+ **3 个独立预测头**组成。每个监督子带（$\ell, h_1, h_2$）一个 $1 \times 1$ 卷积头。

**HH 无速度头**：Euler 积分器永不修改 HH，它只在端点步通过 AdaIN 风格化。

### 4.2 组件清单（对应 `src/model.py`）

| 组件 | 实现 | 作用 | 有效？ |
|------|------|------|--------|
| `input_proj` | `Conv2d(4C→dim, 3×3)` | 4 子带堆叠 → backbone 宽度 | ✅ |
| `time_proj` | 2 层 MLP | 时间嵌入 → backbone 宽度 | ✅ |
| `blocks` ×4 | `ResidualBlock(dim=64)` | 共享主干，含 RMSNorm | ✅ |
| `head_ll/lh/hl` | `VelocityHead` (1×1 Conv) | 预测 3 个子带速度场 | ✅ |
| `head_hh` | 无（默认禁用） | HH 不学速度 | — |
| `style_conditioner` | `StyleConditioner` | style_memory 256 tokens → bridge dim | ✅（载体）|
| cross-attention | 残差块内 | style tokens → backbone | ⚠️ 辅助（ΔCLIP-S <0.001）|
| **Endpoint AdaIN** | 推理端点步 | 主要风格注入 | ✅ **核心** |

### 4.3 关键设计决策

1. **RMSNorm（非 GroupNorm）**：GroupNorm 抹除通道均值，而色调信息恰在通道均值中。
2. **LL 作为分离通道**：LL 进入 backbone 但不进 attention 路径，保持频率分离。
3. **HH 无头**：架构上冻结最细对角细节，只通过端点 AdaIN 风格化。

---

## 5. 训练目标

### 5.1 主损失：频谱 Flow Matching
$$\mathcal{L}_{\mathrm{impl}} := \mathbb{E}_t\left[\lambda_{LL}\|v_\ell - u_\ell\|_2^2 + \|v_{h_1} - u_{h_1}\|_2^2 + \|v_{h_2} - u_{h_2}\|_2^2\right]$$

其中 $\lambda_{LL} = 0.3$（弱化 LL，但非零——某些风格也移动粗色调）。

**预测单步目标**：$\hat{z}_1 = z_0 + \mathcal{W}^{-1}(v_\ell, v_{h_1}, v_{h_2}, 0)$

### 5.2 辅助损失（<5% 梯度，移除无影响）
$$\mathcal{L} = \mathcal{L}_{\mathrm{impl}} + \underbrace{0.1\mathcal{L}_{\mathrm{edge}} + \mathcal{L}_{\mathrm{low}}}_{\text{auxiliary }(<5\%\text{ gradient})}$$

- $\mathcal{L}_{\mathrm{edge}}$：高通残差 $\ell_1$ 匹配
- $\mathcal{L}_{\mathrm{low}}$：低通分量 MSE 锚定

$\mathcal{L}_{\mathrm{impl}}$ 独自驱动学习（~92% 梯度）；辅助项移除后输出无可测变化。

### 5.3 训练配置
- AdamW，peak LR $2 \times 10^{-4}$，cosine schedule
- 5 epochs，batch 96，AMP bf16
- 单 RTX 3060 (12GB)，1.5 分钟收敛
- **学习率是唯一敏感训练超参**；其余（$\sigma$, gate init, $\lambda_{HH}$, loss type）在 $10\times$–$20\times$ 扰动下稳健

---

## 6. Endpoint AdaIN（主要风格注入通道）

Flow Matching 速度场传输内容 latent 向目标域，但速度信号本身是弱风格通道——LL 主导的重建损失给风格条件梯度留的空间很小。

### 6.1 对齐算子
设 $\mathcal{W}(\hat{z}_1) = (\hat{\ell}, \hat{h}_1, \hat{h}_2, \hat{h}_3)$ 为当前预测，$\mathcal{W}(z_1^\star) = (\ell^\star, h_1^\star, h_2^\star, h_3^\star)$ 为目标风格参考。

对 $i \in \{1,2,3\}$，逐子带 AdaIN：
$$T_i(a) = \frac{a - \hat{\mu}_i}{\hat{\sigma}_i}\sigma_i^\star + \mu_i^\star$$

其中 $(\hat{\mu}_i, \hat{\sigma}_i)$ 和 $(\mu_i^\star, \sigma_i^\star)$ 为 $\hat{h}_i$ 和 $h_i^\star$ 的逐通道均值/标准差。

**为何 AdaIN 而非 WCT**：每个 Haar 子带仅 4 通道，全协方差白化-着色退化为对角（mean+std）形式。AdaIN 的 mean+std 匹配对 4 通道 latent 近对角协方差是近最优的：简单、快、且推理时间增加 0%。

### 6.2 高频缩放混合
高频尺度 $(s_1, s_2, s_3) = (0.3, 0.3, 0.5)$：
$$\hat{h}_i^+ = (1-s_i)\hat{h}_i + s_i T_i(\hat{h}_i), \quad i \in \{1,2,3\}$$

$\hat{\ell}$ 不变。对齐仅在最终（端点）Euler 步应用一次。

### 6.3 消融地位
- **移除 AdaIN**：CLIP-S $-0.019$，DINO-C $+0.061$（输出回退到内容保留）
- **AdaIN vs WCT**：WCT 的 LH 风格迁移比 $0.30 \to 0.33$（+8.8%），但内容漂移增加、推理时间 +4%
- **逐子带 AdaIN**：严格更差（$0.30 \to 0.09$，-71%），因模型训练时用全局 spatial-fiber 模式

---

## 7. 推理配置

- **8 步 Euler 求解器**：1 步崩溃（DINO-C $-0.307$），32 步与 8 步持平 → 8 是效率-精度 Pareto 点
- **外推系数** $\alpha_{\mathrm{extrap}} = 0.1$：略微外推对齐统计；$\alpha = 1.0$ 灾难性崩溃（LPIPS 0.59，协方差奇异）
- 推理 750 对：50 秒（VAE decode ~40 秒占主导，速度网络 <10 秒）

---

## 8. DINO-S 0.48 天花板（Stage7-16 验证）

10 个变体全部收敛到 DINO-S $= 0.480 \pm 0.003$，确认 0.48 是 SAT 范式（903K 参数，5 epochs，D5）的 fundamental limit。

| 变体 | DINO-S | 说明 |
|------|--------|------|
| baseline | 0.473 | SAT, LL 锁死 |
| 增量分支 | 0.480 | $v = v_c + g \cdot v_s$ |
| CFG | 0.480 | drop 0.15, scale 1.5 |
| 训练时 AdaIN | 0.480 | 训练阶段注入 |
| LL AdaIN α=0.5 | 0.479 | LL 部分风格化 |
| LL WCT α=0.5 | 0.479 | 全协方差 LL |
| LL AdaIN α=1.0 | 0.476 | 完全替换 LL |
| CFG+LL AdaIN | 0.479 | 组合 |
| Huber+HH | 0.480 | loss 类型 |
| HF WCT β=0.5 | 0.474 | 高频 WCT |

**根因**：LL 子带携带 DINOv2 敏感的色彩/对比度统计，但被 SAT 结构性锁死。解锁 LL 是风格-内容零和博弈，不改善 Pareto 前沿。
- LL 部分风格化：改善内容（LPIPS $-0.03$）但不影响 DINO-S
- HF WCT：DINO-S $-0.006$，DINO-C $+0.019$（零和）

---

## 9. 架构图绘制规范

本节给出架构图的绘制说明，可用于 draw.io / TikZ / 手绘。推荐使用 draw.io-ai 生成或基于 `docs/630/aaai_arch_diagram_v16_staggered_bundle.drawio` 迭代。

### 9.1 整体布局（从左到右横向流）

```
[Content x_c] → [VAE Enc] → z_0 (4×32×32)
                                    │
                                    ▼
                              [Haar DWT]
                                    │
              ┌───────────┬─────────┴────────┬──────────┐
              ▼           ▼                  ▼          ▼
             LL         LH                  HL         HH
           (4×16×16)  (4×16×16)          (4×16×16)  (4×16×16)
              │           │                  │          │
              │     (LL 锁死, 不学速度)      │     (HH 无头, 端点 AdaIN)
              │           │                  │          │
              └─────┬─────┴────────┬─────────┘          │
                    ▼              │                    │
              [Stack 4C → input_proj Conv3×3]           │
                    │                                  │
              [time_proj MLP] ─── t                    │
                    │                                  │
              [ResidualBlock ×4] ←─ style_memory (cross-attn, 辅助)
                    │                                  │
              ┌─────┼─────┐                            │
              ▼     ▼     ▼                            │
          head_ll head_lh head_hl                      │
          (1×1)   (1×1)   (1×1)                        │
              │     │     │                            │
              └─────┼─────┘                            │
                    ▼                                  │
              [iDWT] ← (v_ℓ, v_h1, v_h2, 0)            │
                    │                                  │
                    ▼                                  │
              ẑ_1 = z_0 + W⁻¹(v)                       │
                    │                                  │
                    ▼                                  │
         ┌─── [8步 Euler 积分] ───┐                     │
         │                        │                     │
         ▼                        ▼                     │
    [最终步: Endpoint AdaIN] ← style z_1★ 的 h_i★ 统计
         │  T_i(a) = (a-μ̂)/σ̂·σ★+μ★
         │  ĥ_i⁺ = (1-s_i)ĥ_i + s_i·T_i(ĥ_i)
         ▼
    [VAE Dec] → 风格化输出 x̂_s
```

### 9.2 配色规范

| 元素 | 颜色 | 说明 |
|------|------|------|
| 内容通路（LL, z_0, VAE Enc/Dec） | 蓝色系 | 结构保持 |
| 风格通路（LH/HL/HH, AdaIN, style_memory） | 橙/红色系 | 风格注入 |
| 冻结/不可学（HH 无头, VAE） | 灰色 | 不参与训练 |
| 可训练（backbone, heads） | 实线边框 | 903K 参数 |
| 辅助/无效（cross-attn） | 虚线边框 | 消融验证 Δ<0.001 |

### 9.3 关键标注（必须在图中体现）

1. **LL 锁死标记**：LL 子带旁标注 "locked / no style gradient"，箭头不进 style 路径
2. **HH 无头标记**：HH 子带旁标注 "no velocity head, endpoint AdaIN only"
3. **AdaIN 端点标记**：在 Euler 积分末步突出显示 Endpoint AdaIN 模块，标注 "primary style channel"
4. **参数量**：backbone 标 "903K trainable"，VAE 标 "84M frozen"
5. **SAT 公式**：在 DWT 后标注 target = IDWT(LL_c, LH_s, HL_s, HH_s)
6. **子带尺寸**：每个子带标 "4×16×16"

### 9.4 推荐的图分块（3 个水平带）

1. **编码与分解带**（顶部）：x_c → VAE Enc → z_0 → Haar DWT → 4 子带
2. **速度网络带**（中部）：4 子带 → input_proj → backbone ×4 → 3 heads → iDWT → ẑ_1
3. **推理与端点带**（底部）：8 步 Euler → Endpoint AdaIN ← style stats → VAE Dec → x̂_s

### 9.5 与 `docs/630/` 现有图的关系

`docs/630/aaai_arch_diagram_v16_staggered_bundle.drawio` 是当前最新版本。本规范与 v16 的差异：
- **需补充**：HH 无头的明确标注（v16 未突出）
- **需补充**：Endpoint AdaIN 的 "primary style channel" 强调
- **需弱化**：cross-attention 路径改为虚线（辅助组件）
- **需补充**：SAT 公式标注

建议在 v16 基础上迭代为 v17，或用 draw.io-ai 重新生成符合本规范的版本。

### 9.6 备选：简化版架构图（论文用）

论文正文受版面限制，推荐用简化版：
- 只画 3 个子带通路（LL, LH/HL 合并, HH）
- 省略 style_memory cross-attention（辅助组件）
- 突出 Haar DWT / iDWT 和 Endpoint AdaIN 两个关键算子
- 用色块区分"内容锁"（蓝）和"风格注入"（橙）

---

## 10. 有效组件验证摘要

| 组件 | 消融 | 结论 |
|------|------|------|
| **Flow Matching** | 移除后 CLIP-S $-0.017$，内容指标提升但风格崩 | 核心传输引擎 |
| **Haar Wavelet** | Latent-WCT（零参数）全指标崩 | 正交分离几何基础 |
| **Endpoint AdaIN** | 移除后 CLIP-S $-0.019$，DINO-C $+0.061$ | 主要风格通道 |
| cross-attention | ΔCLIP-S $<0.001$ | 辅助，可删 |
| SWD | loss 占比 4.3%，移除无变化 | 无效，已删 |
| ASG | Δ=0.000 | 无效，已删 |
| edge loss | <5% 梯度，移除无变化 | 辅助 |
| 统计矩损失 | DINO-C $-0.32$（崩溃） | 有害，已删 |

---

## 11. 相关代码位置

| 功能 | 文件 | 关键行 |
|------|------|--------|
| WEAVE 模型 | `src/model.py` | `class WEAVE` L328 |
| Haar DWT/iDWT | `src/wavelet.py` | `dwt2_haar`, `idwt2_haar` |
| 训练目标 (SAT, Flow Matching) | `src/flow.py` | `class FlowMatchingObjective` L20 |
| Endpoint AdaIN | `src/model.py` | `_adain_match_subband` L29 |
| VelocityHead | `src/model.py` | L210 |
| 配置 schema | `src/config_schema.py` | 全部配置项 |
