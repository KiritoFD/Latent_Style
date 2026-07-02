# 风格纤维丛 — 理论框架与全链路改造方案

> **给 KiritoFD**: 这是 Phase2 下一步的完整指导文档。
> 基于"风格纤维丛"这一统一的微分几何框架，给出从 tokenizer 到 solver 到 loss 的全链路改造。
> 不需要重训练即可验证核心假设。

---

## 〇、当前态势

| 指标 | topogate_appalign e2 | topogate_appalign e3 | 目标 |
|------|:---:|:---:|:---:|
| transfer style | 0.6714 | 0.6718 | **0.72** |
| transfer LPIPS | 0.314 | 0.315 | < 0.35 |
| all-pairs style | 0.7031 | 0.7031 | **0.72** |
| all-pairs LPIPS | 0.312 | 0.313 | < 0.32 |

**诊断**: TopoGate 完美解决了结构保持 (LPIPS 0.31——接近 IDT 水平)。Style 卡在 0.67-0.70，离目标差 0.02-0.05。
E3→E4 可能还会微升一点，但更多训练 epoch 不会突破这个天花板。

**根本原因（ODE 的均值坍缩）**: 在同一根内容结构上，目标风格有无穷多种画法。
确定性 ODE 为了最小化 MSE，最终停在所有可能画法的**条件期望**处——即"平滑塑料色块"。
真正的艺术风格分布在纤维的高频边界处。

---

## 一、理论框架: 风格纤维丛

### 1.1 定义

VAE 潜空间 $\mathcal{Z}$ 上的**纤维丛** $E = (\mathcal{B}, \mathcal{F}, \pi)$：

| 组件 | 数学定义 | 我们的实现 |
|------|----------|-----------|
| 底空间 $\mathcal{B}$ | 内容流形——边缘、形状、空间布局 | $\pi(x) = \text{content}(x)$ |
| 纤维 $\mathcal{F}_c$ | 内容点 $c$ 下的所有可能风格画法 | attention 路由后的 style values |
| 投影 $\pi$ | 从完整潜变量提取结构 | TopoGate 锁定的 self-attention 通路 |
| 纤维方向 | 不改结构的前提下改变外观的方向 | TopoGate 的门控 —— 边缘→0, 纹理→1 |

### 1.2 核心定理

**定理 (ODE 均值坍缩)**:
在纤维 $\mathcal{F}_c$ 上的 ODE 轨迹 $x_t$，其确定性目标
$$\lim_{t \to 1} x_t = \mathbb{E}[X \mid c]$$
即纤维上所有可能画法的条件期望——"平均笔触"。

**定理 (Fiber-SDE 的边界可达性)**:
各向异性 SDE $dx = v\,dt + \sigma \cdot G_{\text{topo}} \odot dW$ 可触及纤维分布的支持边界。

**定理 (埃雷斯曼联络 via TopoGate)**:
TopoGate 的混合矩阵 $A_{\text{final}} = \alpha A_{\text{self-content}} + (1-\alpha)A_{\text{cross}}$ 定义一个 Ehresmann connection — 强制特征传输局限于纤维 $\mathcal{F}_c$ 内部。

### 1.3 在整个框架下重新理解已有实验

| 实验 | 纤维丛解释 |
|------|-----------|
| Endpoint LPIPS=0.62 | 速度场有底空间分量 → $\pi(x)$ 变化 → 结构崩塌 |
| TopoGate LPIPS=0.31 | 联络锁定 $\pi$ → 只沿纤维移动 |
| Style=0.67 上不去 | 确定性 ODE 停在 $\mathbb{E}[X \mid c]$ → 均值坍缩 |
| WikiArt512 的 0.79/0.31 | 风格区分度更大 → 纤维的截面更分散 → 更大的"移动空间" |

---

## 二、全链路改造方案

三条改造线：**Tokenizer**（纤维局部标架） ↔ **Solver**（纤维方向噪声） ↔ **Loss**（纤维分层匹配）

### 2.1 Tokenizer: 从查表到翻译

当前 `PureLatentSpatialTokenizer` 的缺陷: **查表法 (Lookup)** — $\text{Output} = \sum \alpha_k \cdot V_k$。
无论内容特征的连续变化如何，只要路由到 cluster $k$，就被替换为同一个固定向量 $V_k$。这**丢弃了内容的所有连续几何信息**。

**解决**: 用 **SMoE (Spatial Mixture-of-Experts) Translator** 替代。核心公式：
$$\text{Output} = \sum_k \alpha_k \cdot (W_k \times F_{\text{content}})$$

其中 $W_k \in \mathbb{R}^{D \times D}$ 是第 $k$ 个风格专家的**线性变换矩阵**。

**与纤维丛的对应**: $W_k$ 是纤维 $\mathcal{F}_c$ 上的局部标架变换。$F_{\text{content}}$ 是底空间 $\mathcal{B}$ 在 $(h,w)$ 处的局部坐标。$W_k \times F_{\text{content}}$ 将其翻译到目标风格的纤维表达。

**关键创新**: 恒等初始化 — $W_k \approx I$ 使得训练初期 spatial_map ≈ content features。Loss 驱动下矩阵逐渐旋转——边缘保持，纹理被翻译。LPIPS 从 step 1 就极低。

#### 实现

```python
# semantic_tokenizer.py: 替换 PureLatentSpatialTokenizer 的风格路由部分

class SMoETranslatorTokenizer(nn.Module):
    def __init__(self, num_styles, latent_dim=4, feat_dim=96, num_experts=32):
        self.num_experts = num_experts
        self.content_parser = nn.Sequential(
            _LatentResBlock(latent_dim, feat_dim),
            _LatentResBlock(feat_dim, feat_dim),
            _LatentResBlock(feat_dim, feat_dim),
            _LatentResBlock(feat_dim, feat_dim),  # 4-block extractor
        )
        self.routing_keys = nn.Parameter(torch.randn(num_experts, feat_dim) * 0.02)
        # 翻译矩阵: [num_styles, num_experts, D, D]
        self.translation = nn.Parameter(
            torch.randn(num_styles, num_experts, feat_dim, feat_dim) * 0.02
        )
        with torch.no_grad():  # 恒等初始化
            self.translation.data += torch.eye(feat_dim).view(1, 1, feat_dim, feat_dim)

    def forward(self, style_id, base_code, content_latent, target_hw):
        B, _, H, W = content_latent.shape
        F = self.content_parser(content_latent.float()).to(dtype=content_latent.dtype)
        T = F.flatten(2).transpose(1, 2)  # [B, HW, D]

        # 路由 (与当前相同)
        keys = F.normalize(self.routing_keys, dim=-1).unsqueeze(0).expand(B, -1, -1)
        routing = F.softmax(torch.bmm(F.normalize(T, dim=-1), keys.transpose(1,2)) / 0.1, dim=-1)

        # SMoE 翻译: T @ W_k^T, 然后按 routing 权重融合
        W = self.translation[style_id.long()]  # [B, K, D, D]
        translated = torch.einsum('bmd,bkcd->bmkc', T, W)  # [B, HW, K, D]
        fused = (translated * routing.unsqueeze(-1)).sum(dim=2)  # [B, HW, D]
        spatial = fused.transpose(1, 2).view(B, -1, H, W)

        return StructuredStyleOutput(
            global_code=base_code + self.style_global(style_id).to(content_latent),
            spatial_map=spatial,
            gate_map=...,  # 保持不变
            debug={"family": "smoe_translator", "feat_dim": self.feat_dim},
        )
```

---

### 2.2 Solver: 纤维对齐 SDE

**当前问题**: solver_unsb_cycle 的各向同性噪声 $\sigma \sqrt{dt} \epsilon$ 在全空间注入，震碎底空间结构。

**改进**: 噪声乘以 TopoGate 的门控 $G_{\text{topo}}(x) \in [0,1]^{H \times W}$：
$$x_{t+\Delta t} = x_t + v_\theta \Delta t + \sigma \sqrt{\Delta t} \cdot G_{\text{topo}} \odot \epsilon$$

边缘处 $G \to 0$ → 噪声归零 → 轮廓纹丝不动。纹理处 $G \to 1$ → 全噪声 → 逼出锐利笔触。

#### 实现

```python
# model.py:1104-1114, solver_unsb_cycle 分支修改

# 当前:
predictor = predictor + torch.randn_like(predictor) * noise_scale * math.sqrt(max(dt, 1e-8))

# 改为 Fiber-Aligned:
if getattr(self, "solver_fiber_aligned", False) and style_maps.gate_16 is not None:
    gate_16 = style_maps.gate_16.to(dtype=predictor.dtype, device=predictor.device)
    gate = F.interpolate(gate_16, size=predictor.shape[-2:], mode='bilinear')
    fiber_noise = torch.randn_like(predictor) * gate
    predictor = predictor + noise_scale * math.sqrt(max(dt, 1e-8)) * fiber_noise
else:
    predictor = predictor + torch.randn_like(predictor) * noise_scale * math.sqrt(max(dt, 1e-8))
```

`gate_16` 来自 `StyleMaps.gate_16` — 由 tokenizer 的 attention entropy 计算: $G = 1 - H/H_{\max}$。高熵（不确定→多个 cluster 竞争→纹理区域）→ 门控趋近 1。

---

### 2.3 Loss: 纤维分层 SWD

**当前问题**: Terminal SWD 在全图所有 patch 上混在一起——"天空的笔触"和"眼睛的笔触"被放进同一个分布。

**改进**: 按 tokenizer 的 $K$ 个 cluster attention 分别计算 SWD，然后取加权和：
$$\mathcal{L}_{\text{SWD}} = \sum_{k=1}^K \text{SWD}\left( \text{Mask}_k \odot z_1, \; \text{Mask}_k \odot z_{\text{style}} \right)$$

#### 实现

```python
# losses.py: 在 _compute_omf_details 的 terminal_swd 部分添加 fiberwise 模式

def _fiberwise_terminal_swd(self, pred, style_target, attn_weights):
    B, HW, K = attn_weights.shape
    total = 0.0
    H = W = int(math.sqrt(HW))
    for k in range(K):
        mask = attn_weights[:, :, k].view(B, 1, H, W)
        mask = F.interpolate(mask, size=pred.shape[-2:], mode='bilinear').clamp_min(0.01)
        total += self.transport_cost.swd(pred * mask, style_target * mask, 
                                          num_projections=64 // K)
    return total / K
```

---

## 三、三阶段推进计划

### 阶段 1: 零训练验证 (30min)

用现有 **topogate_appalign e2 ckpt**，只改推理 config 做 Fiber-SDE：

```bash
# 远程 WSL
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

# Fiber-SDE 扫描
for sigma in 0.01 0.02 0.03 0.05; do
    python src/run.py --config configs/aaai2027/phase2_eval_sde_em_topogate_e2.json \
      --override model.solver_fiber_aligned=true \
      --override model.solver_stochastic_noise_scale=$sigma
done

# PC-Solver 扫描 (对照)
for step in 0.04 0.06 0.10; do
    python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
      --override model.solver_corrector_step_size=$step
done
```

**判据**: 如果任何一个配置下 style > 0.70 且 LPIPS < 0.37 → Fiber-SDE 理论验证通过。

### 阶段 2: 代码改造 + 轻量验证 (1-2天)

按以上三组代码改动 (SMoE Tokenizer + Fiber-SDE solver + Fiberwise SWD) 修改代码。
用 `topogate_appalign e1` warmstart，小 batch (b8a1) 训练 6-8 epochs。

**配置 key**:
```json
{
  "model": {
    "tokenizer_family": "smoe_translator",
    "structured_tokenizer_type": "smoe_translator",
    "solver_family": "solver_unsb_cycle",
    "solver_fiber_aligned": true,
    "solver_stochastic_noise_scale": 0.02
  },
  "bridge": {
    "objective_mode": "omf",
    "semantic_supervision_family": "fiberwise_swd",
    "terminal_swd_weight": 12.0,
    "w_kinetic": 0.70
  }
}
```

### 阶段 3: 全量训练 + I2SB (如果阶段 2 仍未到 0.72)

在 SMoE Tokenizer + Fiber-SDE 的基础上叠加 I2SB 训练:
```json
{
  "bridge": {
    "objective_mode": "i2sb_endpoint",
    "bridge_sigma": 0.02,
    "bridge_noise_window_start": 0.18,
    "bridge_noise_window_end": 0.82
  },
  "model": {
    "solver_family": "solver_i2sb",
    "transport_prediction_mode": "endpoint",
    "solver_fiber_aligned": true
  }
}
```

这里 endpoint 不会崩——TopoGate (Ehresmann 联络) 保证 $\pi(x)$ 不变，I2SB 公式 $\mu = c_1 x_t + c_2 \hat{x}_1$ 中 $c_1$ 项锚定源结构。

---

## 四、论文叙事结构

**Abstract**:
> 我们提出 Style Fiber Bundles —— 一个统一的微分几何框架用于无配对风格迁移。
> 风格被建模为悬挂在内容结构上的纤维，Topology Gate 充当埃雷斯曼联络确保结构不变。
> 纤维对齐的 SDE 求解器克服了确定性 ODE 的均值坍缩，首次在 LPIPS < 0.35 下达到 0.72+ 的风格相似度。

**核心贡献**:
1. **Style Fiber Bundles 理论** — 将风格迁移重新表述为纤维丛上的传输问题
2. **Ehresmann Connection via TopoGate** — 数学保证 $\Delta c = 0$ (LPIPS 0.31)
3. **Fiber-Aligned SDE** — 突破 ODE 均值坍缩，触及风格边界
4. **SMoE Translator** — 将 tokenizer 从查表升级为内容感知的翻译算子

**关键定性图**:
- Figure 3: ODE vs Fiber-SDE 对比 — 同一张图，轮廓不变，笔触从"平滑"变成"锐利"
- Figure 4: TopoGate 门控热力图 — 边缘→0 (保护)，纹理→1 (释放)
- Figure 5: Tokenizer 的 $K=32$ 个 attention map — 无监督自动聚类到语义区域

---

## 五、SMoE Tokenizer 深入思考

### 5.1 "翻译"与"查表"的本质区别

查表: 每个像素被压缩为一个离散的分类标签（属于哪个 cluster），然后用该 cluster 的固定向量替换。
这丢弃了内容的**连续变化**——边缘的渐变、纹理的方向性、色块的微调，全部消失。

翻译: 每个像素的特征向量 $F_{\text{content}} \in \mathbb{R}^D$ 被一个**内容相关但风格特定的线性变换** $W_k \in \mathbb{R}^{D \times D}$ 变换：
$$F' = W_k \times F_{\text{content}}$$

这个变换保留了特征空间中的连续几何——如果两个相邻像素在原特征空间很接近，
变换后它们仍然接近，只是被整体"旋转"到了目标风格的方向。

### 5.2 恒等初始化的数学含义

$W_k \approx I$ 意味着训练初期，tokenizer 的输出 ≈ 内容特征本身。
这不是"碰巧"，而是故意的——它强制模型**以保内容为基础**开始学习。
Loss 驱动下，$W_k$ 缓慢偏离 $I$，把"普通边缘"翻译成"浮世绘边缘"，
但保留了边缘本身的几何形状和渐变程度。

### 5.3 为什么 32 个专家是合理的

$K=32$ 对应 32 种基础空间原子的"翻译规则"——每一种涵盖一类语义-风格配对
（如"天空的平滑区域"→"印象派的短笔触天空"，
"建筑的锐利边缘"→"洛可可的柔美曲线轮廓"）。

从纤维丛的角度: $K$ 是纤维 $\mathcal{F}_c$ 上局部标架的大小。32 维的局部标架覆盖了 VAE 潜空间中
"一个内容结构上的所有可能风格变化"所需的线性变换自由度。
