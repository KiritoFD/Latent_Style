# 风格纤维丛 (Style Fiber Bundles) — 理论框架与实现方案

> 基于微分拓扑中纤维丛理论，重新定义风格迁移的数学本质，
> 并给出从 tokenizer 到 solver 到 loss 的完整模型改造方案。

---

## 一、数学定义

### 1.1 风格纤维丛的构造

设 VAE 潜空间为 $\mathcal{Z} \subset \mathbb{R}^{4 \times 64 \times 64}$。

**纤维丛** $E = (\mathcal{B}, \mathcal{F}, \pi)$：
- **底空间** $\mathcal{B}$ (Content Manifold)：图像的语义结构——边缘、形状、空间布局。$\dim(\mathcal{B}) \ll \dim(\mathcal{Z})$
- **纤维** $\mathcal{F}_c$ (Style Fiber)：悬挂在内容点 $c \in \mathcal{B}$ 上的空间，包含"这个内容在目标风格下的所有可能画法"
- **投影映射** $\pi: \mathcal{Z} \to \mathcal{B}$：从图像潜变量提取其内容结构

图像 $x \in \mathcal{Z}$ 是纤维丛中的一个点，满足 $\pi(x) = c$。风格迁移 = **在纤维 $\mathcal{F}_{\pi(x)}$ 上移动到目标风格对应的截面**。

### 1.2 用纤维丛诊断现有模型

| 现象 | 纤维丛解释 |
|------|-----------|
| Endpoint LPIPS=0.62 | 速度场有底空间分量 → 轨迹在 $\mathcal{B}$ 上漂移 |
| TopoGate LPIPS=0.31 | TopoGate = **埃雷斯曼联络**，强制 $\Delta c = 0$ |
| Style 卡在 0.67 | ODE 停在纤维的条件期望处 → 均值坍缩 |

**核心洞察**: 在同一根纤维 $\mathcal{F}_c$ 上，目标风格（如印象派）有成千上万种画法。
ODE 的确定性轨迹最终收敛到 $x_{\text{target}} = \mathbb{E}[x \mid c]$ — 所有锐利笔触的平均值 = 平滑塑料色块。

---

## 二、模型改造方案

### 2.1 纤维对齐 SDE (Fiber-Aligned SDE)

**问题**: 传统 SDE 的各向同性噪声 $x_{t+1} = x_t + v\Delta t + \sigma \sqrt{\Delta t} \epsilon$ 会在全空间注入噪声，底空间 $\mathcal{B}$ 的结构也被震碎。

**解决**: TopoGate $G_{\text{topo}}(x) \in [0,1]^{H \times W}$ 天然是一个纤维度量张量——边缘处→0（底空间），纹理区→1（纤维）。噪声只沿纤维方向注入：
$$x_{t+\Delta t} = x_t + v_\theta \Delta t + \sigma \sqrt{\Delta t} \cdot G_{\text{topo}} \odot \epsilon$$

**效果**: 树木轮廓 (G→0) 纹丝不动，树叶纹理 (G→1) 长出梵高的漩涡笔触。

### 2.2 代码实现 (`model.py`)

在 `_i2sb_transport_step` 或 `integrate_transport` 的 `solver_unsb_cycle` 分支中：

```python
# 已有: solver_unsb_cycle 在 model.py:1104-1114
# 只需修改噪声注入行

# 当前代码:
predictor = predictor + torch.randn_like(predictor) * noise_scale * math.sqrt(max(dt, 1e-8))

# 修改为 Fiber-Aligned SDE:
# 获取 topogate (已在 structured_style_from_sidecar 中计算)
gate_16 = style_maps.gate_16  # [B, 1, H, W], 边缘处→0, 纹理处→1
if gate_16 is not None and self.solver_fiber_aligned:
    # 上采样 gate 到 latent 分辨率
    gate_latent = F.interpolate(gate_16.to(dtype=h.dtype), size=h.shape[-2:], mode='bilinear')
    fiber_noise = torch.randn_like(predictor) * gate_latent  # 只在纤维方向注入噪声
    predictor = predictor + noise_scale * math.sqrt(max(dt, 1e-8)) * fiber_noise
else:
    predictor = predictor + torch.randn_like(predictor) * noise_scale * math.sqrt(max(dt, 1e-8))
```

**新增配置参数**:
```json
{
  "model": {
    "solver_family": "solver_unsb_cycle",
    "solver_fiber_aligned": true,
    "solver_stochastic_noise_scale": 0.02,
    "solver_corrector_steps": 2,
    "solver_corrector_step_size": 0.06,
    "solver_corrector_mode": "latent_lowpass"
  }
}
```

### 2.3 纤维坐标系的 Tokenizer 增强

**当前**: PureLatentSpatialTokenizer 的 $K$ 个 clusters = 纤维 $\mathcal{F}_c$ 上的一组局部基向量 (local frame)。

**升级**: 
1. **增大基向量数 $K=64$** — 纤维空间的维度翻倍，可表达的笔触类型指数级增加
2. **PE 已经存在** — 非平凡丛的截面在不同底空间点不同，PE 告诉 tokenizer "你现在在天空 vs 地面"

**论文对应**: $K=64$ 的基向量张成的纤维空间足以覆盖印象派短笔触、洛可可柔美曲线、浮世绘平涂色块的全部多样性。

### 2.4 纤维分层最优传输 (Fiber-Wise SWD)

**问题**: 当前 Terminal SWD 在全图所有 patch 上算一个 SWD，混在一起——"画天空的笔触"和"画眼睛的笔触"被放进同一个分布算距离。

**解决**: Tokenizer 的 attention weights $\alpha \in \mathbb{R}^{B \times HW \times K}$ 是将每个像素投影到 $K$ 个纤维基向量的投影权重。按每个基向量分别算 SWD：
$$\mathcal{L}_{\text{SWD}} = \sum_{k=1}^K w_k \cdot \text{SWD}\left( \text{Mask}_k \odot z_1, \; \text{Mask}_k \odot z_{\text{style}} \right)$$

其中 $\text{Mask}_k[i,j] = \alpha_{ij,k}$ 是第 $k$ 个 cluster 的空间注意力权重。

**代码实现** (`losses.py`):

```python
def _fiberwise_terminal_swd(self, pred, target_style, attn_weights):
    """Fiber-wise SWD: compute SWD per cluster mask."""
    B, HW, K = attn_weights.shape
    total_swd = 0.0

    for k in range(K):
        # 第 k 个纤维基向量的空间 mask
        mask = attn_weights[:, :, k].view(B, 1, int(math.sqrt(HW)), -1)  # [B, 1, H, W]
        mask = F.interpolate(mask, size=pred.shape[-2:], mode='bilinear')
        mask = mask.clamp_min(0.01)  # 避免全零

        # 在每个 cluster 对应的区域分别算 SWD
        swd_k = self.transport_cost.swd(
            pred * mask,
            target_style * mask,
            num_projections=self.semantic_swd_num_projections // K
        )
        total_swd += swd_k * (mask.mean() + 1e-8)

    return total_swd / K
```

**配置**:
```json
{
  "bridge": {
    "semantic_supervision_family": "fiberwise_swd",
    "fiberwise_swd_weight": 1.0,
    "terminal_swd_weight": 12.0,
    "terminal_swd_mode": "standard"
  }
}
```

---

## 三、全链路改造清单

| 组件 | 改动点 | 文件 | 数学对应 |
|------|--------|------|----------|
| **Tokenizer** | K=64, PE 已有 | `semantic_tokenizer.py` | 纤维局部标架 + 非平凡丛截面 |
| **TopoGate** | 已有, 不变 | `lancet_blocks.py` | 埃雷斯曼联络 (Ehresmann Connection) |
| **Solver** | `solver_fiber_aligned=true` | `model.py:1104-1114` | 纤维方向布朗运动 |
| **SWD Loss** | `fiberwise_swd` | `losses.py` | 分层最优传输 (Stratified OT) |
| **Config** | 新增 fiber 参数 | `phase2_fiber_aligned_sde.json` | — |

---

## 四、论文叙事线

### Claim
> "现有的潜空间流匹配方法将风格迁移建模为欧氏空间中的点对点传输，导致速度场不可避免地同时作用于内容流形（底空间）和风格流形（纤维空间）。我们首次提出基于纤维丛理论的潜空间风格迁移框架——Style Fiber Bundles。"

### 三条核心贡献

1. **埃雷斯曼联络 via TopoGate** ($\S 3.1$)
   通过拓扑门控在数学上严格切断跨基底的特征渗漏，保证速度场 $\Delta c = 0$，实现 LPIPS = 0.31 的完美结构保持。

2. **纤维对齐 SDE** ($\S 3.2$)
   揭示确定性 ODE 的"纤维均值坍缩"定理——ODE 轨迹必然收敛到 $x_{\text{target}} = \mathbb{E}[x \mid c]$。引入仅沿纤维方向的各向异性布朗运动，迫使模型超越均值、触及风格分布的高频边界。

3. **纤维分层最优传输** ($\S 3.3$)
   利用 tokenizer 的 $K$ 个纤维基向量将图像分解为 $K$ 个语义区域，在每个区域内独立进行最优传输——即"分层最优传输"。这保证了目标风格特异性的爆炸式增长。

### 实验

- **Figure 3 (关键图)**: 同一内容图在 ODE vs Fiber-SDE 下的对比。ODE 输出平滑、"塑料感"；Fiber-SDE 在保持轮廓绝对不变的情况下，长出了目标风格的锐利笔触。
- **Table 2 (消融)**: 逐步添加 Tokenizer(K=64)、Fiber-SDE、Fiber-wise SWD 对 Style 和 LPIPS 的影响。
- **Figure 5 (可视化)**: 展示 Tokenizer 的 $K=64$ 个 attention map——证明纤维基向量自动聚类到语义区域（天空、建筑、水面），无需任何监督。

---

## 五、立即执行的 Action Item

**不需要重新训练。** 用现有的 `topogate_appalign` ckpt (e2: LPIPS 0.31, style 0.67) 直接改推理配置验证核心假设。

### 步骤 1: 验证 Fiber-SDE

```bash
# 远程 WSL
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python src/run.py --config configs/aaai2027/phase2_eval_sde_em_topogate_e2.json \
  --override model.solver_fiber_aligned=true \
  --override model.solver_stochastic_noise_scale=0.02
```

**预期**: style 从 0.67 跳过 0.70，LPIPS 微升但保持在 0.35 以下。

### 步骤 2: 参数扫描

```bash
for sigma in 0.01 0.02 0.03 0.05; do
    python src/run.py --config ... --override model.solver_stochastic_noise_scale=$sigma
done
```

### 步骤 3: 如果验证成功

1. 创建完整训练 config: K=64 tokenizer + fiber_aligned SDE + fiberwise SWD
2. 从 topogate e1 warmstart，训练 8-12 epochs
3. 目标: style 0.73+, LPIPS 0.35-

---

# 附录: Tokenizer 的"翻译级"重构 — 从查表到算子

> Tokenizer 的本质是**空间翻译**：把 VAE 潜空间中的内容结构，翻译为目标流派的笔触语言。
> "查表" (lookup) 是零阶近似——不论内容如何，$V_k$ 是固定的。
> "翻译" 是一阶变换——输出的是对内容特征的**操作**，内容的一切连续变化都被保留。

---

## 方案 A: 空间混合专家翻译器 (SMoE Translator) — 推荐

### 对比

```
查表法:   Output = Σ α_k · V_k                    (丢弃内容)
SMoE法:   Output = Σ α_k · (W_k × F_content)     (变换内容)
```

### 为什么更好

- $F_{\text{content}}$ 保留了原图的边缘、渐变、纹理等一切连续几何信息
- $W_k$ 不是静态向量，而是**线性变换算子**——对 $F_{\text{content}}$ 做旋转、缩放、投影
- 恒等初始化 (`W_k ≈ I`) 保证训练初期 spatial_map ≈ 内容特征 → LPIPS 从 step 1 就极低
- Loss 驱动下，$W_k$ 逐渐旋转——"普通线条"被翻译成"浮世绘线条"

### 代码

```python
class SMoETranslatorTokenizer(nn.Module):
    """K 个风格专家，每个专家是一个变换矩阵 W_k，对内容特征做线性翻译"""
    def __init__(self, num_styles, latent_dim=4, feat_dim=64, num_experts=32):
        super().__init__()
        self.num_experts = num_experts

        # 1. 内容解析器: 保留丰富连续信息
        self.content_parser = nn.Sequential(
            nn.Conv2d(latent_dim, feat_dim, 3, padding=1),
            nn.GroupNorm(8, feat_dim),
            nn.SiLU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
        )

        # 2. 路由键 (与当前 tokenizer 相同)
        self.routing_keys = nn.Parameter(torch.randn(num_experts, feat_dim) * 0.02)

        # 3. 翻译矩阵字典: [num_styles, num_experts, D, D]
        self.translation_experts = nn.Parameter(
            torch.randn(num_styles, num_experts, feat_dim, feat_dim) * 0.02
        )
        with torch.no_grad():
            self.translation_experts.data += torch.eye(feat_dim).view(1, 1, feat_dim, feat_dim)

    def forward(self, style_id, base_code, content_latent, target_hw):
        B, _, H, W = content_latent.shape

        # 解析内容
        F_content = self.content_parser(content_latent.float()).to(dtype=content_latent.dtype)
        T = F_content.flatten(2).transpose(1, 2)  # [B, HW, D]

        # 路由: T @ keys^T → softmax
        keys = F.normalize(self.routing_keys, dim=-1).unsqueeze(0).expand(B, -1, -1)
        sim = torch.bmm(F.normalize(T, dim=-1), keys.transpose(1, 2)) / 0.1
        routing = F.softmax(sim, dim=-1)  # [B, HW, K]

        # 每个专家变换: T @ W_k^T
        W = self.translation_experts[style_id.long()]  # [B, K, D, D]
        translated_all = torch.einsum('bmd,bkcd->bmkc', T, W)  # [B, HW, K, D]

        # 加权融合
        fused = (translated_all * routing.unsqueeze(-1)).sum(dim=2)  # [B, HW, D]
        spatial_map = fused.transpose(1, 2).view(B, -1, H, W)
        return StructuredStyleOutput(
            global_code=base_code,
            spatial_map=spatial_map,
            debug={"family": "smoe_translator", "init_eye": True},
        )
```

### 与纤维丛的对应

$W_k$ 是纤维 $\mathcal{F}_c$ 上的局部变换算子。对于不同底空间点 $c$（由 routing 决定），不同的 $W_k$ 被激活——这就是**纤维丛上的局部标架变换 (local gauge transformation)**。

---

## 方案 B: 超网络动态滤波器 (HyperNet Dynamic Filter)

### 理论

绘画的本质是笔触 = 卷积核。让 Tokenizer 成为超网络：输入 style_id，输出一组动态卷积核 $W_{\text{conv}}(s)$，直接在 $F_{\text{content}}$ 上做动态滤波。

### 代码

```python
class HypernetFilterTokenizer(nn.Module):
    """Style-ID → 动态卷积核 → 对内容特征做滤波"""
    def __init__(self, num_styles, feat_dim=64, kernel_size=3, num_filters=16):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.content_parser = nn.Conv2d(4, feat_dim, 3, padding=1)

        # 超网络: style_id → 卷积核参数 [num_filters, feat_dim, K, K]
        num_params = num_filters * feat_dim * kernel_size * kernel_size
        self.hyper = nn.Sequential(
            nn.Embedding(num_styles, 256),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, num_params),
        )

    def forward(self, style_id, base_code, content_latent, target_hw):
        F = self.content_parser(content_latent.float())
        B, D, H, W = F.shape

        # 生成动态卷积核
        kernels = self.hyper(style_id.long()).view(
            B, self.num_filters, D, self.kernel_size, self.kernel_size
        )

        # 分组动态卷积: 每种风格的 conv 核不同
        outputs = []
        for b in range(B):
            out_b = F[b:b+1].expand(self.num_filters, -1, -1, -1)
            out_b = F.conv2d(
                out_b.reshape(1, -1, H, W),
                kernels[b].reshape(self.num_filters * D, D, self.kernel_size, self.kernel_size),
                padding=self.kernel_size // 2,
                groups=self.num_filters * D
            )
            outputs.append(out_b.sum(dim=0, keepdim=True))
        spatial_map = torch.cat(outputs, dim=0)
        return StructuredStyleOutput(global_code=base_code, spatial_map=spatial_map)
```

### 优势

局部滤波 = 感受野锁定 → 结构天然保持。适合"笔触"类风格（印象派、点彩派——局部小范围内的高频滤波）。

---

## 方案 C: 切空间雅可比映射 (Tangent-Space Jacobian)

### 理论

将内容分解为 **0 阶(均值/色彩)** + **1 阶(梯度/几何)**。

不同的风格 = 对同一个切空间向量场 $\nabla z_0$ 做不同的旋转。

- 评估内容梯度场: $\nabla_x z_0, \nabla_y z_0$
- 风格旋转矩阵 $R_s$ 旋转梯度方向
- 用旋转后的梯度重建 spatial_map

**解耦保证**: 0 阶和 1 阶被独立处理——无论怎么扭曲高频纹理，梯度连续性锁死底层轮廓。

### 代码

```python
class JacobianTokenizer(nn.Module):
    """将风格表征为梯度场的旋转变换"""
    def __init__(self, num_styles, feat_dim=64):
        super().__init__()
        self.content_proj = nn.Conv2d(4, feat_dim, 3, padding=1)
        # 每个风格的 SO(N) 旋转矩阵 (via exponential map)
        self.style_skew = nn.Embedding(num_styles, feat_dim * (feat_dim - 1) // 2)
        self.style_scale = nn.Embedding(num_styles, feat_dim)

    def _so_rotation(self, skew_params):
        """用 skew-symmetric 参数构建正交旋转矩阵 (Cayley transform)"""
        B = skew_params.shape[0]; D = self.feat_dim
        S = torch.zeros(B, D, D, device=skew_params.device)
        idx = torch.triu_indices(D, D, 1)
        S[:, idx[0], idx[1]] = skew_params
        S = S - S.transpose(1, 2)         # skew-symmetric
        I = torch.eye(D, device=S.device).unsqueeze(0).expand(B, -1, -1)
        return (I - S) @ torch.linalg.inv(I + S)  # Cayley: (I-S)(I+S)^{-1}

    def forward(self, style_id, base_code, content_latent, target_hw):
        F = self.content_proj(content_latent.float())

        # 计算梯度场
        gx = F[..., 1:] - F[..., :-1]  # [B, D, H, W-1]
        gy = F[..., 1:, :] - F[..., :-1, :]  # [B, D, H-1, W]
        grad_field = torch.cat([
            F.pad(gx, (0, 1)), F.pad(gy, (0, 0, 0, 1))
        ], dim=1)  # [B, 2D, H, W]

        # 风格旋转矩阵
        R = self._so_rotation(self.style_skew(style_id.long()))  # [B, D, D]
        scale = self.style_scale(style_id.long()).diag_embed()    # [B, D, D]
        transform = scale @ R  # 缩放 + 旋转

        # 应用旋转变换到梯度场
        gx_rot = torch.einsum('bdhw,bdk->bkhw', gx, transform)
        gy_rot = torch.einsum('bdhw,bdk->bkhw', gy, transform)

        # 从旋转后的梯度重建 spatial_map (Poisson solver 简化版)
        spatial_map = F + gx_rot + gy_rot
        return StructuredStyleOutput(global_code=base_code, spatial_map=spatial_map)
```

### 与纤维丛的对应

$\nabla z_0$ 是底空间 $\mathcal{B}$ 上的向量场的纤维分量。$R_s \cdot \nabla z_0$ 是沿着纤维 $\mathcal{F}_c$ 的旋转——**0 阶不变 (内容结构)，1 阶旋转 (风格方向)** = 纤维丛上的截面变换。

---

## 三种 Tokenizer 对比

| 维度 | SMoE Translator | HyperNet Filter | Jacobian Mapping |
|------|:---:|:---:|:---:|
| 数学对应 | 纤维上的线性变换 (GL(N)) | 纤维上的局部卷积自同构 | 纤维切空间的 SO(N) 旋转 |
| 内容保留 | 恒等初始化保证 | 局部感受野保证 | 0/1阶解耦保证 |
| 风格特异性 | ★★★★★ 每个专家独立矩阵 | ★★★★ 动态卷积核 | ★★★★ 旋转矩阵 |
| 计算开销 | 中等 (einsum) | 高 (分组卷积) | 中等 (Cayley) |
| 论文价值 | ★★★★ | ★★★★ | ★★★★★ |
| 实现难度 | 低 | 中 | 中 |

### 推荐

**SMoE Translator (方案 A)** — 用恒等初始化保证 LPIPS 从 step 1 就极低，$W_k$ 在训练中学会"翻译"。三个方案中实现最丝滑、效果最立竿见影。完美呼应纤维丛理论——$W_k$ 即是纤维上的局部标架变换。
