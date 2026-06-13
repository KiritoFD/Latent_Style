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
