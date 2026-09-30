# OT 结构代价设计修正: 从 Tokenizer 到 TopoGate Attention

> 当前问题: `coupling_structure_cost_mode` 的 `tokenizer_entropy_affinity_gw`、
> `encoder_self_affinity_gw` 等依赖 tokenizer 输出（已确认 PureLatentSpatial ZERO ROI）或 encoder（额外前向传播）。
> **OT 结构代价应该从 TopoGate 的内生 attention 矩阵中提取，零额外成本。**

## 关键决策

- PureLatentSpatial tokenizer 确认 ZERO ROI — style/LPIPS 不变，白耗 ~1.2GB VRAM
- 代码已切回 `legacy_factorized` + `ablation_disable_spatial_prior=true`
- OT 结构代价来源: `topogate_attention_gw`（TopoGate 内生 attention 矩阵）

---

## 一、为什么 TopoGate Attention 是理想的结构指纹

TopoGate 在 `SemanticCrossAttn` 中计算:
$$A_{\text{final}} = \alpha \cdot A_{\text{self-content}} + (1-\alpha) \cdot A_{\text{cross}}$$

其中 $A_{\text{self-content}} = \text{softmax}(Q_{\text{content}} K_{\text{content}}^T / \sqrt{d})$ 只依赖内容特征的内部空间关系。

这个矩阵天然编码了图像的**结构拓扑**——相邻像素有高注意力权重，跨语义边界有低权重。
这是训练过程中已经计算好的，不需要额外的前向传播。

### 1.1 为什么比 tokenizer 方案好

| 维度 | tokenizer_entropy | topogate_attention |
|------|---|---|
| 信号源 | tokenizer 路由输出（已确认垃圾） | UNet 自身的 attention 矩阵 |
| 成本 | tokenizer forward + aux 提取 | **零**（forward 中已计算） |
| 语义含义 | 无意义（ZERO ROI） | 内容特征的内部空间关系 |
| 训练耦合 | tokenizer 未训练好 → 代价矩阵垃圾 | 不受 tokenizer 训练影响 |
| 结构编码 | 间接（通过 cluster routing） | 直接（像素间注意力权重） |

## 二、新的结构代价模式

### 2.1 模式: `topogate_attention_gw`

从 TopoGate 的注意力熵图中提取结构复杂度画像，然后用 GW 框架匹配。

```python
# 伪代码
def topogate_attention_descriptors(model, content_batch, target_batch):
    """从 TopoGate attention 提取结构复杂度画像"""
    # 1. 收集 TopoGate attention maps（zero-cost，已在 forward 中计算）
    content_attn = model.body_blocks.get_topogate_attention(content_batch)
    target_attn = model.body_blocks.get_topogate_attention(target_batch)
    
    # 2. 对每个样本计算结构复杂度画像
    #    画像 = [熵均值, 熵标准差, 偏度, 高熵像素占比]
    content_profiles = compute_complexity_profile(content_attn)  # [B, 4]
    target_profiles = compute_complexity_profile(target_attn)    # [B, 4]
    
    # 3. GW-style 代价: 复杂度差异
    C_struct = torch.cdist(content_profiles, target_profiles, p=2).pow(2)
    
    return C_struct


def compute_complexity_profile(attn_maps):
    """从 attention maps 提取每个样本的结构复杂度画像"""
    # attn_maps: [B, H_heads, N_tokens, N_tokens] or [B, N, N]
    # 对最后一个维度做熵
    entropy = -(attn_maps * (attn_maps + 1e-8).log()).sum(dim=-1)  # [B, N]
    
    return torch.stack([
        entropy.mean(dim=1),              # 平均复杂度
        entropy.std(dim=1),               # 复杂性方差（结构均匀度）
        entropy.max(dim=1).values,        # 最大复杂度（最复杂区域）
        (entropy > entropy.median()).float().mean(dim=1),  # 复杂区域占比
    ], dim=-1)  # [B, 4]
```

### 2.2 与 total_cost 的整合

```python
# 在 _coupling_cost_matrix 中
if coupling_structure_cost_mode == 'topogate_attention_gw':
    C_appearance = transport_cost.pairwise_cost(content, target)
    C_structure = topogate_attention_descriptors(model, content, target)
    
    # 归一化到相同尺度
    app_scale = C_appearance.detach().mean()
    struct_scale = C_structure.detach().mean()
    
    total_cost = (1 - w_struct) * C_appearance / app_scale + w_struct * C_structure / struct_scale
```

**关键**: TopoGate 的 attention 不需要 tokenizer，不需要 encoder 全是内容特征的内部关系。且 `torch.no_grad()` 保证不计梯度。

## 三、实现计划

### 3.1 删除的依赖
- `tokenizer_entropy_affinity_gw` → 删除 (依赖垃圾 tokenizer)
- `tokenizer_aux_self_affinity_gw` → 删除
- `encoder_self_affinity_gw` → 降低优先级 (需要额外 forward)

### 3.2 新增的
- `topogate_attention_gw` — 从 TopoGate 收集 attention 矩阵，提取复杂度画像做 GW 匹配

### 3.3 保留的
- `self_affinity_gw` — 纯潜变量的 self-affinity，不依赖任何模块 (最快, 作为 baseline)
- `none` (fallback stats descriptor) — 纯统计量

## 四、与其他模块的配合

TopoGate attention 的优势: 它本来就是 TopoGate 机制的一部分。
如果 `semantic_self_topology_gate=false`，attention 退化为普通 self-attention——仍然有效作为结构指纹。
如果 `semantic_self_topology_gate=true`，attention 被 topogate 保护——结构信息更纯粹。

**这意味着**: 无论 TopoGate 开不开，attention 都是可用的结构特征源。

## 五、实验计划更新

当前实验 `exp/20250618_lite_ot_vertical/` 中:

| 实验 | 耦合结构代价模式 | 测试 |
|------|------|------|
| h5 | `topogate_attention_gw` | TopoGate attention 作为结构指纹 |
| h0-h4 | `self_affinity_gw` (潜变量, 不变) | baseline 对照 |

h5 的 `coupling_structure_cost_mode` 已从 `tokenizer_entropy_affinity_gw` 改为 `topogate_attention_gw`。
