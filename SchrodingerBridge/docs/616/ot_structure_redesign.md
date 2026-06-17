# OT 结构代价设计修正: 从 Tokenizer 到 TopoGate Attention

> 当前问题: `coupling_structure_cost_mode` 的 `tokenizer_entropy_affinity_gw`、
> `encoder_self_affinity_gw` 等依赖 tokenizer 输出（已确认无收益）或 encoder（额外前向传播）。
> **OT 结构代价应该从 TopoGate 的内生 attention 矩阵中提取，零额外成本。**

---

## 一、为什么 TopoGate Attention 是理想的结构指纹

TopoGate 在 `SemanticCrossAttn` 中计算:
$$A_{\text{final}} = \alpha \cdot A_{\text{self-content}} + (1-\alpha) \cdot A_{\text{cross}}$$

其中 $A_{\text{self-content}} = \text{softmax}(Q_{\text{content}} K_{\text{content}}^T / \sqrt{d})$ 只依赖内容特征的内部空间关系。

这个矩阵天然编码了图像的**结构拓扑**——相邻像素有高注意力权重，跨语义边界有低权重。
这是训练过程中已经计算好的，不需要额外的前向传播。

## 二、新的结构代价模式

在 `_structure_pairwise_cost` 中新增:

```python
# 模式: "topogate_self_affinity_gw"
# 从 TopoGate 的 last_attn 中提取 self-affinity descriptor

def _topogate_attention_descriptor(self, model, x):
    # 1. 跑一次 forward 获取 TopoGate attention
    with torch.no_grad():
        # 获取 content encoder 特征 (复用 _ot_encoder_feature_map 的轻量版本)
        feat = self._topogate_content_feature_map(model, x)
        # 在 model 的 body_blocks 中收集 last_attn  
        # 这些 attention 矩阵已经在 TopoGate 中计算好了
        attn_maps = self._collect_topogate_attention_maps(model, feat)
    
    # 2. 从 attention maps 构建结构 descriptor
    # attention map 本身就是空间关系的编码
    # 对 attention map 做 self-affinity = GW descriptor
    return self._affinity_descriptor_from_attention(attn_maps)
```

**关键**: TopoGate 的 attention 不需要 tokenizer，不需要 encoder 全是内容特征的内部关系。且 `torch.no_grad()` 保证不计梯度。

## 三、实现计划

### 3.1 删除的依赖
- `tokenizer_entropy_affinity_gw` → 删除 (依赖垃圾 tokenizer)
- `encoder_self_affinity_gw` → 降低优先级 (需要额外 forward)
- `tokenizer_aux_self_affinity_gw` → 删除

### 3.2 新增的
- `topogate_attention_gw` — 从 TopoGate 收集 attention 矩阵，做 self-affinity

### 3.3 保留的
- `self_affinity_gw` — 纯潜变量的 self-affinity，不依赖任何模块 (最快, 作为 baseline)
- `none` (fallback stats descriptor) — 纯统计量

## 四、与其他模块的配合

TopoGate attention 的优势: 它本来就是 TopoGate 机制的一部分。
如果 `semantic_self_topology_gate=false`，attention 退化为普通 self-attention——仍然有效作为结构指纹。
如果 `semantic_self_topology_gate=true`，attention 被 topogate 保护——结构信息更纯粹。

**这意味着**: 无论 TopoGate 开不开，attention 都是可用的结构特征源。

## 五、实验计划更新

当前 7 个实验中，`h5_token_entropy` 依赖已删除的 tokenizer entropy 模式。
替换为:

| 实验 | 新的耦合结构代价模式 | 测试 |
|------|------|------|
| h5 | `topogate_attention_gw` | TopoGate attention 作为结构指纹 |
| h0-h4 | `self_affinity_gw` (潜变量, 不变) | baseline 对照 |

需要改 `gen_lite_batch.py`: h5 的 `coupling_structure_cost_mode` 从 `tokenizer_entropy_affinity_gw` 改为 `topogate_attention_gw`。
