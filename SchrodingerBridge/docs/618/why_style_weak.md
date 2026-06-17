# 为什么我们的 Style 这么差 — 深层根因

> 实验数据: 全部7组 style 0.66-0.67, LPIPS 0.29-0.30. H1(线性FM)反而最好.

---

## 一层: TopoGate blend=1.0 把 Style 彻底堵死了

**这是最主要的、最直接的原因。**

`semantic_self_topology_blend=1.0` 意味着 UNet 的 self-attention 层:
$$A_{\text{final}} = 1.0 \times A_{\text{self-content}} + 0.0 \times A_{\text{cross}}$$

**style 信号在 attention 层面被完全阻隔。** UNet 的每个像素只看自己周围的 content pixels, 绝不看 style features。不论 tokenizer 输出什么 spatial_map, 不论 OT 匹配多精准——style 信息在 attention 层就被截断了。

**证据**: H1(线性FM)反而最好(0.670) — 因为线性FM没有"结构冻结"的额外约束, 给 style 留了一点点自由度。垂直FM把底空间锁死 → style 完全出不来。

**这解释了为什么所有机制都无效**: 垂直FM、结构OT、SDE、非平衡OT——它们都在 attention 之前或之后操作。如果 attention 根本不看 style, 这些全部无效。

## 二层: Legacy tokenizer 几乎没有风格表征能力

当前 tokenizer: `legacy_factorized` — style values 是 `Embedding(5, style_dim)` 的查表。5个向量代表5种风格。

**这意味着 model 看到的"风格"只是5个固定向量**, 没有任何内容适应性、没有任何实例级风格信息。这个 tokenizer 甚至比之前被我们废弃的 PureLatentSpatial 更弱——后者至少有 content query → routing 机制。

配合 blend=1.0: **两处都被堵死了**。Attention层锁死 + tokenizer输出的style values太弱。

## 三层: 和对比方法的本质差距

| 方法 | 风格来源 | 我们 |
|------|---------|------|
| StyleShot | 从参考图编码 (多尺度MoE, Transformer) | style_id查表 (5个向量) |
| CSGO | Perceiver Resampler从参考图压缩 | style_id查表 |
| StyleGallery | 扩散特征聚类 + 区域匹配 | 无参考图 |
| SCSA | 语义mask引导的attention约束 | TopoGate blend=1.0(过度) |

**所有人都有风格参考图, 我们没有。** 这是最根本的设定差异。

在无参考图设定下, 风格必须完全来自训练数据中学到的类别表征。但我们的:
1. 训练数据只有5类 (且 Impressionism 13030 vs Minimalism 1307, 严重不平衡)
2. Tokenizer 只有5个可学习的 style embedding
3. 没有任何从"风格实例"中提取信息的机制

## 四层: 整个架构偏向内容保持

| 机制 | 效果 | 对style的影响 |
|------|------|-------------|
| TopoGate blend=1.0 | 完美结构保持 | 完全阻断style |
| Residual connections | 保留内容 | 稀释style变化 |
| Skip connections | 内容直通 | 绕过style注入 |
| Velocity prediction | 残差预测(Δx) | 默认锚定内容 |

**整个架构被设计为"在保证结构的前提下做最小的风格改变"**。

---

## 从相关工作能学到什么

### 学到的1: 风格表征必须生动

**StyleShot 的核心洞察**: 风格表征的质量决定了风格迁移的上限。CLIP encoder 不够好 → 需要专门的 Style-Aware Encoder。

**我们的问题**: 5 个 style_id embedding → 风格表征极度贫乏。**这是最根本的瓶颈, 比 TopoGate blend 更深层。**

**可做的**: 从 OT 匹配后的 `matched_target` 中提取风格特征 → 注入 tokenizer。这样即使没有参考图, tokenizer 也能看到"具体的风格实例"。

### 学到的2: 结构约束必须有度

**SCSA 的启示**: 硬约束 (G1/G2 $-\infty$ mask) 和软约束 (blending) 应该在**不同层用不同强度**。所有层都用最强约束 = 过度压制。

**我们的问题**: blend=1.0 在所有层 → 全局过度约束。

**可做的**: 多尺度 blend — 粗尺度 (8×8) blend=1.0 (保大局), 细尺度 (64×64) blend=0.2 (放笔触)。

### 学到的3: 无参考图设定下, 数据量和分布是关键

**StyleShot 的 StyleGallery 数据集**: 风格均衡、多样。LAION 只有 7.7% stilized → 训练效果差。

**我们的问题**: 5 类, Impressionism 13030 vs Minimalism 1307 (10:1 不平衡), 且没有文本描述、没有风格标注。

**可做的**: 扩大风格类别, 平衡采样, 或使用 WikiArt 更多类 (如 27 类)。

---

## 突破路径优先级

1. **降 blend**: 0.4/0.5/0.6 sweep — 当前最快的验证
2. **多尺度 blend**: 不同层不同强度
3. **matched_target style encoding**: 让 tokenizer 从实际风格图像中学 (中等代码量)
4. **数据扩充**: 更多风格类别, 均衡分布 (需要数据处理)
