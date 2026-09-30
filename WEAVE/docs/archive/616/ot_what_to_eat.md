# OT 匹配应该吃什么信息 — 从第一性原理出发

> 风格迁移中 OT 的角色不是"找到最相似的图"，而是"找到对当前模型来说最容易风格化的配对目标"。
> 这改变了对代价矩阵设计的整个思路。

---

## 一、重新定义 OT 在风格迁移中的角色

### 1.1 传统视角 (错误)

$$C_{ij} = \|x_i - y_j\|^2$$

"找和我的内容图最像的目标图" → 颜色相似 → 退化为亮度匹配 → 平凡解。

### 1.2 正确视角

$$C_{ij} = \text{"如果把内容图 } x_i \text{ 风格化为目标 } y_j \text{，网络学到的速度场有多干净？"}$$

OT 的目标不是找"最像的"，而是找**"对当前的 transport 模型来说，学习梯度最平滑、结构污染最小的配对"**。

这意味着代价矩阵应该反映的是 **transport 的困难度匹配**，而不是图像的视觉相似度。

---

## 二、Transport 困难度的定义

考虑两张内容图:
- $x_A$: 建筑照片，大量垂直边缘，精细结构
- $x_B$: 天空照片，大面积平滑渐变，几乎没有结构

考虑两张目标风格图:
- $y_1$: 洛可可风格，极度复杂，到处都是曲线和细节
- $y_2$: 极简主义，大面积平涂色块

**直观**:
- $x_A \to y_1$ 的 transport: 结构已经复杂 → 目标也复杂 → 速度场只需要改变纹理方向 → 容易
- $x_A \to y_2$ 的 transport: 结构复杂 → 目标简单 → 速度场需要"抹平"结构 → 困难，产生巨大 $\mathcal{H}$ 分量
- $x_B \to y_1$ 的 transport: 结构简单 → 目标复杂 → 速度场需要在空间中生成新结构 → 极其困难
- $x_B \to y_2$ 的 transport: 两者都简单 → 速度场几乎只是少量纹理变化 → 最容易

**好的匹配**: $x_A$ 配 $y_1$（都复杂），$x_B$ 配 $y_2$（都简单）。

**代价矩阵应该编码的**: 不是"这两张图像不像"，而是"这对 (content, target) 的网络学习难度"。

## 三、如何度量 transport 困难度

### 3.1 结构复杂度

图像的"结构复杂度"可以通过以下指标近似:

**频率域**: 高频能量占比 → 细节丰富的图像高频能量高
**空间域**: 边缘密度 → 建筑照片边缘多，天空边缘少
**信息论**: TopoGate attention 的熵 → 注意力分布越均匀，结构越复杂

这些指标不依赖 tokenizer 的质量，不依赖 encoder 的前向传播。它们是图像潜变量本身的统计性质。

### 3.2 内容-目标的结构差异

对于一对 (content, target)，transport 的困难度 $D(x, y)$ 可以近似为:

$$D(x, y) \approx |\text{complexity}(x) - \text{complexity}(y)|$$

原因: 如果复杂度假差大，网络需要"抹平"或"生成"结构 → $\mathcal{H}$ 分量大 → 学习困难。

更精确的度量:

$$D(x, y) = \text{KL}\left(P_{\text{structure}}(x) \| P_{\text{structure}}(y)\right)$$

即两张图像结构分布的 KL 散度。$P_{\text{structure}}$ 可以是 TopoGate 注意力熵的空间分布、Laplacian 金字塔的系数分布、或潜变量的空间自相关函数。

### 3.3 与垂直 FM 的关系

注意: 我们已经在 `bridge_path_mode="vertical"` 中强制了 $\mathcal{H}$ 分量为 0。
这意味着**在垂直 FM 下，结构差异不应该影响 transport 质量**——因为结构是锁定的。

但 OT 配对仍然重要: 即使结构锁定，一个内容简单目标复杂的配对会让网络"凭空生成"纹理，而一个内容复杂目标复杂的配对只是"改变"纹理。前者更难学习。

**垂直 FM + 好的 OT = 结构锁定 + 纹理自然对齐**。

## 四、具体的信息源与匹配策略

### 4.1 已有的、可靠的信息源

| 信息源 | 类型 | 计算成本 | 可靠性 |
|--------|------|:---:|:---:|
| 潜变量 self-affinity | 结构 GW 描述符 | 低 ($O(n^2)$ 在 downsampled tokens) | ✅ 已验证 |
| TopoGate attention 熵 | 结构复杂度 | 零 (forward 中计算) | ✅ 不依赖 tokenizer |
| 潜变量统计 (mean, std, edge) | 混合 | 极低 | ✅ baseline |
| tokenizer 输出 | 伪结构 | 中等 | ❌ 已确认无效 |
| encoder 特征 | 结构 | 中等 (额外 forward) | ✅ 但成本高 |

### 4.2 推荐的三层代价矩阵

**第一层: 结构复杂度匹配 (TopoGate attention)**

```python
def structural_complexity(x):
    # 从模型获取 TopoGate attention (零成本)
    attn_maps = model.get_last_topogate_attention()
    # 计算每个 attention 头的熵
    entropy = - (attn_maps * attn_maps.clamp_min(1e-8).log()).sum(dim=-1)
    # 空间统计: [mean, std, skewness, max_location_entropy]
    return torch.stack([
        entropy.mean(dim=(1,2)),
        entropy.std(dim=(1,2)),
        (entropy > entropy.median()).float().mean(dim=(1,2)),
        entropy.amax(dim=(1,2)),
    ], dim=-1)  # [B, 4]

# 代价: 复杂度差异
C_struct = torch.cdist(complexity(x_content), complexity(y_target), p=2).pow(2)
```

**第二层: 潜变量 self-affinity (当前默认)**

在 downsampled 潜变量 tokens 上计算 self-attention triu → 紧凑结构描述符。
这是已实现的 `self_affinity_gw` 模式，不需要模型 forward。

**第三层: 混合代价**

```python
C_total = (1 - w_struct) * C_appearance / scale_appearance + w_struct * C_structure / scale_structure
```

其中 $w_{\text{struct}} = 0.3-0.5$（给结构复杂度 30-50% 权重）。

### 4.3 配对的期望行为

良好的匹配应该呈现:
1. **低 Gini 系数** ($< 0.4$): 没有枢纽现象
2. **高 plan 熵**: 配对多样化
3. **结构复杂度相关性**: 内容复杂度与配对目标的复杂度正相关
4. **训练稳定**: 速度场范数不出现 spike

---

## 五、全新的实验设计

去掉依赖 tokenizer 的实验，保留内生结构描述符的实验:

| 实验 | bridge_path | coupling_cost_composition | coupling_structure_cost_mode | 测试 |
|------|------------|---------------------------|------------------------------|------|
| h0 | vertical | structure_only | self_affinity_gw | baseline (当前默认) |
| h1 | linear | structure_only | self_affinity_gw | 垂直 FM 效果 |
| h2 | vertical | appearance_only | — | 欧氏 OT 对照 |
| h3 | vertical | structure_only | self_affinity_gw + sigma=0.02 | SDE 噪声 |
| h4 | vertical | structure_only | self_affinity_gw + unbalanced | 非平衡 OT |
| h5 | vertical | appearance_plus_structure | **topogate_attention_gw** | **新: TopoGate attention 结构代价** |
| h6 | vertical | appearance_plus_structure | topogate_attention_gw + unbalanced + sigma | 全组合 |

**关键改动**: h5 从 `tokenizer_entropy_affinity_gw` 改为 `topogate_attention_gw`。后者不依赖 tokenizer，从 TopoGate attention 中提取结构指纹。

---

## 六、文档更新待办

需要修改的 616 文档:
1. `ot_theory.md` — 添加第 4 节，写清楚 OT 应该吃什么信息
2. `ot_structure_redesign.md` — 补充 TopoGate attention descriptor 的详细实现
3. `README.md` — 更新当前状态和推荐实验
4. `module_diagnosis.md` — 确认 tokenizer 相关依赖已标记删除
5. `experiment_landscape.md` — 更新实验表格
