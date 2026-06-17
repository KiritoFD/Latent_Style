# OT 匹配的理论本质与修正路径

> 不涉及代码细节，聚焦数学本质和设计决策

---

## 一、OT 在风格迁移中的角色

最优传输解决的核心问题：给定一个 batch 的内容图 $\{x_i\}_{i=1}^B$ 和目标风格图 $\{y_j\}_{j=1}^B$，
找到配对的传输计划 $\Pi_{ij}$，最小化总代价 $\sum_{ij} \Pi_{ij} \cdot C(x_i, y_j)$。

然后 $y_{\text{matched}, i} = \sum_j \Pi_{ij} \cdot y_j$ 成为训练中"内容图 $x_i$ 应该变成什么"的目标。

---

## 二、为什么当前的 OT 退化为平凡解

### 2.1 代价度量的失效

**当前代价**: $C(x_i, y_j) = \| x_i - y_j \|_2^2$（潜空间的欧氏距离）

**为什么失效**: 潜空间中，"一只写实猫"和"一幅印象派风景画"的欧氏距离，不等于它们的结构差异。
欧氏距离被颜色面积（亮度、对比度）主导，而非语义结构。结果是 OT 把颜色中庸的图像当作"枢纽"——
大量源图被匹配到同几个目标 → 目标多样性坍缩 → 网络学到的是"怎样变成那几张平均图"。

### 2.2 底层数学: 度量空间错配定理

**定理（非正式）**: 在无配对风格迁移中，如果代价函数 $C(x, y)$ 只依赖潜空间欧氏距离，
则存在一个严格正的枢纽概率 $P(\text{hub}) > 0$，使得当 batch 中目标风格分布的支撑集
与内容分布的结构多样性不匹配时，OT 计划退化为 Many-to-One 映射。

**证明直觉**: 在高维空间中（$\dim=4 \times 64 \times 64 = 16384$），欧氏距离的集中现象
(concentration of measure) 意味着大多数点对之间的距离非常接近。OT 的解在这种情况下
由少数"离群点"（更亮/更暗/更饱和的图像）主导——它们成为其他所有点的最近邻。

### 2.3 实验证据

从 debug.md 观察到的 SMoE 训练数据:
- `plan_entropy` 趋势显示 OT 匹配的多样性随时间衰减
- `tok_delta=0.0187` 说明 tokenizer 几乎不做变换——因为 loss 信号被 $\mathcal{H}$ 分量主导
- I2SB orthogonal e1: style=0.705 但 LPIPS=0.447——说明即使 I2SB 的正交约束也没有阻止
  水平分量的泄漏

---

## 三、结构感知 OT: 为什么比欧氏 OT 好

### 3.1 核心假设

两张图像在**风格上**应该匹配，当且仅当它们在**结构复杂度上**相似。

- 结构复杂的源图（城市、树林）→ 应匹配结构复杂的目标（细节丰富的画作）
- 结构简单的源图（天空、水面）→ 应匹配结构简单的目标（大色块、平涂）

这不需要外部先验（如 DINO）——结构复杂度可以从纯潜空间的 self-affinity 矩阵推导。

### 3.2 Gromov-Wasserstein 视角

GW 距离不比较点值，而是比较**内部距离结构**:

$$C^{\text{GW}}_{ijkl} = \left| d(x_i, x_j) - d(y_k, y_l) \right|^2$$

直觉: "如果源图的两块区域 A 和 B 的潜空间距离是 0.5，
找目标图中两块距离也是 0.5 的区域来匹配"。
这叫**拓扑同构传输 (Topologically Isomorphic Transport)**。

### 3.3 现有实现的状态

代码中已有多达 **7 种**结构代价模式 (`coupling_structure_cost_mode`):

| 模式 | 含义 |
|------|------|
| `self_affinity_gw` | 对原始潜变量做 self-affinity |
| `lowedge_self_affinity_gw` | 低频+边缘的混合 descriptor |
| `encoder_self_affinity_gw` | 用 UNet encoder 特征做 self-affinity |
| `tokenizer_aux_self_affinity_gw` | 用 tokenizer 的 aux 特征 |
| `tokenizer_entropy_affinity_gw` | 用 tokenizer 的路由熵做结构指纹 |
| `encoder_hybrid_affinity_gw` | encoder 特征 + 低频/边缘混合 |
| `tokenizer_aux_hybrid_affinity_gw` | tokenizer aux + 低频/边缘混合 |

默认配置: `coupling_cost_composition = "appearance_plus_structure"`, `coupling_structure_cost_weight = 1.0`.

---

## 四、为什么结构 OT 理论上应该工作但没有

### 4.1 诊断假设

最可能的解释: **结构代价的尺度问题**。

结构代价（基于 self-affinity 矩阵的 cdist²）和外观代价（基于潜空间欧氏距离）的尺度可能差几个数量级。
即使设 weight=1.0（50/50），如果两者没有正确归一化，其中一个会主导总代价。

从 `_coupling_cost_matrix` 代码中可以看到归一化逻辑:
```python
app_scale = appearance_cost.detach().mean()
struct_scale = structure_cost.detach().mean()
total_cost = (1-weight) * appearance_cost / app_scale + weight * structure_cost / struct_scale
```

这个归一化确保了两者在 **mean 尺度**上相当。但:
- 如果结构代价的 **方差**远大于外观代价，归一化只对均值有效
- 在 self-affinity 矩阵中，对角元素（同一张图与自身的 affinity）可能主导
- 这导致结构代价矩阵退化为"近似恒等"或"近似随机"

### 4.2 替代解释

也可能是 **self_affinity 描述符本身对潜变量的区分度不足**。
在 4 通道的 VAE latent 上，两张不同风格但结构相似的图的 self-affinity 可能高度接近。
这种情况下，结构代价实际上退化为常数矩阵 → OT 回到纯外观匹配。

### 4.3 如何验证

不需要改代码。观察 `ot_structure_cost_var` 指标:
- 如果 $\text{Var}(C_{\text{structure}}) \approx 0$ → 结构描述符退化 → 确认 4.2
- 如果 $\text{Var}(C_{\text{structure}}) \gg 0$ 但 Gini 仍然高 → 确认 4.1（归一化不足）

---

## 五、推荐修正路径

### 路径 A: Tokenizer 熵指纹 (最小改动)

`tokenizer_entropy_affinity_gw` 模式用 tokenizer 的路由熵作为结构描述符。
路由熵天然编码了空间复杂度——复杂区域的 entropy 高，平坦区域的 entropy 低。

**为什么这可能是最优解**: 
- 不依赖 UNet encoder（避免额外前向传播，且 encoder 被 velocity 训练目标污染）
- 天然与 tokenizer 的 cluster 路由耦合——"软路由"的像素比"硬路由"的像素更"结构复杂"
- 每个像素由一个 K 维向量（attention weight）表示，比 4 维潜变量更具判别性

**验证**: 设置 `coupling_structure_cost_mode="tokenizer_entropy_affinity_gw"`，
观察 `ot_target_gini` 是否 ≤ 0.4（低于 0.6 的安全红线）。

### 路径 B: 分层 OT (Per-Cluster OT)

616/design.md 提出的方案: 按 tokenizer 的 K 个 cluster 分别做 OT。

**数学**: 对每个 cluster k:
$$\Pi^{(k)} = \arg\min_{\Pi} \sum_{i,j} \Pi_{ij} \cdot C(\text{Mask}_k \odot x_i, \text{Mask}_k \odot y_j)$$

**优势**: 天空的 OT 只在天空区域比较，水面的 OT 只在水面区域比较——避免了"天空笔触 vs 建筑轮廓"的跨语义匹配。

### 路径 C: 非平衡 OT (Unbalanced Sinkhorn)

允许部分概率质量被丢弃:
$$\min_{\Pi} \langle \Pi, C \rangle + \epsilon H(\Pi) + \tau_1 D_{KL}(\Pi \mathbf{1}, \mu) + \tau_2 D_{KL}(\Pi^T \mathbf{1}, \nu)$$

**优势**: 如果目标库里没有合适的匹配（16 张源图中有 3 张是极简风格但目标库里只有巴洛克作品），
Unbalanced OT 允许这些源图"放弃匹配"而不是强行拉拽到最近的（不合适的）目标。
避免噪声梯度。

代码中已有 `sinkhorn_use_unbalanced` 标志和相关参数。

---

## 六、与垂直流匹配的互补关系

结构感知 OT 和垂直流匹配是互补的，但解决不同的根本问题：

| 问题 | 解决方案 | 机制 |
|------|----------|------|
| OT 配对不准 → 平凡解 | 结构感知 OT (路径 A/B) | 改进传输计划的**目标** |
| 水平分量污染速度场 | 垂直流匹配 (`bridge_path_mode="vertical"`) | 改进速度场的**方向** |

**两者应该叠加使用**: 结构 OT 找对配对目标 → 垂直 FM 确保只用纤维分量训练 → tokenizer 得到干净的风格信号。

---

## 七、实验验证路线

### 今天: 验证结构 OT 是否有效

1. 设置 `coupling_structure_cost_mode = "tokenizer_entropy_affinity_gw"`
2. 设置 `coupling_cost_composition = "appearance_plus_structure"`
3. 设置 `coupling_structure_cost_weight = 0.5` (给外观留空间)
4. 用现有 topogate ckpt 作为 warmstart，训练 4-6 epochs
5. 观察 `ot_target_gini`: 应 < 0.6
6. 观察 `ot_structure_cost_var`: 应 > 0（不退化）
7. 观察 `plan_entropy`: 应稳定不衰减

### 明天: 叠加垂直 FM

将验证通过的结构 OT 配置 + `bridge_path_mode="vertical"` 一起训练。

### 如果 Gini 仍然高

说明 tokenizer_entropy 描述符也不能区分足够的结构 → 需要路径 B（分层 OT）或
使用 UNet encoder 特征。

---

## 八、总结

**OT 效果不好的根因**: 不是理论错了，不是实现错了——是**默认参数可能让结构代价退化**。
self_affinity_gw 在 4 通道 VAE latent 上的区分度可能不足，归一化只做了 mean 对齐但没做 variance 对齐。

**最优先的修改**: `coupling_structure_cost_mode` 从 `self_affinity_gw` 改为 `tokenizer_entropy_affinity_gw`，
用 tokenizer 的路由熵（比 4 维潜变量更丰富的结构描述符）做 GW 匹配。

**是否需要同时做垂直 FM**: 是。结构 OT 修配对，垂直 FM 修速度场——两者互补。
