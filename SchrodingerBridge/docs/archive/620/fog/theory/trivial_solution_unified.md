# 平凡解突破的统一数学理论：从 Loss 景观到架构约束

> 本文档建立平凡解形成的统一数学框架，分析"为什么保守策略是 loss 最优"的深层原因，
> 并提出可证伪的突破路径与实验验证方案。

---

## 1. 核心问题重述

### 1.1 现象：保守策略的反复出现

6个月实验反复验证同一个模式：**模型总是在某个维度上选择保守策略**。

| 时间 | 保守维度 | 量化指标 | 尝试的修复 | 结果 |
|------|---------|---------|-----------|------|
| 01月 | Cross-attn softmax ≡ 均匀 | aent ≈ log(N) | 换 sparsemax/relu2 | 部分改善 |
| 04月 | InstanceNorm 白化 | WFI ↑ | 移除 IN | 换地方保守 |
| 05月 | Heuristic loss 膨胀 | black-dot | 阻尼/退火 | 更保守 |
| 06月 | Gate Collapse | gate → 0.048 | 增大 gate_init | 收敛到同一值 |
| 06月 | Endpoint Shrinkage | α ≈ 0.16 | FiLM endpoint head | 部分改善 |
| 06月 | WFI随训练恶化 | 0.39 → 0.47 | 各种正则 | 越训越保守 |

**核心洞察**：每次解决一个保守问题，模型就在另一个维度重新选择保守。
这不是偶然——**在当前训练目标下，保守策略确实是 loss 最优的**。

### 1.2 问题的数学表述

设模型参数为 θ，训练目标为最小化：

$$
\mathcal{L}(\theta) = \mathbb{E}_{x,y,s,t}\left[ \ell(v_\theta(x_t, t, s), v_{\text{target}}, y_{\text{proj}}) \right]
$$

其中 ℓ 是单样本损失（FM + SWD + edge + ...）。

**平凡解**指存在一个"保守流形" $\mathcal{M}_0$，使得：
1. $v_\theta(x, t, s) \approx v_0(x, t)$（与 style 无关，或 style 影响极小）
2. $\|v_0\| \ll \|v_{\text{target}}\|$（位移很小）
3. $\theta \in \mathcal{M}_0$ 是局部最优（梯度为零或指向流形内）

**本文要回答**：
1. 为什么 $\mathcal{M}_0$ 是局部最优？（loss 景观分析）
2. 架构如何强化了这个局部最优？（架构约束分析）
3. 如何打破这个局部最优？（突破策略）

---

## 2. Loss 景观分析：为什么平凡解是吸引子

### 2.1 简化模型：单步、单风格

先考虑最简单的情况：固定 $x, y, t$，只看速度 $v$ 对 loss 的影响。

$$
L(v) = w_{\text{FM}} \cdot \|v - v^*\|_2^2 + w_{\text{SWD}} \cdot \text{SWD}(x_t + (1-t)v, y_{\text{proj}})
$$

其中 $v^* = v_{\text{target}} = y_{\text{proj}} - x$。

#### FM Loss 的景观

FM loss 是强凸二次函数：
- 全局最小值在 $v = v^*$
- Hessian = $2 w_{\text{FM}} \cdot I$，处处正定
- 梯度处处指向最小值

如果只有 FM loss，模型会直接学到 $v = v^*$，不会有平凡解。

#### SWD Loss 的景观

SWD loss 的景观要复杂得多。根据 SWD 梯度平坦性定理：

**定理（SWD 梯度平坦性）**：
当投影值的排序排列不变时，$\nabla_v \text{SWD} = C$（常数向量）。

这意味着 SWD 在排序稳定区内是**线性的**，不是凸的。其梯度是常数，不随 $v$ 变化。

**关键推论**：
- SWD 没有自己的"最小值点"——它只是在排序稳定区内提供一个恒定方向的推力
- 这个推力的方向和大小取决于当前排序与目标排序的差异
- 如果模型已经在排序稳定区内，SWD 梯度不会随优化而改变

#### 联合 Loss 的景观

联合 loss 的梯度：

$$
\nabla L(v) = 2 w_{\text{FM}} (v - v^*) + w_{\text{SWD}} (1-t) \cdot C(v)
$$

其中 $C(v)$ 是 SWD 梯度，在排序稳定区内为常数。

**临界点条件** $\nabla L(v^c) = 0$：

$$
v^c = v^* - \frac{w_{\text{SWD}} (1-t)}{2 w_{\text{FM}}} \cdot C(v^c)
$$

现在关键问题：$C(v)$ 指向什么方向？

### 2.2 SWD 梯度方向的统计分析

考虑 batch 中有多个样本、多个风格。对每个样本 $i$，有自己的 $x_i, y_i, s_i$。

SWD 梯度 $C_i$ 的方向统计性质：

1. **跨样本平均**：$\bar{C} = \frac{1}{N} \sum_i C_i$
   - 如果不同样本的 $C_i$ 方向分散，平均后模长变小
   - 在多风格训练中，不同风格的目标方向不同，$C_i$ 方向互相抵消

2. **跨投影方向平均**：SWD 对 $K$ 个随机投影方向取平均
   - 投影方向越多，梯度方向越"平均化"
   - 极端情况：$K \to \infty$ 时，梯度接近最优传输方向，但计算成本高

**核心发现**：在多风格训练中，SWD 的 batch 平均梯度 $\bar{C}$ 模长很小，因为不同风格的方向互相抵消。

这意味着：
- 对 batch 平均而言，SWD 提供的"风格推力"很弱
- FM loss 主导了优化方向
- 模型优先满足 FM（接近 $v^*$），SWD 只是微小修正

但等等——如果 $v^*$ 本身就有风格信息，为什么模型学不到？

### 2.3 条件期望坍缩：多风格的平均化效应

这是关键。考虑 style 条件下的期望：

$$
\bar{v}(x, t) = \mathbb{E}_{s \sim p(s)} [v_\theta(x, t, s)]
$$

模型有两种"保守策略"：

**策略 A：条件恒等（完全保守）**
- $v_\theta(x, t, s) = v_0(x, t)$（完全与 s 无关）
- 平均 FM loss: $\mathbb{E}_s[\|v_0 - v^*_s\|^2] = \|v_0 - \bar{v}^*\|^2 + \text{Var}_s(v^*_s)$
- 最优 $v_0 = \bar{v}^* = \mathbb{E}_s[v^*_s]$

**策略 B：条件依赖（理想情况）**
- $v_\theta(x, t, s) = v^*_s$（完美拟合每个 style）
- 平均 FM loss: $\mathbb{E}_s[\|v^*_s - v^*_s\|^2] = 0$

显然策略 B 的 loss 更低。那为什么模型不学策略 B？

**答案：架构约束和优化难度**。

策略 B 要求模型：
1. 从 style 输入中提取有用信息
2. 将 style 信息注入到特征中
3. 为每个 style 生成不同的 velocity

这些都需要"额外的容量"和"正确的梯度路径"。如果架构不支持（或梯度路径被阻塞），模型就会"偷懒"选择策略 A。

### 2.4 平凡解的吸引盆理论

**定义（平凡解流形）**：
设 $\mathcal{M}_0 = \{ \theta : v_\theta(x,t,s) \approx \bar{v}(x,t) \}$ 是"style-无关"的参数流形。

**定理（平凡解是局部最优的条件）**：
如果以下条件成立，则 $\mathcal{M}_0$ 是局部最优流形：

1. **FM 主导条件**：$w_{\text{FM}} \gg w_{\text{SWD}} \cdot \|C\| \cdot (1-t) / \|v^*\|$
   - FM loss 的权重远大于 SWD 的有效推力
   - 模型优先满足 FM

2. **Style 梯度衰减条件**：style→velocity 的雅可比矩阵范数很小
   - $\|\partial v_\theta / \partial s\| \ll \|\partial v_\theta / \partial x\|$
   - style 信号对输出的影响远小于 content 信号
   - 架构原因：gate 小、cross-attn 均匀、GN 压缩等

3. **SWD 平坦条件**：在 $\mathcal{M}_0$ 附近，SWD 梯度近似常数
   - 排序稳定，SWD 不提供额外的 style 方向梯度
   - 模型没有动力离开 $\mathcal{M}_0$

**证明概要**：
- 条件 1 保证 FM 主导优化方向
- 条件 2 保证 style 方向的梯度分量很小
- 条件 3 保证 SWD 不会在 style 方向提供额外推力
- 三个条件合起来：$\mathcal{M}_0$ 附近的梯度都指向流形内部
- 因此是局部最优流形。∎

**实验证据**：
- gate=0.048（条件 2：style 梯度被 gate 衰减）
- cos_sim(v(s1), v(s2)) = 0.9995（接近 $\mathcal{M}_0$）
- cross_attn_entropy = 6.24（接近均匀，条件 2）
- endpoint_alpha = 0.16（位移很小，条件 1+3）

---

## 3. 架构约束：模型如何主动选择保守

Loss 景观提供了平凡解的"可能性"，但架构决定了模型是否"容易"落入平凡解。

### 3.1 五层保守机制

我们识别出五层架构上的"保守机制"，每一层都让平凡解的吸引盆更大、更深。

#### 第 1 层：Style Gate（总开关）

$$
\text{style\_delta} = \tanh(g) \cdot \text{CA}(x, S)
$$

其中 $g$ 是可学习标量，初始 ~0.3，但所有实验都收敛到 ~0.048。

**为什么 gate 会收敛到小值？**

考虑 gate 对 loss 的影响：
- gate 大 → style 注入多 → velocity 大 → FM loss 可能更大（如果偏离了 FM 的最优方向）
- gate 小 → style 注入少 → velocity 小 → 更接近保守解

当满足以下条件时，小 gate 是最优的：
1. style 注入的方向与 FM loss 的最优方向不完全一致
2. FM loss 的权重足够大
3. SWD 的收益不足以补偿 FM 的损失

**量化**：设 gate 为 g，最优 gate 满足：

$$
\frac{\partial \mathcal{L}}{\partial g} = 0 \Rightarrow g^* = \arg\min_g \Big( w_{\text{FM}} \|\bar{v} + g \cdot \Delta v_s - v^*\|^2 + w_{\text{SWD}} \cdot \text{SWD}(g) \Big)
$$

如果 $\Delta v_s$ 与 $(v^* - \bar{v})$ 的夹角大（style 注入方向不对），则 $g^*$ 小。

**实验验证**：增大 gate_init 不改变最终 gate 值——模型会把 gate 调回小值。
这说明小 gate 确实是 loss 最优的。

#### 第 2 层：Cross-Attention 均匀化

Cross-attention 的 softmax 输出接近均匀分布：

$$
\text{CA}(x, S) \approx \frac{1}{N} \sum_i V(S_i) = \bar{V}
$$

这意味着：
1. 不同 style 输入产生几乎相同的 CA 输出（都是平均）
2. Style 信息在 attention 层就被"平均掉"了
3. 后续层接收到的 style 信号已经很弱

**为什么 attention 会均匀？**

- 1-token style 情况下，softmax 自然 = 1.0（只有一个 token）
- 多 token 情况下，如果 Q 和 K 的匹配度差异小，softmax 就均匀
- InstanceNorm / GroupNorm 会压缩特征差异，使 QK^T 的差异更小

#### 第 3 层：GroupNorm 动态范围压缩

网络中的 GroupNorm（尤其是 endpoint head 的 GroupNorm(1) = InstanceNorm）：

$$
\text{GN}(h)_{b,c} = \frac{h_{b,c} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

作用：
1. 强制每个样本的特征均值为 0，方差为 1
2. 压缩了跨样本的动态范围
3. 削弱了 style 注入的效果——即使 style 改变了特征，GN 又把它"拉回来"

**对平凡解的强化**：
- GN 使输出分布更"平均"→ 更接近 $\mathcal{M}_0$
- GN 压缩动态范围 → endpoint 位移更小 → α 更低
- GN 在 endpoint head 中最有害，因为直接影响输出

#### 第 4 层：零初始化 / 小初始化

多个投影层使用零初始化或极小初始化（std=1e-3 ~ 0.02）。

后果：
- 初始时 model ≈ identity（恒等映射）
- Loss 景观在 identity 附近有局部最优
- 训练从 identity 开始，容易卡在保守盆地
- 如果保守解附近的 loss 梯度很小，模型就"走不出去"

**这是优化问题，不是局部最优问题**：
即使非保守解的 loss 更低，如果初始点在保守盆地内，且盆地边缘的梯度很小，模型也可能走不出去。

#### 第 5 层：Target Projection 的低频锚定

在 `source_low_target_high` 模式下，低频被锚定在 source：

$$
y_{\text{proj, low}} = x_{\text{low}}
$$

这意味着：
- 低频目标速度 = 0
- 模型的低频输出自然接近 0
- 低频占图像能量的大部分 → 整体位移看起来很小

这本身不是"保守"，而是设计选择。但如果锚定过强，会抑制风格迁移的视觉效果。

### 3.2 乘积效应：为什么修复一个没用

五种机制的效应是**相乘**的，不是相加：

$$
\alpha_{\text{total}} = \alpha_{\text{gate}} \cdot \alpha_{\text{attn}} \cdot \alpha_{\text{norm}} \cdot \alpha_{\text{init}} \cdot \alpha_{\text{proj}}
$$

每个因子都在 (0, 1] 范围内。

当前估计：
- $\alpha_{\text{gate}} \approx 0.1$（gate=0.048，tanh 后 ~0.048，对比 gate=1）
- $\alpha_{\text{attn}} \approx 0.3$（条件期望坍缩，不同 style 抵消 70%）
- $\alpha_{\text{norm}} \approx 0.7$（GN 压缩动态范围 30%）
- $\alpha_{\text{init}} \approx 0.8$（初始化限制了训练初期的探索）
- $\alpha_{\text{proj}} \approx 0.95$（低频锚定的影响较小）

乘积：$0.1 \times 0.3 \times 0.7 \times 0.8 \times 0.95 \approx 0.016$

这比观测到的 α=0.16 小，因为各机制之间有非线性耦合（不是完全独立的）。
但数量级是对的：保守机制的叠加效应是毁灭性的。

**关键推论**：
- 只修复一个机制，效果会被其他机制乘回去
- 例如：把 gate 从 0.048 提到 0.3（×6），如果 attn 还是 0.3，总提升只有 ×6
- 需要多机制同时突破，才能跳出平凡解

---

## 4. 突破策略：从数学到工程

### 4.1 突破的数学条件

要跳出平凡解，需要打破"平凡解是局部最优"的三个条件：

| 条件 | 打破方式 | 对应方案 |
|------|---------|---------|
| FM 主导 | 降低 FM 权重，或增强 style loss | Endpoint-supervised、SWD 增权 |
| Style 梯度衰减 | 增强 style→output 的梯度路径 | FiLM-only、去 gate |
| SWD 平坦 | 让 SWD 梯度在 style 方向更强 | Style strength 正则、两阶段训练 |

**突破的充分条件**：
至少打破两个条件，且剩余条件的乘积 < 1。

更精确地说，设突破后各因子为 $\alpha'_i$，需要：

$$
\prod_i \alpha'_i > \alpha_{\text{threshold}}
$$

其中 $\alpha_{\text{threshold}}$ 是"非平凡解"的阈值（如 > 0.5）。

### 4.2 三层干预策略

#### 第一层：架构去安全阀（最高 ROI）

**目标**：增大 style→output 的梯度路径强度，打破"style 梯度衰减条件"。

**方案 A1：去 Endpoint Head GroupNorm**
- 成本：1-2 行代码
- 预期：α_norm 从 0.7 → 0.9（提升 ~30%）
- 风险：训练不稳定，动态范围爆炸
- 验证：1 epoch smoke test，看 WFI 和 α

**方案 A2：FiLM-only 注入（移除 gate）**
- 成本：配置开关 + 少量代码
- 预期：α_gate 从 0.1 → 0.8（提升 ~8x）
- 风险：过度注入、内容崩溃
- 验证：1 epoch smoke test，看 clip_style 和 LPIPS

**组合效应**：如果 A1 和 A2 独立，总提升 = 0.7→0.9 × 0.1→0.8 = 1.03→10.3 倍
当然不会完全独立，但即使有耦合，提升也应该很显著。

#### 第二层：训练目标重构（范式级改变）

**目标**：打破"FM 主导条件"，让 style loss 直接优化 endpoint 质量。

**方案 B1：Endpoint-supervised 训练**
- 直接优化 endpoint 的 content + style，不经过 velocity 中转
- 消除 Training-Output Mismatch
- 预期：从根本上改变 loss 景观
- 成本：中等（新 loss 实现 + 调试）
- 风险：训练不稳定

**方案 B2：Style Strength 正则化**
- 在 loss 中奖励"朝 style 方向移动"
- 直接给 style 方向一个梯度推力
- 预期：额外的 style 方向梯度，帮助跳出平凡解
- 成本：低（几十行代码）
- 风险：过度移动、内容崩溃

#### 第三层：训练策略优化（辅助手段）

**目标**：帮助优化器跳出保守盆地。

**方案 C1：两阶段训练**
- Stage 1：高 SWD / 低 FM → 强制 style 注入
- Stage 2：正常权重 → 微调内容平衡
- 类比：先把球踢出去，再调整准度

**方案 C2：课程学习**
- 从大 style strength 开始，逐渐降低
- 帮助模型先学会"使用 style"，再学"精细控制"

### 4.3 实验验证顺序（按 ROI 排序）

| 顺序 | 方案 | 预期收益 | 实现成本 | 风险 |
|------|------|---------|---------|------|
| 1 | 去 Endpoint Head GN | 中（WFI↓ 0.03+） | 极低（1行） | 低 |
| 2 | FiLM-only 注入 | 高（gate 从 0.05→~1） | 低 | 中 |
| 3 | 1+2 组合 | 高（协同效应） | 低 | 中 |
| 4 | Style Strength 正则 | 中高 | 中 | 中 |
| 5 | Endpoint-supervised | 最高（范式改变） | 高 | 高 |
| 6 | 两阶段训练 | 中 | 中 | 低 |

---

## 5. 可证伪预测与验证实验

### 5.1 架构类预测

**P-1（去 GN 预测）**：
- 假设：Endpoint head 中的 GroupNorm 压缩了动态范围，导致 WFI 升高
- 预测：移除 GN 后，WFI 下降 > 0.03，endpoint_alpha 上升 > 0.05
- 验证：1 epoch smoke test，对比有 GN / 无 GN
- 反例：如果 WFI 不变或上升，则 GN 不是白化主因

**P-2（FiLM-only 预测）**：
- 假设：Style gate 是模型控制保守程度的旋钮，且当前最优在小 gate
- 预测：移除 gate（FiLM-only）后，clip_style 提升 > 0.01，cross_attn_entropy 下降
- 验证：1 epoch smoke test，对比 gated / film_only
- 反例：如果 clip_style 不变，说明瓶颈不在 gate 而在别处

**P-3（组合预测）**：
- 假设：各保守机制是乘积效应，组合收益 > 单独收益之和
- 预测：去 GN + FiLM-only 的组合效果 > 单独去 GN + 单独 FiLM-only
- 验证：对比三组实验的指标
- 反例：如果组合效果 = 单独效果之和（线性叠加），则机制独立；如果 < 单独效果，则有冲突

### 5.2 训练目标类预测

**P-4（Endpoint-supervised 预测）**：
- 假设：Training-Output Mismatch 是保守策略的根因之一
- 预测：Endpoint-supervised 模式下，endpoint_alpha 提升 > 20%，且训练早期 style 注入更快
- 验证：对比 velocity mode 和 endpoint mode 的训练曲线
- 反例：如果 endpoint mode 反而更差，说明 velocity 监督有稳定化作用

**P-5（Style Strength 正则预测）**：
- 假设：直接奖励 style 方向位移可以帮助跳出平凡解
- 预测：开启 style_strength_reg 后，endpoint_alpha 提升 > 20%，LPIPS 恶化 < 10%
- 验证：对比开启/关闭 style_strength_reg
- 反例：如果 LPIPS 恶化 > 20%，说明奖励不加区分地鼓励"乱动"

### 5.3 训练策略类预测

**P-6（两阶段预测）**：
- 假设：单阶段训练容易卡在保守盆地，两阶段可以先跳出再微调
- 预测：两阶段训练的最终效果 > 相同总 epoch 数的单阶段训练
- 验证：对比 2 epoch 单阶段 vs 1+1 两阶段
- 反例：如果效果相同，说明保守盆地不是训练策略问题

---

## 6. 结论与展望

### 6.1 核心结论

1. **平凡解不是 bug，是特征**：在当前的训练目标和架构下，保守策略确实是 loss 最优的。
2. **五层乘积效应**：gate × attention × norm × init × proj，共同导致 α ≈ 0.16。
3. **单一修复无效**：只改一个地方，会被其他机制乘回去。
4. **架构去安全阀是最高 ROI**：去 GN 和 FiLM-only 都是低成本高收益的改动。
5. **训练目标重构是根本解**：Endpoint-supervised 从根本上改变 loss 景观。

### 6.2 下一步行动

1. **立即验证**：去 Endpoint Head GN（成本最低，预期收益明确）
2. **快速跟进**：FiLM-only 注入模式（已有代码，主要是验证）
3. **组合测试**：去 GN + FiLM-only（看协同效应）
4. **范式升级**：如果组合还不够，再上 Endpoint-supervised 训练
5. **精细调优**：Style strength 正则 + 两阶段训练

### 6.3 开放问题

1. 去 GN 后训练稳定性如何？需要其他正则化来替代吗？
2. FiLM-only 模式下内容保持会崩溃吗？需要多强的 content loss？
3. Endpoint-supervised 训练需要什么样的 content loss？低频 L1 够吗？
4. 多风格泛化的"平均化效应"有什么根本解法？per-style 专家混合？
5. 当前的 WFI 指标能准确衡量"白化"吗？需要补充感知指标吗？

---

## 7. 代码索引

| 机制 | 代码位置 |
|------|----------|
| Style gate | [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) `style_gate` |
| Cross-attention | [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) `_attention_stats` |
| GroupNorm (trunk) | [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) `norm1`, `norm2` |
| GroupNorm (endpoint head) | [model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) `FiLMEndpointHead` |
| FiLM (pre/post CA) | [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) `film_q_proj`, `film_proj` |
| FM loss | [losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) `w_flow` |
| SWD loss | [losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) `single_step_swd_weight` |
| Target projection | [losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) `_project_training_target` |
| Endpoint alpha 计算 | [trainer.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/trainer.py) debug metrics |
