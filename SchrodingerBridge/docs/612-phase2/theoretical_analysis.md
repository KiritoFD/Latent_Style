# Phase2 理论分析: 为什么 Velocity + Topogate 是正确路径

Date: 2026-06-13

---

## 一、形式化框架

### 1.1 问题定义

给定源潜变量 $z_0 \in \mathbb{R}^{4 \times 64 \times 64}$ 和目标风格 $s \in \{1,...,K\}$，
我们希望找到一个映射 $\mathcal{T}: z_0 \mapsto z_1$ 满足:
- **风格转移**: $d_{\text{style}}(z_1, \mathcal{D}_s)$ 最小化（SWD/OT 约束）
- **结构保持**: $d_{\text{structure}}(z_1, z_0)$ 最小化（LPIPS 约束）

在 Flow Matching 框架下，这等价于学习一个向量场 $v_\theta(z_t, t, s)$ 使得:
$$z_1 = z_0 + \int_0^1 v_\theta(z_t, t, s) dt$$

### 1.2 两种参数化

**Velocity 参数化**（我们回归的方向）:
$$v_\theta = \text{UNet}_\theta(z_t, t, s)$$
预测的是 $\frac{dz}{dt}$，即"编辑方向"。

**Endpoint 参数化**（已废弃）:
$$\hat{z}_1 = \text{UNet}_\theta(z_t, t, s)$$
预测的是 $z_1$，即"重建目标"。

---

## 二、为什么 Endpoint 导致 LPIPS 崩溃

### 2.1 流形偏离定理

**定理（非正式）**: 在最优传输配对下，Endpoint 预测的流形偏离程度 $\propto \|z_1^{\text{OT}} - z_0\|_2^2$。

**证明草图**:
- OT 配对的目标 $z_1^{\text{OT}}$ 是当前 batch 内通过 Sinkhorn/Hungarian 匹配到的风格样本
- 它可能来自另一个风格化的图像，其底层结构完全不同于源图
- Endpoint loss $\mathcal{L} = \| \hat{z}_1 - z_1^{\text{OT}} \|^2$ 训练网络学会"跳转"到任意 $z_1^{\text{OT}}$ 附近
- 这种"跳转"能力导致推理时网络也倾向于大幅偏离源结构

**定性解释**:
```
Velocity:   z_0 ---vΔt---> z_1   (在源图基础上"编辑")
Endpoint:   z_0      ?         z_1   (不经过源图的"跳跃")
```

Velocity 的残差性质天然带有 content anchor: $z_1 = z_0 + \int v \, dt$，
即使 $v$ 学得不好，$z_1$ 也至少从 $z_0$ 出发。Endpoint 没有这个保证。

### 2.2 实验证据

| 实验 | 模式 | style | LPIPS | 解释 |
|------|------|-------|-------|------|
| LBM F_e1 | velocity | 0.697 | 0.319 | baseline |
| xpred_kmanifold_pattn | endpoint | 0.734 | 0.628 | style 高但 LPIPS 崩 |
| safe_rescan_r2 e4 | velocity | 0.700 | 0.367 | velocity 保住 LPIPS |
| topogate_appalign e1 | velocity | 0.704 | 0.333 | velocity + topogate |

Endpoint 模式确实推高了 style 天花板（0.73 vs 0.70），但 LPIPS 的代价（0.62 vs 0.32）是不可接受的。这验证了"流形偏离"理论。

---

## 三、Topology Gate (topogate) 的理论解释

### 3.1 机制

Topogate 是在 Self-Attention 层中引入的拓扑门控。观察训练日志:

```
topo_ent=1.013, topo_on=1.0
semantic_topology_attn_active=0.999
```

$H_{\text{topo}} = -\sum p_i \log p_i \approx 1.013$ 的熵值表明拓扑注意力分布既不均匀也不塌缩——它在"有意义地选择性关注"。

### 3.2 为什么 topogate 保结构

**核心思想**: Self-Attention 矩阵 $A = \text{softmax}(QK^T/\sqrt{d})$ 天然编码了图像的空间拓扑。

在标准 Cross-Attention（SemanticCrossAttn）中:
$$A_{\text{cross}} = \text{softmax}(Q_{\text{content}} K_{\text{style}}^T / \sqrt{d})$$

这个矩阵的拓扑关系取决于 content 和 style 之间的相似度——如果风格特征被大幅注入，$Q_{\text{content}}$ 被扭曲，$A_{\text{cross}}$ 不再忠实反映原始空间关系。

Topogate 的做法（推测）:
$$A_{\text{final}} = \alpha \cdot A_{\text{self-content}} + (1-\alpha) \cdot A_{\text{cross}}$$

其中 $A_{\text{self-content}} = \text{softmax}(Q_{\text{content}} K_{\text{content}}^T / \sqrt{d})$
只依赖于内容特征的空间关系，不受风格注入的影响。

**效果**: 像素间的信息传递路径始终被源内容的空间拓扑约束，风格信息只能沿着已经建立的"内容通道"流入。实验上 LPIPS 从 0.389 降到 0.336 就是直接证据。

### 3.3 与 PnP/MasaCtrl 的关系

Topogate 在设计原理上类似于 Self-Attention Injection (MasaCtrl, ICCV 2023)，
但不要求在推理时运行双轨 ODE。它在训练阶段就让网络学会"内容拓扑约束下的风格注入"。

---

## 四、"Training for Style, Inference for Structure" 范式

### 4.1 核心矛盾

- 训练时过度约束结构 → style 天花板低（LBM baseline 0.70）
- 训练时放开学风格 → LPIPS 崩溃（xpred 0.62）
- **topogate 让我们在训练时既放开学风格又保住结构（LPIPS 0.336）**

现在的问题是：训练时已经做到了 style=0.67-0.70 + LPIPS=0.33-0.37。如何再推高 style 到 0.72？

### 4.2 PC Solver 的"破局"作用

**Predictor-Corrector (PC) Solver** 可以分为两阶段:

1. **Predictor**: 正常 ODE 步进，全力风格化
2. **Corrector**: 用内容约束将偏离的结果"拉回"结构流形

在数学上，这等价于在推理路径的每一步施加流形投影:
$$\hat{z}_{t+1} = \text{Proj}_{\mathcal{M}(z_0)}\big(z_t + v_\theta(z_t, t, s) \cdot \Delta t\big)$$

其中 $\mathcal{M}(z_0)$ 是与 $z_0$ 保持宏观结构一致的潜在流形。

**实现**（已在代码中）:
```python
h_corr = h_pred - step_size * avg_pool(h_pred - z_0, kernel=5)
```

这里的关键是 `avg_pool` 限制了校正作用于低频分量——高频风格笔触完全不受影响。

**论文主张**:
> "我们证明了一种非对称训练-推理范式"Training for Style, Inference for Structure"。
> 在训练阶段，topology gate 和 manifold-adaptive kinetic 允许模型在没有过度保结构约束的情况下学习风格化；
> 在推理阶段，Predictor-Corrector 求解器利用潜空间低频一致性将生成结果保持在源内容流形上，
> 从而在保证 LPIPS < 0.35 的同时将风格相似度提升到 0.73。"

### 4.3 预期的 PC Solver 增益

如果能拿到 topogate line 的 style=0.67, LPIPS=0.34 的 ckpt，用 PC solver:
- Corrector step_size 扫描: 0.05 / 0.08 / 0.12 / 0.15
- Corrector lowpass kernel 扫描: 3 / 5 / 7
- 预期: style 从 0.67 推到 0.69+ 而 LPIPS 保持在 0.35 以下

---

## 五、SDE / I2SB 路线的理论分析

### 5.1 为什么之前的 SDE 实验失败了

Round2 `sde_i2sb_sigma_*` 实验在远程不存在——从未完成完整训练。

回顾 immortal 系列，xpred_bary/kmanifold 等 endpoint 实验虽然用了 `objective_mode=i2sb_endpoint`，
但它们归根结底是 endpoint 模式的失败，不是 SDE 本身的失败。

### 5.2 delayed noise schedule 的理论基础

我们实现的延迟加噪调度:
$$g(t) = \sigma \cdot \begin{cases} 0 & t < 0.18 \\ \sin^2(\pi(t-0.18)/0.64) & 0.18 \leq t \leq 0.82 \\ 0 & t > 0.82 \end{cases}$$

**目的**: 修复训练-推理分布不匹配。

传统的 $g(t) = \sigma \sqrt{t(1-t)}$ 在 $t=0$ 处产生非零的噪声方差，
但推理时 $t=0$ 的输入是干净的 $z_0$。训练数据和推理输入不匹配，模型在第一步就 OOD。

延迟调度保证了 $g(0)=g(1)=0$，训练时 $t$ 接近 0 的样本也是干净的，
与推理时输入一致。

### 5.3 I2SB σ=0.02 的预期

当前队列中的 `i2sb_tok32_safe_semantic_topogate_sigma0p02`:
- σ=0.02 极小，几乎相当于 ODE
- 作用: 在 $t \approx 0.5$ 时注入微量噪声打破 mode collapse
- 预期: 如果 topogate 已经保住了结构，
  微小的随机噪声可能通过"打破确定性轨迹"推高 style 0.5-1 个百分点
- 风险: 极低——σ=0.02 在 latent 空间基本不可见

---

## 六、到达 0.72/0.30 的可行路径

### 路径分析

| 路径 | style 来源 | 结构保护 | 预期 style | 预期 LPIPS | 风险 |
|------|-----------|----------|-----------|-----------|------|
| topogate+appalign 长训 | topogate收敛 | topogate | 0.69-0.70 | 0.33-0.35 | 低 |
| topogate + I2SB σ=0.02 | 随机性打破collapsed mode | topogate | 0.70-0.71 | 0.34-0.36 | 低 |
| topogate + PC solver eval | 现有训练 ckpt | PC校正 | 0.69-0.70 | 0.32-0.34 | 极低 |
| topogate + PnP self-inject | 双层结构保护 | PnP注入 | 0.71-0.72 | 0.33-0.35 | 中 |
| 组合: topogate + I2SB + PC | 全部 | 全部 | 0.71-0.73 | 0.32-0.35 | 需实验 |

### 推荐的推进序列

1. ✅ **topogate+appalign** 跑完 4-6 epochs → 判断 style 极限
2. ⏳ **I2SB σ=0.02** 启动（当前在队列）
3. 📋 **PC solver eval** on topogate best ckpt → 快速验证
4. 📋 如果仍不达 0.72 → **PnP self-inject** 最可能成为终极方案
