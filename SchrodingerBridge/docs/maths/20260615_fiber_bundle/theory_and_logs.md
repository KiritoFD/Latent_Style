# 风格纤维丛与随机流桥：数学设计与实验反思日志 (2026-06-15)

## 1. 风格纤维丛的微分几何形式化 (Mathematical Formalization of Style Fiber Bundles)

### 1.1 潜空间上的纤维丛定义
将 VAE 潜空间 $\mathcal{Z} \subset \mathbb{R}^{C \times H \times W}$ 建模为底空间（内容流形）$\mathcal{B}$ 上的**纤维丛** $E = (\mathcal{B}, \mathcal{F}, \pi)$：
- **底空间 $\mathcal{B}$**：表示内容特征的拓扑和几何布局（形状、边缘、宏观布局）。我们通过投影 $\pi: E \to \mathcal{B}$ 来锁定并提取该底空间。
- **纤维 $\mathcal{F}_c = \pi^{-1}(c)$**：在给定内容结构 $c \in \mathcal{B}$ 下，所有可能的风格画法和纹理表现所构成的子流形。
- **投影算子 $\pi$**：提取底空间坐标。在我们的网络中，由 TopoGate（自注意力拓扑门控）锁定。
- **纤维方向**：在保持底空间坐标 $c$ 不变的情况下，在纤维 $\mathcal{F}_c$ 内部移动的切向量方向（即“不改结构改外观”）。

### 1.2 埃雷斯曼联络与 TopoGate (Ehresmann Connection via TopoGate)
在纤维丛上，我们定义一个埃雷斯曼联络 (Ehresmann Connection)，它将切丛 $TE$ 分解为水平分布 $\mathcal{H}$ 和垂直分布 $\mathcal{V}$：
$$T_x E = \mathcal{H}_x \oplus \mathcal{V}_x$$
其中垂直分布 $\mathcal{V}_x = \ker(d\pi_x)$ 是切于纤维的方向，而水平分布 $\mathcal{H}_x$ 决定了底空间在纤维间的平行移动。
**TopoGate** 正是这个联络的物理算子化实现：
$$A_{\text{final}} = \alpha \cdot A_{\text{self-content}} + (1-\alpha) \cdot A_{\text{cross-style}}$$
当 $\alpha \to 1.0$ 时，联络强力约束切向量完全局限于垂直分布 $\mathcal{V}_x$ 内，即强制 $\Delta c = 0$。这就解释了为什么 TopoGate 能将 LPIPS 稳定锁定在 $\approx 0.31$（极接近无操作 IDT 的水平）。

### 1.3 确定性 ODE 的均值坍缩定理 (ODE Mean Collapse)
**定理**：如果传输轨迹 $x_t$ 遵循确定性常微分方程（ODE），在损失函数（如 MSE 或单图 SWD）约束下，极限点满足条件期望：
$$\lim_{t \to 1} x_t = \mathbb{E}[X \mid c]$$
由于在纤维 $\mathcal{F}_c$ 上可能对应无数种艺术风格的画法（如 Impressionism 的笔触位置可以有无限种偏置），最小化 MSE 导致确定性模型最终收敛于所有可能画法的“期望平滑笔触”（即平滑塑料色块）。这是 ODE 无论如何训练，其 style 极限都卡在 $\approx 0.70$ 的数学根本原因。

### 1.4 随机微分方程的边界可达性 (Fiber-aligned SDE)
为了打破均值坍缩，必须在纤维方向引入随机各向异性布朗运动。
定义各向异性 Fiber-SDE：
$$dx_t = v_\theta(x_t, t, s) dt + \sigma(t) \cdot G_{\text{topo}}(x_t) \odot dW_t$$
- $G_{\text{topo}}(x_t)$ 是基于注意力熵的局部拓扑门控。
- 在边缘处（熵低），$G \to 0$，噪声消失，保护内容边界不受布朗运动侵蚀。
- 在纹理处（熵高），$G \to 1$，允许沿纤维方向注入最大噪声，从而使生成轨迹触及风格分布的支持边界。

---

## 2. 核心架构设计改造 (Core Architectural Improvements)

### 2.1 Tokenizer：从“查表”到“翻译”的连续几何映射
- **旧方案**：`PureLatentSpatialTokenizer` 通过注意力路由将像素映射到离散的 cluster $k$（查表法），输出为固定的基向量 $V_k$。这完全丢弃了特征空间的连续变化和局部几何信息。
- **新方案：SMoE Translator Tokenizer (空间混合专家翻译器)**：
  $$\text{Output}(x) = \sum_k \alpha_k(x) \cdot (W_k \cdot F_{\text{content}}(x))$$
  其中 $W_k$ 是局部标架变换矩阵（对应底空间到风格纤维的局域坐标翻译）。
- **恒等初始化**：$W_k = I + \Delta W_k$，当训练开始时 $\Delta W_k = 0$，模型以最纯净的内容特征做热启动，极大地保护了初始结构。

### 2.2 Loss：分层 SWD (Fiberwise SWD)
传统 SWD 忽略了空间位置的语义相关性。分层 SWD 按照专家的注意力权重进行局部概率测度匹配：
$$\mathcal{L}_{\text{SWD}} = \sum_k \text{SWD}\left( \text{Mask}_k \odot z_1, \; \text{Mask}_k \odot z_{\text{style}} \right)$$
这保证了“天空的纤维”只与“天空的风格”相匹配，“眼睛的纤维”只与“眼睛的风格”相匹配，避免了空间跨越导致的质地混乱。

---

## 3. 实验结果日志与分析反思 (Experimental Logs & Reflections)

### 3.1 确定性 ODE 极限与 SMoE 翻译器瓶颈分析
- **实验 `aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1`** (SMoE 专家数=32，15 epoch 收敛)：
  - **Epoch 1**: Style = 0.6724 / LPIPS = 0.3332 (all-pairs style = 0.7035 / LPIPS = 0.3297)
  - **Epoch 8**: Style = 0.6699 / LPIPS = 0.3178 (all-pairs style = 0.7019 / LPIPS = 0.3153)
  - **Epoch 15**: Style = 0.6713 / LPIPS = 0.3336 (all-pairs style = 0.7022 / LPIPS = 0.3304)
  - **反思**：即便引入了保持局部几何的 SMoE 翻译器，由于在推理时采用的是**确定性 ODE** 求解器，模型依然强烈受到均值坍缩定理的控制。Style 指标被死死锁在 0.70 左右，无法突破。
- 实验 `topogate_appalign` 也呈现出高度一致的指标（0.6714 style / 0.314 LPIPS），再次交叉印证了**确定性 ODE 的条件期望坍缩**是阻碍 style 上升的主导数学力量。

### 3.2 推理期 SDE / PC 求解器扫描（廉价验证实验）
- **在 `k070 epoch_0003` 亲本上进行推理期 `Fiber-SDE` 扫描**：
  - $\sigma = 0.08$ 纤维对齐噪声：Style 达到 0.6811，LPIPS 稍微升至 0.3391。
  - $\sigma = 0.08$ 各向同性噪声：All-pairs Style 达到 0.7107，LPIPS 升至 0.3368。
  - **反思**：注入随机噪声确实抬升了 style 指标，这印证了 SDE 能够向外扩散以逃逸均值吸引子的假设。但由于**模型在训练时是基于确定性流设计的**，突如其来的推理期噪声与模型的学习模式有一定的不匹配，导致 LPIPS 上升且风格增量依然未能触及 0.73 的瓶颈。
- **推理期 `PC Solver` 结构校正扫描（纠偏低频）**：
  - step = 0.10：LPIPS 降至 0.3117，但 Style 也微跌了 0.0007。
  - **反思**：PC 求解器是强力保结构手段（推理期的 Ehresmann 联络投影），应该将其作为 SDE 训练释放风格后的“保底安全网”，而不应指望在没有风格能量的模型上仅靠 PC 提升风格。

### 3.3 决策树从头训练 (High-pass + Phase Envelope SWD) 实验进展
- 目前正在运行 `decision_tree_highpass_run` (对应 `task-285`)，第一轮正从头训练。
- 该实验取消了 `resume_checkpoint`，直接从头建立骨干网络。
- 核心改变：
  - `transport_high_strength = 0.3`（相比 W34 的 0.02 极大释放了高频）。
  - `swd_abs_highpass_weight = 1.0` 且 `swd_signed_highpass_weight = 0.0`（利用绝对包络 SWD 匹配宏观风格，防止 signed phase 产生的无意义结构扰动）。
- 这属于利用**绝对高频匹配数学机制**去拓宽通道容量的探索。

---

## 4. 下一步大刀阔斧的实验路线 (Next Milestone Directions)
为了彻底解决 “LPIPS < 0.30” 的严苛限制，同时将 “Style 推到 > 0.73”，我们接下来必须将理论全链路合一：
1. **SMoE Translator + Fiberwise SWD + SDE (I2SB/Unsb-Cycle) 联合从头训练**：
   - 之前只进行了 SMoE 翻译器的 ODE 训练（导致均值坍缩），以及在 ODE ckpt 上的推理期 noise 注入（导致 OOD 不匹配）。
   - 正确解法：在训练时就加入各向同性/纤维对齐 SDE 噪声（如 `i2sb_endpoint` 或 `solver_unsb_cycle`），配合延迟加噪调度（delayed noise schedule），让网络学会随机降噪。
2. **多尺度 TopoGate 分级控制 (Multi-scale Topogate)**：
   - 宏观低频（8x8, 16x16）：TopoGate 强度设为 1.0（锁定内容流形布局）。
   - 微观高频（32x32, 64x64）：TopoGate 强度降至 0.3-0.5，释放局域风格的随机扩散。
