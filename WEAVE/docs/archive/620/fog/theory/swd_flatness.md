# SWD梯度平坦性定理

## 1. 定义与符号

### 1.1 Sliced Wasserstein Distance

$$
\mathrm{SWD}(P, Q) = \frac{1}{K} \sum_{k=1}^{K} W_1\!\left(\mathrm{proj}_{\mathrm{dir}_k}(P),\; \mathrm{proj}_{\mathrm{dir}_k}(Q)\right)
$$

其中 $W_1$ 是一维 Wasserstein 距离：

$$
W_1(P_1, Q_1) = \frac{1}{N} \sum_{i=1}^{N} \bigl| p_{(i)} - q_{(i)} \bigr|
$$

其中 $p_{(i)}$ 和 $q_{(i)}$ 是排序后的样本。

### 1.2 SWD的梯度

对于单个样本 $p_i$：

$$
\frac{\partial \mathrm{SWD}}{\partial p_i} = \frac{1}{K} \sum_{k} \mathrm{dir}_k \cdot \mathrm{sign}\!\left(p_i \text{在排序后的位置与} q \text{对应位置的差}\right)
$$

更精确地：

$$
\frac{\partial \mathrm{SWD}}{\partial p_i} = \frac{1}{KN} \sum_{k} \mathrm{dir}_k \cdot \mathrm{sign}\!\left(p_{(\sigma(i))} - q_{(\sigma(i))}\right)
$$

其中 $\sigma$ 是排序排列。

## 2. 定理：排序不变性 $\Rightarrow$ 梯度为零

**定理 1（SWD梯度平坦性）**：设 $P, Q \in \mathbb{R}^N$ 有 $N$ 个样本。若存在 $\varepsilon > 0$ 使得对所有 $\|\delta\| < \varepsilon$，$\mathrm{sort}(P + \delta)$ 的排列与 $\mathrm{sort}(P)$ 相同，则 $\nabla \mathrm{SWD}(P, Q) = 0$ 在 $P$ 的 $\varepsilon$-邻域内恒为常数。

**证明**：

1. SWD的值仅依赖于排序后的值。
2. 排序排列不变 $\Rightarrow$ 每个 $p_i$ 在排序中的位置不变。
3. 因此 $\mathrm{sign}\!\left(p_{(\sigma(i))} - q_{(\sigma(i))}\right)$ 是常数。
4. 所以 $\partial \mathrm{SWD} / \partial p_i = \text{常数}$。
5. 在 $P$ 的一个邻域内，$\nabla \mathrm{SWD}$ 是常数。 $\square$

**推论 1**：若 $\nabla \mathrm{SWD}(P, Q)$ 在 $P$ 处为 $0$，则 $P$ 是 $\mathrm{SWD}(P, Q)$ 的局部极小值。

## 3. 排序稳定性条件

**定理 2**：排序排列在扰动 $\delta$ 下不变的充分条件是：

$$
\|\delta\|_\infty < \frac{1}{2} \min_{i \neq j} \bigl| p_{(i)} - p_{(j)} \bigr|
$$

即扰动幅度小于相邻排序值最小间距的一半。

**证明**：

- 排序排列改变当且仅当存在 $i, j$ 使得 $p_i + \delta_i > p_j + \delta_j$ 但 $p_i < p_j$。
- 这意味着 $\delta_i - \delta_j > p_j - p_i > 0$。
- 因此 $\max_i \delta_i - \min_j \delta_j > \min$ 相邻间距。
- 若 $2\|\delta\|_\infty < \min$ 相邻间距，则不可能发生。 $\square$

## 4. 数值估计

### 4.1 投影值的间距分布

对于 VAE 潜空间中的自然图像（$64 \times 64$，4 channels）：

- 投影值方差 $\approx \mathrm{Var}(\mathrm{pixel}) \approx 0.04$（归一化后）。
- 相邻排序值间距 $\approx \sigma_{\mathrm{proj}} / \sqrt{N} \approx 0.2 / \sqrt{4096} \approx 0.003$。

### 4.2 扰动幅度

在 $v = 0$ 附近，velocity 扰动 $\delta$ 产生的投影扰动：

$$
\|\mathrm{proj}(P + \delta) - \mathrm{proj}(P)\| = \|\mathrm{dir} \cdot \delta\| \leq \|\delta\|
$$

对于 $v_{\mathrm{target}} \approx 0.3\text{--}0.5$（典型值），相邻间距 $\approx 0.003$。

因此排序稳定性区域半径 $\approx 0.003$（在投影空间）。

## 5. 平凡解是局部极小值的条件

**定理 3**：联合 loss

$$
L(v) = \alpha \cdot \|v - v_{\mathrm{target}}\|^2 + \beta \cdot \mathrm{SWD}\!\left(x + (1 - t)v,\; y\right)
$$

中，$v = 0$ 是局部极小值的条件：

存在 $\delta > 0$ 使得对所有 $\|v\| < \delta$：

1. $\nabla \mathrm{SWD}\!\left(x + (1 - t)v,\; y\right) = C$（常数）。
2. $\displaystyle \|C\| < \frac{2\alpha \cdot \|v_{\mathrm{target}}\|}{\beta(1 - t)}$。

**证明**：

$$
\begin{aligned}
\nabla L(v) &= 2\alpha(v - v_{\mathrm{target}}) + \beta(1 - t) \cdot C \\
\nabla L(0) &= -2\alpha \cdot v_{\mathrm{target}} + \beta(1 - t) \cdot C
\end{aligned}
$$

若 $\|C\| < \dfrac{2\alpha \cdot \|v_{\mathrm{target}}\|}{\beta(1 - t)}$，则 $\nabla L(0)$ 方向与 $-v_{\mathrm{target}}$ 相同。

Hessian $= 2\alpha \cdot I > 0$。

因此 $v = 0$ 是局部极小值。 $\square$

## 6. 实验验证方案

通过 `probe_swd_gradient.py` 验证：

1. 计算 $\|\nabla \mathrm{SWD}(v = 0)\|$，验证其 $\approx 0$。
2. 测量排序稳定性：对 $v$ 加扰动 $\varepsilon$，统计排序变化率。
3. 扫描 loss landscape 沿 $v_{\mathrm{target}}$ 方向，验证在 $v \approx 0.16 \cdot v_{\mathrm{target}}$ 处有极小值。