# math.md — 风格迁移的数学基础: 纤维丛、垂直流与均值坍缩

> 本文件给出整个方案所依赖的核心数学. 不引用不证明的命题一律标注"假设".
> 实验数字取自仓库内 `S-add__K-1_C-0_W-20_Col-0`, `aaai2027/round1_*`, `docs/612-lookback/analysis.md`, `docs/618/why_style_weak.md` 中已沉淀的真实结果.
> 612/616/619/model 的几何推导只保留被实验或严格论证支持的部分, 其余作为"历史假设"标注.

---

## 0. 记号约定

| 符号 | 含义 |
|------|------|
| $\mathcal{Z} \cong \mathbb{R}^{C\times H\times W}$ | VAE 潜空间, $C=4, H=W=64$ |
| $z_0$ | 内容图潜变量 |
| $z_1, z_s$ | 目标风格图潜变量 |
| $v_\theta(z,t,s)$ | 参数化速度场 |
| $s$ | 风格条件 (标量 id / 参考图特征序列) |
| $\pi:\mathcal{Z}\to\mathcal{B}$ | 内容投影 (结构提取) |
| $\mathcal{F}_c = \pi^{-1}(c)$ | 内容 $c$ 上的纤维 (同结构下的所有风格画法) |
| $\mathcal{H}_z, \mathcal{V}_z$ | $z$ 处的水平/垂直切子空间, $\mathcal{V}_z=\ker d\pi_z$ |
| SWD | 切片 Wasserstein 距离 |
| $\Pi$ | OT 传输计划 |

---

## 1. 风格迁移 = 纤维上的受控传输

把 $\mathcal{Z}$ 视为纤维丛 $E = (\mathcal{Z}, \mathcal{B}, \pi, \mathcal{F})$:

- 底空间 $\mathcal{B}$ = 内容拓扑 (轮廓/布局/语义).
- 纤维 $\mathcal{F}_c$ = "给定内容结构 $c$, 所有可能画风"的集合.
- 一次风格迁移就是沿联络做平行移动:
  $$z_{\text{src}}=(c,f_{\text{src}}) \longmapsto z_{\text{tgt}}=(c,f_{\text{tgt}}).$$

**第一性要求**: 理想速度场 $v^\*$ 应完全落在垂直分布里, $v^\*(z,t)\in\mathcal{V}_z$.
任何水平分量 $\delta c\ne 0$ 直接转换为 LPIPS 上升.

### 1.1 Pareto 前沿的几何读法

近似地:
$$\text{LPIPS}(z_0,z_1)\;\propto\;\big\|d\pi(z_1-z_0)\big\|_{\text{VGG}}\;\propto\;\text{水平分量大小},$$
$$\text{Style Score}\;\propto\;d_{\mathcal{F}}(f_{\text{out}}, f_{\text{tgt}})\;\propto\;\text{垂直分量到达深度}.$$

于是 LPIPS-Style 的 Pareto 前沿就是 $\mathcal{H}\perp\mathcal{V}$ 分解下的最优权衡.
仓库内真实前沿:

| 方法 | Style | LPIPS | 位置 |
|------|------:|------:|------|
| LBM H_e2 (velocity baseline) | 0.6994 | 0.3484 | 前沿上的"平衡点" |
| LBM K_e1 (轻 endpoint) | 0.7010 | 0.3623 | 沿前沿向右上爬 |
| inmortal xpred+kmanifold+pattn | 0.7338 | 0.6278 | **跳出前沿**, LPIPS 崩 |
| SaMAM step3000 | 0.6978 | 0.3221 | RGB 侧的前沿点 |
| topogate_appalign e2 | 0.6718 | 0.315 | 前沿左下角, "过保结构" |
| IDT no-op | 0.6801 | 0.0000 | 全锁死的极限点 |

关键读法:
- `topogate` 把 LPIPS 压到 0.315 但 style 也压到 0.67 — 联络太强, 垂直方向被锁死.
- `xpred+pattn` 把 style 推到 0.73 但 LPIPS 崩到 0.63 — 速度场里塞了太多水平分量.
- **前沿本身有天花板**, 这正是下一节要解释的均值坍缩.

---

## 2. 均值坍缩定理 (Conditional Expectation Attractor)

这是整个方案最关键的命题, 它把"style 卡在 0.70"的工程现象落到一个严格不等式上.

### 2.1 命题

**命题 (MSE 下的条件期望吸引子)**: 设训练目标为
$$\mathcal{L}(\theta)=\mathbb{E}_{t,(z_c,z_s)}\Big[\big\|v_\theta(z_t,t,s)-(z_s-z_c)\big\|^2\Big].$$
若同一状态 $z_t$ 在不同 batch / epoch 中被匹配到不同的目标 $\{z_s^{(1)},z_s^{(2)},\dots\}$ (即 OT 计划不稳定 或 闭集style_id 下多实例混合), 则 $L_2$ 意义下的贝叶斯最优速度场为
$$v_\theta^\*(z_t)=\mathbb{E}\big[z_s-z_c\,\big|\,z_t\big].$$

**推论 (高频抵消)**: 若目标 $z_s^{(k)}$ 的高频分量 (笔触相位) 互不相关, 则
$$\mathbb{E}[\,\text{HF}(z_s)\,]\;\approx\;0 \quad(\text{样本数}\to\infty),$$
因此 $v_\theta^\*$ 的高频分量被平均掉, 推理输出落在纤维上的**均值点** — 即"平滑塑料色块".

### 2.2 实验证据

- `h0_vertical_fm` (616/618 阶段, 纯垂直流 + Minibatch OT):
  13个epoch完美收敛, **LPIPS = 0.286** (结构锁得极好), **clip_style 卡在 0.669**.
  → 完美的"垂直"方向, 但塌到均值, 风格信号几乎为零.
- legacy tokenizer (5 个 embedding, blend=1.0) 七组 0.66–0.67.
- `xpred+pattn+stokes002`: style 0.7307, LPIPS 0.6183 → 突破均值靠的是泄漏到水平方向, 不是真的解决.
- SDE 上的所有"hack" (σ=0.02 / overdrive ×1.8 / affine 校准): 拼到 0.722, 还在均值吸引子的余晖里.

> 619/model/01 的"均值坍缩"表述在仓库真实结果上完全成立, 不是过拟合叙事.

### 2.3 突破均值的数学条件

均值坍缩的成因是"目标不确定性". 打破它至少满足以下一条:

1. **确定性配对 (deterministic pairing)**: 同一 $(z_c, z_s)$ 对在所有 epoch 中恒定, $\mathbb{E}[z_s\mid z_t]$ 退化为 $z_s$ 本身.
2. **实例级条件 (instance conditioning)**: 模型看到的 $s$ 携带本次 $z_s$ 的实例信息 → 即使目标在变, 模型能区分"这次是哪一个".
3. **改为分布匹配而非常数回归** — 但这会复活 SWD / OT 路径, 见 OT.md.

> 当前 LANCET 主线: 条件 1 缺 (Minibatch OT 跳变), 条件 2 缺 (闭集 id 查表), 条件 3 退化 (SWD 在 ODE unrolling 里被梯度爆炸阉割).
> 这是 style=0.70 天花板的**根因**, 不是超参问题.

---

## 3. 切空间分解与垂直流匹配

### 3.1 分解算子 (616/design 草案, 已实现)

定义低通 / 高通分离:
$$\text{Base}(z)=\text{LowPass}(z),\qquad \text{Fiber}(z)=z-\text{Base}(z).$$

仓库实现 (`bridge_path_mode="vertical"`): 5×5 avg_pool kernel. 616 补充已指出 5×5 在 64×64 latent 上截止频率偏高, 会把中频误划入纤维. 但工程上可用.

切空间分解:
$$T_z\mathcal{Z}=\mathcal{H}_z\oplus\mathcal{V}_z,\quad \mathcal{V}_z\approx\{\delta z:\text{Base}(\delta z)=0\}.$$

### 3.2 垂直流匹配的目标

构造训练样本时强制让结构分量静止:
$$\mu_t = \underbrace{\text{Base}(z_c)}_{\text{不随 }t\text{ 变}} + \big((1-t)\,\text{Fiber}(z_c) + t\,\text{Fiber}(z_s)\big).$$

目标速度:
$$v_{\text{target}}=\text{Fiber}(z_s)-\text{Fiber}(z_c),\qquad \text{Base}(v_{\text{target}})=0.$$

**意义**: 网络被禁止学习"如何把猫的形状变成风景", 100% 容量用于学习笔触.
**理论收益**: 假设成立时, LPIPS 应自然很小 (结构不动).
**真实结果**: `h0_vertical_fm` 确实拿到 LPIPS 0.286 — **该理论收益已观测到**.
**真实代价**: style 没动 (0.669) — 因为**目标 side 没有被锁死**.

> **关键诊断**: 垂直 FM 解决了"防止水平泄漏", 但**没有解决"垂直方向往哪儿走"**.
> OT.md 处理前者, tokenizer.md 处理后者.

---

## 4. 信息瓶颈定理 (DPI 上限)

设 $S$ = 风格参考图 (3×512², ~786K floats), $C_s$ = tokenizer 输出, $Y$ = 生成图.
**数据处理不等式**:
$$I(S;Y) \le I(C_s;Y) \le I(S;C_s).$$

仓库的几个 $C_s$ 对应的 $I(S;C_s)$ 上限估计:

| tokenizer | 输出维度 | 估信息量 |
|-----------|---------|---------|
| `legacy_factorized` (5×Embedding(256)) | 256 | <1 KB |
| PureLatentSpatial (16 cluster×128, 已 ZERO ROI) | 2K | ~8 KB |
| `lowrank_code_map` | ~2K | ~8 KB |
| **True Cross-Attn (DINO 16×16×384)** | 98K | **~400 KB** |

> 619/model/01 的"信息高速公路"叙事在数学上完全成立. 它是 DPI 的直接推论, 不依赖任何假设.
> 这解释了为什么 612-lookback 里所有"在 latent 内部自组织"路线都拿不到 style: 没有什么内生信号能凭空把 256D 的 $C_s$ 重建回 400KB 的纹理.

---

## 5. 梯度动力学: 为什么训练阶段的 ODE unrolling 必然失败

当前 `losses.py:2082` 的 `_terminal_swd`:
```python
endpoint = model.integrate(content, num_steps=N)  # autograd 内 N 步展开
loss = SWD(endpoint, style_target)
loss.backward()
```

反向传播的雅可比连乘:
$$\frac{\partial \mathcal{L}}{\partial\theta} = \frac{\partial\mathcal{L}}{\partial x_1}\sum_{k=1}^N\Big(\prod_{j=k+1}^N \frac{\partial x_j}{\partial x_{j-1}}\Big)\frac{\partial v_\theta(x_k)}{\partial\theta}\Delta t.$$

非线性 UNet 的状态转移谱半径几乎不可能恒等于 1, 二选一:
- $\rho>1$ → 爆炸 → 代码中大量 `clamp` / `nan_to_num` 是直接症状.
- $\rho<1$ → 风格梯度根本传不到早期层, 全靠 `w_flow` 单步回归的"假监督".

**结论**: terminal SWD 在训练态里**信息论意义上的有效梯度几乎为零**. 这一条与 619/model/04 一致.

### 5.1 单步假想终点 (619/model/06)

不展开 ODE, 改为在每个 $t$ 直接预测 $\hat z_1 = z_t + (1-t)v_\theta$, 然后在 $\hat z_1$ 上算 SWD.
梯度链长 1, 谱半径无所谓. 这把分布匹配监督从"训练动力学不可达"变成"每步可达".

---

## 6. 时空纠缠的梯度干涉

当前 `model.py`:`_compute_style_code` 的致命加法:
```python
return style_code + time_code
```

下游线性变换 $W$:
$$W(s+\tau) = Ws + W\tau.$$
$$\nabla_s\mathcal{L} = \nabla_\tau\mathcal{L} = W^\top\nabla_{\text{cond}}\mathcal{L}.$$

→ 优化器调 style 时同步扭曲时间动力学. 616~618 的 `w_kinetic` 扫描剧烈不稳定就是这个数学的直接结果.

**正交解耦条件**:
- AdaLN(t): 只动 $W_{\text{ada}}\in\mathbb{R}^{C\times 2}$ 调均值方差 — time 专属.
- CrossAttn(s): $K,V$ 由风格条件决定 — style 专属.
两组参数矩阵无交集, $\nabla_{\text{ada}}\mathcal{L}$ 与 $\nabla_{\text{ca}}\mathcal{L}$ 分布在 disjoint 参数空间, 干涉为 0.

> 这是 619/model/04 "正交梯度路径"的真实数学内容, 也是 619/system_diagnosis "缺陷 1" 的精确化.

---

## 7. 桥 / SDE: 为什么之前没拿到好结果

bridge.md (612) 的 I2SB 精确后验数学本身正确, 实现也对. 真正的问题不在求解器:

1. **训练-推理分布不匹配**: 训练 $z_t$ 带布朗噪声, 推理 $t{=}0$ 输入是干净 $z_0$, 第一步就 OOD. → 已通过 delayed noise schedule 部分缓解.
2. **均值坍缩未解除**: SDE 在采样态打噪声, 但若 $v_\theta^\*$ 本身已坍缩到条件期望, 加噪声只是采样均值周围的扰动 — 出不了纤维截面.
3. **错误地把 endpoint 当 velocity 用**: 619/system_diagnosis 缺陷 2 + theoretical_analysis.md 已经证明 endpoint 模式把 LPIPS 推到 0.62.
4. **网络架构不是为去噪设计**: LANCET 没有强 time embedding, SDE 噪声下 $v_\theta$ 的预测退化.

> **核心结论**: SDE 不是无效, 是被卡在均值吸引子上. 在 OT.md + tokenizer.md 解决均值坍缩之前, SDE 的所有 σ 扫描都不会有质变.
> 619/model 把这点表述为"先解决信息流, 再谈 SDE" — 我同意.

---

## 8. 这套数学导出的设计约束

把上面 6 个命题组合, 净化出**必须同时满足**的条件:

| 编号 | 数学约束 | 设计含义 |
|------|---------|---------|
| **C1** | 打破均值坍缩 | 训练目标要么 deterministic pairing, 要么 instance-level conditioning |
| **C2** | 信息通量 $\ge$ 风格信息量 | 风格条件走空间序列 path, 不走 1D embedding |
| **C3** | 切空间分解 | vertical FM / TopoGate / 结构正则之一阻止水平泄漏 |
| **C4** | 梯度链 $\le O(1)$ | 训练态禁止 ODE unrolling, 单步预测 $\hat z_1$ |
| **C5** | 梯度正交解耦 | time → AdaLN, style → Cross-Attention, 不加法混合 |
| **C6** | SDE 推理不引入训练态 OOD | delayed noise schedule + bridge_sigma 极小 (`≤0.05`) |

接下来:
- **OT.md**: 在 C1 + C3 下重新设计传输层 (解决"目标在哪儿").
- **tokenizer.md**: 在 C2 + C5 下重新设计风格表征 (解决"风格长什么样").
- **bridge.md**: 在 C4 + C6 下重新设计动力学 + SDE (解决"怎么走过去").

---

## 9. 与历史文档的差异 (audit 表)

来自 619/model 的所有可保留命题:

| 619/model 命题 | 状态 | 出处 |
|----------------|------|------|
| 风格迁移=纤维上平行移动 | ✅ 保留 | 01 §1.1 |
| LPIPS∝水平, Style∝垂直 | ✅ 保留 (近似) | 01 §1.3 |
| 均值坍缩定理 | ✅ 保留, 加强为 §2 形式 | 04 §1 |
| ODE unrolling 梯度爆炸 | ✅ 保留 | 04 §2.1 |
| 时空纠缠梯度干涉 | ✅ 保留, 形式化为 §6 | 04 §2.2 |
| DPI 信息瓶颈 | ✅ 保留 | 04 §3 |
| 闭集查表→纤维压缩点 | ✅ 保留 | 01 §2.1 |
| Minibatch OT→目标抖动 | ✅ 保留 | 01 §2.4 |
| **TopoGate=Ehresmann 联络** | ⚠️ 弱化为"软约束": 实测只把 plain forward delta 推到 1e-3 | 612-phase2/FIBER_BUNDLE_DESIGN |
| PureLatentSpatial 自发涌现语义 | ❌ 实验 ZERO ROI, 丢弃 | 612/plan-612 |
| AffineConnectionTokenizer 翻译算子 | ⚠️ 待 tokenizer.md 验证, 默认不采纳 | 616/design §3 |
| Bures-Wasserstein 全局统计量 | ✅ 保留 (作为可选 post-process) | 616/design §2 |
