# bridge.md — 桥动力学与 SDE 设计: 从 ODE unrolling 到单步回归 + 推理 SDE

> 承接 math.md C4+C6. 回答: **训练时怎么求梯度, 推理时怎么走过去**.
> 历史所有 SDE 方案都没拿到好结果 — math.md §7 已经定位根因 (均值坍缩未解除时 SDE 是空噪声). 
> 在 OT.md + tokenizer.md 解决均值坍缩之后, SDE 才有意义.

---

## 1. 现状回顾: 历史动力学方案全表

来自 `losses.py`, `model.py:integrate_transport`, `style_families.py`, `docs/612-phase2`:

| Solver | 训练目标 | 推理 | 训练-state ODE unrolling | 实测结果 |
|--------|---------|------|--------------------------|---------|
| `euler_legacy` (velocity) | MSE(v_pred, z_s-z_c) | Euler ODE | 否 | 0.70 / 0.35 (LBM baseline) |
| `euler_legacy` (endpoint) | MSE(x_1_pred, z_s) | 直接返回 x_1_pred | 否 | 0.73 / 0.62 (xpred 系列, LPIPS 崩) |
| `solver_i2sb` (endpoint + bridge_noise) | MSE(x_1_pred, z_s) | I2SB 后验 + 布朗噪声 | 否 | 远程 round2 未跑完; σ=0.02 弱噪声微改 0.71 |
| `solver_tangent_rk` | velocity + RK4 | 4阶 ODE | 否 | round1 跑了 32 epoch, 0.70 / 0.34 |
| `solver_pc` | velocity + 内容校正 | ODE + Langevin 校正 | 否 | round1 36 epoch, 0.69 / 0.33 |
| `solver_unsb_cycle` | velocity + SDE + cycle | SDE + 反向重建 | 否 | round1 30 epoch, 0.70 / 0.35 |
| **`_terminal_swd` (legacy)** | SWD on `model.integrate(z_0)` | — | **是, N 步** | 大量 clamp/nan_to_num 症状 |

### 1.1 失败原因统一诊断

按 math.md:
1. **均值坍缩**: 所有方案都受困于 OT 在线 → 跨-epoch target 跳变 → $v^* = \mathbb{E}[z_s - z_c \mid z_t]$ → 高频抵消. 不管 solver 多花哨都没救.
2. **ODE unrolling**: `_terminal_swd` 是唯一展开 ODE 的方案, 在 math.md §5 被判梯度链失败. 代码里大量 `clamp` 是直接症状.
3. **endpoint 模式的水平泄漏**: endpoint 训练 ($\hat z_1$ 直接回归 $z_s$) 把网络 capacity 全压在"跳到目标 latent"上, $z_s$ 的结构与 $z_c$ 完全无关, $v$ 的水平分量巨大, LPIPS 崩. 这是 612-phase2/theoretical_analysis.md §2 的定理.
4. **SDE 在坍缩速度场上 = 空噪声**: I2SB/UNSB 的 σ 扫描理论上"打破 mode collapse", 但坍缩后的 $v_\theta^*$ 是单点 delta — 加噪声也只是从这个 delta 周围采点, 出不了纤维截面.

---

## 2. 训练动力学: 数学要求 (math.md C4)

**硬约束 C4**: 训练态梯度链长 $\le O(1)$. → 禁止任何形式的 ODE unrolling.

由此推出:
- **目标**: 单步 velocity `$v_\theta(z_t, t, F_s)$` 的 MSE 回归.
- **训练样本构造**: 直线流 `z_t = (1-t) z_c + t z_s`, $v_{target} = z_s - z_c$.
- **可选结构监督**: 在 $z_t$ 处算假想单步终点 $\hat z_1 = z_t + (1-t) v_\theta$ 上算 loss. 这是 619/model/06 §5 的提案. 单步假想终点 → 梯度链长 1.

### 2.1 训练 loss (推荐)

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{FM}}}_{\text{主干}} + \lambda_{\text{SWD}} \mathcal{L}_{\text{SWD}}^{\text{single-step}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}^{\text{single-step}}$$

各项含义:

- **FM loss** (math.md C4):
  $$\mathcal{L}_{\text{FM}} = \big\|v_\theta(z_t, t, F_s) - (z_s - z_c)\big\|^2.$$
  梯度链长 1, 数学上稳定.

- **单步 SWD** (替代 `_terminal_swd`):
  $$\hat z_1 = z_t + (1-t)\,v_\theta,\qquad \mathcal{L}_{\text{SWD}}^{\text{ss}} = \text{SWD}(\hat z_1, z_s).$$
  把分布匹配监督从"训练态需要 ODE unrolling 才能 reach $z_1$"变成"随时可算".
  
  **取舍**: 这个 loss 强迫单步预测的 $\hat z_1$ 在分布上匹配 $z_s$, 是 tokenizer.md §3.2 的 soft transport 的"训练态驱动力". 没有它, Cross-Attention 的 $W_K/W_V$ 几乎收不到有效风格梯度 (只有 FM 通过 1D 残差泄漏过来的弱信号).

- **单步 edge L1** (math.md C3 的训练态补充, vertical FM 退化时的备份):
  $$\mathcal{L}_{\text{edge}}^{\text{ss}} = \big\|\text{HighPass}(\hat z_1) - \text{HighPass}(z_c)\big\|_1.$$
  在垂直 FM (OT.md §6 配置) 失效时, 这是结构保护的最低保险.

### 2.2 推荐 $\lambda$ 与扫描

```jsonc
{
  "lambdas": {
    "fm": 1.0,              // 主干, 不动
    "single_step_swd": 8.0, // 接近 619/model/06 §5 的 20%, 经验上 4-12
    "single_step_edge": 0.1 // 防御性, 不调高
  }
}
```

扫描优先级:
1. `single_step_swd` ∈ {4, 8, 12} — 这是 style 上限的主驱动.
2. `single_step_edge` 固定 0.1 — 不动, 除非 LPIPS >0.45 才提到 0.2.
3. `fm` 永远 1.0.

### 2.3 与历史 `_terminal_swd` 的关键差异

| 维度 | 旧 `_terminal_swd` | 本方案 SWD^ss |
|------|-------------------|---------------|
| $\hat z_1$ 怎么算 | `model.integrate(z_0, num_steps=N)` (ODE 展开) | `z_t + (1-t) v_theta` (单步假想) |
| 梯度链长 | $O(N)$ (5–16) | $O(1)$ |
| 雅可比稳定性 | 谱半径 ρ≠1, 爆炸/消失 | $I + (1-t) \partial v/\partial z_t$, 局部稳定 |
| 何时算 SWD | ODE 终点 | 任意 $t$ 处 |
| 镜像历史 `clamp/nan_to_num` | 必要 | 不需要 |

→ 这是 math.md §5 的直接落实.

---

## 3. 推理动力学: SDE 何时引入

历史 SDE 失败的统一原因 (math.md §7): 速度场已坍缩到 delta, 加噪声只能从 delta 周围采点. 
所以 **SDE 推理只能在 OT + tokenizer 都修复后再考虑**, 且 σ 必须**极小**.

### 3.1 SDE 不再是"打破 mode collapse"的工具, 是"高维扩散扰动"的工具

历史叙事: "SDE 的布朗噪声让轨迹散布到条件分布的边界, 取代 ODE 的 delta 输出". 
数学真相: 这个叙事要求 $v_\theta$ 在条件分布**整个支撑**上有定义, 但 MSE 训练出的是 delta attractor, SDE 只能在 attractor 附近扰动.

新方案下:
- 训练态: OT.md (确定性配对) 把 $v_\theta^*(z_t)$ 退化到 $z_s^{(i)} - z_c^{(i)}$ (单点), 不再是均值.
- 推理态: 该单点附近本来就有 SWD^ss 训练出的"风格高频细节"分布, SDE σ 扰动可以**强化采样多样性**, 但**不**承担"突破坍缩"的任务.

→ **σ 用极小值 (≤0.05), 不要扫描 0.25/0.5/1.0**. 619/model/02 §6 实验 4 的 overdrive 是推理期的 trick, 不在训练动力学范畴.

### 3.2 数学形式 (采纳 I2SB 精确后验, 612/bridge.md 公式正确)

I2SB 推理一步:
$$\hat z_1 = v_\theta(z_t, t, F_s),$$
$$\mu = c_{\text{curr}}\,z_t + c_{\text{tgt}}\,\hat z_1,\quad c_{\text{curr}}=\frac{1-t_{\text{next}}}{1-t_{\text{curr}}},\;c_{\text{tgt}}=\frac{t_{\text{next}}-t_{\text{curr}}}{1-t_{\text{curr}}},$$
$$\text{var}=\sigma^2\,\frac{(t_{\text{next}}-t_{\text{curr}})(1-t_{\text{next}})}{1-t_{\text{curr}}},$$
$$z_{t_{\text{next}}} = \mu + \sqrt{\text{var}}\,\epsilon.$$

最后一步 $t_{\text{next}}=1$ 时 var=0, 输出确定性 $\mu$.

数学本身由 612/bridge.md 已经正确推导, 实现也对 (`model.py:516-542`).
保留.

### 3.3 关键参数约束

```jsonc
{
  "inference": {
    "solver_family": "solver_i2sb",
    "transport_prediction_mode": "endpoint",  // endpoint 在推理期用, 训练用 velocity
    "num_steps": 8,                            // math.md C6: NFE 4-8 足够
    "bridge_sigma": 0.02,                      // 极小, 612-phase2/theoretical_analysis.md §5.3 推荐
    "bridge_noise_schedule": "delayed",        // math.md §7.1: 训练-推理不匹配防御
    "bridge_noise_window_start": 0.18,
    "bridge_noise_window_end": 0.82
  }
}
```

**为什么训练用 velocity, 推理用 endpoint?**
- 训练: velocity $v_\theta(z_t, t)$ 的 MSE 回归 → $\hat z_1 = z_t + (1-t)v$ 作为假想终点 → $\nabla$ 干净.
- 推理: I2SB 后验公式直接用 $\hat z_1$, 不需要中间表示 $v$. 把 velocity 输出通过 $z_t + (1-t)v$ 翻译成 endpoint 给后验用即可.

这避免了 612-phase2 §2 endpoint 训练导致 LPIPS 崩的问题 (因为训练时网络学的是 $v$, 它本身就是 $z_1 - z_0$ 形式, 残差性质保留; 推理时才转 endpoint).

---

## 4. delayed noise schedule 的精确推导 (math.md §7.1 + C6)

历史 Brownian bridge std: $g(t)=\sigma\sqrt{t(1-t)}$, 在 $t\to 0$ 时 $g(0)\to 0$ 但梯度有奇点, 训练时 $t\approx 0$ 的样本需要特殊处理.

更重要的是推理-训练 mismatch:
- 训练: $z_t = \mu_t + \sigma g(t) \epsilon$, $t\in(0,1)$ 加噪.
- 推理: $z_0$ 是干净源 latent (无噪), 第一步 $t=0$ 输入与训练分布不匹配.

delayed schedule:
$$g(t) = \begin{cases} 0 & t < 0.18 \\ \sigma\,\sin^2\!\left(\frac{\pi(t-0.18)}{0.64}\right) & 0.18 \le t \le 0.82 \\ 0 & t > 0.82 \end{cases}$$

效果:
- $t\in[0, 0.18]$ 和 $[0.82, 1]$ 上 $z_t = \mu_t$, 训练样本和推理输入都在干净分布上 → 推理 t=0 和 t=1 不 OOD.
- $t\in[0.18, 0.82]$ 注入噪声, 这是 I2SB 的探索阶段.

这是仓库已有的实现 (`style_families.py:resolves_exact_brownian_schedule`), 直接保留. 但 σ 必须 ≤0.05, 否则噪声会压过新 tokenizer 的细节信号 (DPI 输入被噪声污染等于把信息高速公路在中间步堵了).

---

## 5. Step 数 (NFE) 选择

math.md C6 + §7.2 要求: NFE 小但足以让单步 $\hat z_1$ 在多步积分中收敛.

| NFE | 训练目标 | 推理 | 风险 |
|-----|---------|------|------|
| 1 | 单步假想终点 | 直接 $\hat z_1 = z_0 + v$ | 推理只算一次 $v_\theta$, 没 I2SB 后验. |
| 4 | 单步假想 | I2SB 4 步 | 数学下界, 619/model/02 §6 声称可接受 |
| 8 (推荐) | 单步假想 | I2SB 8 步 | 平衡, 612/bridge.md 默认 |
| 16 | 单步假想 | I2SB 16 步 | 老的 round2 默认, 略慢 |

第一轮跑 8. 若 8 步推理和 16 步差距 <0.005, 切 4 步走效率; 若 8 步推理的 LPIPS > 0.42, 增到 16 步 (I2SB 后验本身在多步中收敛更稳).

---

## 6. 退役清单 (历史代码大扫除)

依据 math.md + 本文件分析, 退役以下代码路径:

| 路径 | 文件 | 退役理由 |
|------|------|---------|
| `_terminal_swd` ODE unrolling | `losses.py:2082` | math.md C4 |
| `terminal_swd_weight` config | `BridgeConfig` | C4 |
| `solver_tangent_rk` (RK4 推理) | `model.py` | velocity 残差够用, RK4 多步多算 |
| `solver_unsb_cycle` (cycle consistency) | `model.py` | math.md 619/model/06 §5 拒绝 cycle (cycle 强制可逆, 杀死高频不可逆纹理) |
| `solver_pc` (Langevin corrector) | `model.py` | 单步 SWD^ss + delayed noise 已覆盖其功能 |
| Minibatch OT 整套 | `losses.py`, `ot_cost.py` 的 sinkhorn path | OT.md §1.1 |
| `w_stokes_viscous`, `w_anisotropic_kinetic`, `w_curvature`, `w_phase_separation` | `losses.py` | 在 vertical FM + Cross-Attn 下冗余 |
| `cycle_consistency_loss`, `w_content_lowpass_anchor`, `w_content_edge_anchor` | `losses.py` | 同上, vertical FM 已硬约束 |

保留:
| 路径 | 文件 | 保留理由 |
|------|------|---------|
| `solver_i2sb` | `model.py:516` | 推理 SDE, σ 极小 |
| `solver_euler_legacy` | `model.py` | debug baseline, 不删 |
| vertical FM (`bridge_path_mode=vertical`) | `losses.py` | math.md C3, 结构保护 |
| `SWDTransportCost` | `ot_cost.py` | SWD^ss 需要 SWD 计算, 但用法变了 |
| delayed noise schedule | `style_families.py` | math.md C6 |

---

## 7. 推理后的 Bures-Wasserstein 全局校准 (可选 post-process)

math.md §2 的均值坍缩定理指出 "$v^*$ 高频抵消". 616/design.md §2 提的"亮度对比度漂移"是同一问题的低频侧面.

616/design §4 的简化 Bures-Wasserstein 校准:
```python
mu_src, std_src = z_out.mean((2,3), keepdim=True), z_out.std((2,3), keepdim=True).clamp_min(1e-6)
mu_tgt, std_tgt = self.style_stats[style_id]  # 训练时预存风格库均值方差
z_out_final = (z_out - mu_src) / std_src * std_tgt + mu_tgt
```

这是 **推理后 post-process**, 不进入训练. 作用是把全局亮度对比度精确搬到目标风格的统计量上 — 在 SWD^ss 仍校不准低频时是免费的红利.

第一轮**不启用**. 若出现"笔触对了但整体亮度偏"的现象再开.

---

## 8. 验证指标

训练态 (前 8 epoch 内):
- `loss_fm` 收敛曲线应平滑下降, 无 NaN/Inf (math.md §5 clamp 退役的判断).
- `loss_swd_ss` 应呈下降趋势. 若不动 → Cross-Attention 训练不动, 转 tokenizer.md §7 检查 `style_gate_value`.
- `loss_edge_ss` 应保持小常数 (vertical FM 工作). 若上升 → vertical FM 失效, 检查 `bridge_path_mode=vertical`.

推理态 (每 epoch 末):
- `clip_style` 应在 5 epoch 内过 0.70, 8 epoch 接近 0.72.
- `LPIPS` 应保持 <0.40.
- 同一 content 多次采样 (SDE σ=0.02) 的 std 应有 1e-3 量级扰动 (说明 SDE 不空).

---

## 9. 与历史文档的精确对接

参考采纳:
- 612/bridge.md 方案 1 (I2SB 精确后验): ✅ 数学正确, 推理期保留.
- 612-phase2/theoretical_analysis.md §5.2 (delayed noise schedule): ✅ 保留, σ 上限收紧.
- 619/model/04 §2.1 (ODE unrolling 梯度爆炸): ✅ 直接作为 C4 数学基础.
- 619/model/06 §5 (单步假想终点 + 多目标监督): ✅ 训练 loss 直接采用 §2.1.

不采纳:
- 612/bridge.md 方案 2 (Stochastic Flow Matching, velocity + score correction): ❌ Math 复杂, 在均值坍缩下等价无增益. 在新方案下不必要.
- 612/bridge.md 方案 3 (Langevin PC): ❌ math.md §5 已证明 PC solver 是 round1 36 epoch 实验, 与新方案 SWD^ss 功能重叠, 退役.
- 619/model/04 §4.3 正交梯度解耦: ✅ 采纳, 这是 tokenizer.md §5 的实现基础.
- 619/model/02 §6 推理 overdrive: ⚠️ 不进入第一轮, 作为 Phase B trick 保留.
- 616/design §2 Bures-Wasserstein: ⚠️ 作为可选 post-process (本文件 §7), 不进训练动态.
- 616/design §1 大扫除清单中所有 heuristic losses (cycle, content anchor, proximal): ✅ 全部退役.
