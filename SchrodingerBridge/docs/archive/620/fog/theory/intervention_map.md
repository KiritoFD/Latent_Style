# 候选修复路径图谱（Intervention Map）

> Round M 理论文档：列出 7 条候选修复路径，对每条路径给出理论动机、最小实验设计、预期成功信号、失败信号/淘汰条件；并给出基于 Round A 证据的优先级排序与 DINO 去留门槛。

---

## 1. 背景与排序原则

排序基于以下 Round A 证据：

1. 白化起源于 $t=0$ endpoint 预测，而非 solver；
2. 高频端点方向错误（$\alpha_\text{high}(t=0) = -0.050$），低频弱但正（$\alpha_\text{low}=0.426$）；
3. Style 信号进入网络（gate=0.3 生效，FiLM gamma 增长），但未转化为足够端点位移；
4. Cross-attention 后的 GroupNorm / LayerNorm 可能洗掉 style 调制的一阶/二阶信息；
5. 单纯改变 attention 模式（gated/gated_raw/relu2/style_select）未能解决白化；
6. 直接预测 endpoint（endpoint_lowhigh）而无强 style 调制会坍回 source。

排序原则：**先解决 style 信号到端点位移的转化，再解决动态范围/统计塌缩，最后考虑容量升级与 DINO 去留。**

---

## 2. 候选修复路径

### 2.1 Endpoint FiLM Head（P0）

#### 理论动机

当前问题最精确的表述是：**style 信号存在，但未能转化为 target-facing 的 endpoint 位移**。FiLM 直接在 endpoint head 内部对 feature map 做通道级 style 调制，绕过 cross-attention 的平均化瓶颈和后续 GN 的部分洗掉效应：

$$
\text{FiLM}(h; s) = (1 + \gamma(s)) \odot h + \beta(s)
$$

其中 $\gamma, \beta$ 从 `style_global` 预测。与 additive style offset 不同，FiLM 调制的是 head 的 trunk feature，使 style 直接影响 endpoint 的高频/低频结构。

#### 最小实验设计

1. 配置：`endpoint_head_mode=endpoint_lowhigh`，`endpoint_film_enabled=true`；
2. 基于 `target_linear` 路径；
3. `style_cross_attn_gate_init=0.3`；
4. 移除或替换 endpoint head 中的 GroupNorm(1)（当前 `FiLMEndpointHead` 仍含 `self.norm = nn.GroupNorm(1, dim)`，需要改为无 GN 的 Conv trunk）；
5. 训练 1–3 epoch smoke；
6. 跑 `probe_620_hypothesis_metrics.py` 和 `probe_620_fog_path.py`。

#### 预期成功信号

- `latent_alpha_mean(t=0)` 从 0.16 提升到 $\geq 0.40$；
- `high_alpha_mean(t=0)$ 从 -0.05 提升到 $\geq 0.20$；
- `style_sensitivity_latent` 保持或提升；
- WFI $\leq 0.40$（本地基线 0.49 以下）。

#### 失败信号 / 淘汰条件

- `film_gamma_abs` 不增长（FiLM 未被使用）；
- 端点再次坍回 source（`to_source_img_delta_rms` 极小，`to_target_img_delta_rms` 大）；
- WFI 与基线持平或恶化；
- `style_sensitivity_latent` 下降。

---

### 2.2 Velocity Magnitude Constraint（P1）

#### 理论动机

训练 log 显示 `velocity_abs`（0.09–0.10）远小于 `target_velocity_abs`（0.51–0.54）。直接约束预测速度的幅度接近目标速度，可防止优化选择小位移局部最优：

$$
\mathcal{L}_\text{vel\_scale} = \left\| \frac{\|v_\theta\|_2}{\|v_\text{target}\|_2 + \epsilon} - 1 \right\|_2^2
$$

或更简单的 MSE：

$$
\mathcal{L}_\text{vel\_scale} = \| \|v_\theta\|_2 - \|v_\text{target}\|_2 \|_2^2
$$

#### 最小实验设计

1. 在 `losses620.py` 中加入 `velocity_target_ratio` MSE loss；
2. 权重 $w \in \{0.05, 0.1, 0.2\}$ 扫描；
3. 保持其他配置与 P0 最优基线一致；
4. 训练 1–3 epoch，监控 `velocity_abs` / `target_velocity_abs` 比值。

#### 预期成功信号

- `velocity_abs / target_velocity_abs` 从 ~0.2 提升到 ~0.6–0.8；
- `latent_alpha_mean` 提升；
- WFI 下降。

#### 失败信号 / 淘汰条件

- 比值提升但方向错误（$\alpha$ 不变或下降）；
- 训练不稳定（loss 震荡）；
- WFI 无改善或 content LPIPS 显著恶化（>0.36）。

---

### 2.3 Network Capacity Upgrade + Self-Attention（P1）

#### 理论动机

Round 1 诊断指出模型容量严重不足：4 层 dim=64 的 block 只有 183K 参数，DINO 投影占 1.34M。增大到 dim=128、6 层并加入 self-attention 可：

- 提升 style 梯度传播能力；
- 让空间位置间通信，学习笔触一致性；
- 为 FiLM/AdaLN 提供更丰富的特征表示。

#### 最小实验设计

1. `base_dim=128`，`num_res_blocks=6`；
2. 加入 self-attention（代码中 block 已包含 self-attention，但 Round 1 曾建议恢复/强化）；
3. 保持 P0 的 endpoint FiLM + gate=0.3；
4. 学习率降至 1e-4，batch=64；
5. 训练 3–8 epoch。

#### 预期成功信号

- `clip_style` 从 ~0.70 突破到 0.72+；
- WFI 在容量提升后进一步下降；
- `style_sensitivity_latent` 提升；
- 多 epoch 后稳定改善（无 late-stage mismatch）。

#### 失败信号 / 淘汰条件

- 容量提升后 WFI 仍高，说明瓶颈不在容量；
- 训练 OOM 或不稳定；
- 8 epoch 后出现 late-stage mismatch 且更严重。

---

### 2.4 Explicit Dynamic Range Loss（P2）

#### 理论动机

图像空间白化表现为低对比度、低动态范围。直接在 latent 或图像空间惩罚生成结果与 target 的动态范围差异：

$$
\mathcal{L}_\text{DR} = \big| \sigma(I_\text{gen}) - \sigma(I_\text{target}) \big| + \big| \mu(I_\text{gen}) - \mu(I_\text{target}) \big|
$$

或在 latent 空间：

$$
\mathcal{L}_\text{DR} = \big| \sigma(\hat{z}_1) - \sigma(y_\text{proj}) \big|
$$

#### 最小实验设计

1. 在 `losses620.py` 中加入 latent 空间动态范围 loss；
2. 权重 $w \in \{0.1, 0.5, 1.0\}$；
3. 基于 P0 或当前最优基线；
4. 训练 1–3 epoch，监控 WFI 分量（contrast_ratio, dynamic_range, saturation）。

#### 预期成功信号

- WFI 的 contrast_ratio 和 dynamic_range 分量改善；
- `endpoint_img_std_vs_source_ratio` 上升；
- 视觉低对比度减轻。

#### 失败信号 / 淘汰条件

- 动态范围提升但 style transfer 质量下降（target 结构被破坏）；
- `clip_style` 显著下降；
- 训练不稳定。

---

### 2.5 High-Frequency Residual / Endpoint Split（P2）

#### 理论动机

探针显示高频端点方向错误（$\alpha_\text{high}<0$），说明模型在高频 band 未能正确迁移风格。显式分离 low/high endpoint head 并分别监督：

$$
\mathcal{L}_\text{HF} = \| \hat{z}_{1,\text{high}} - y_{\text{proj, high}} \|_2^2
$$

当前代码已有 `endpoint_lowhigh` mode，但需要确保 high head 有独立的 style 调制和足够的初始化幅度。

#### 最小实验设计

1. `endpoint_head_mode=endpoint_lowhigh`；
2. 为 low/high head 分别加入 FiLM；
3. 使用独立的 high-frequency SWD/edge loss；
4. 初始 high scale $\gamma$ 从 1.0 扫描到 2.0；
5. 训练 1–3 epoch。

#### 预期成功信号

- `high_alpha_mean(t=0)$ 从负转正；
- `endpoint_latent_high_vs_source_ratio` 提升；
- 图像细节/纹理迁移改善。

#### 失败信号 / 淘汰条件

- high head 输出幅度过大导致 artifacts；
- low/high 分离后风格不连贯；
- `clip_style` 下降。

---

### 2.6 Noisy SWD (NSWD)（P1–P2）

#### 理论动机

SWD 在投影值排序稳定时梯度为 0，形成平坦区。NSWD 在投影值上加噪声：

$$
\tilde{p}^{(i)} = p^{(i)} + \sigma \cdot \varepsilon^{(i)}
$$

打破排序稳定性，使梯度非零。代码中已支持 `swd_noise_sigma`。

#### 最小实验设计

1. 设置 `swd_noise_sigma \in \{0.01, 0.02, 0.05\}`；
2. 与 P0/P1 联合实验；
3. 监控 SWD loss 和 `latent_alpha_mean`；
4. 训练 1–3 epoch。

#### 预期成功信号

- `loss_swd_ss` 更稳定或下降更快；
- `latent_alpha_mean` 提升；
- 与无 NSWD 相比 WFI 改善。

#### 失败信号 / 淘汰条件

- NSWD 导致训练不稳定；
- $\sigma$ 过大时 loss 不降；
- 单独 NSWD 无法提升 $\alpha$（说明 SWD 梯度不是唯一瓶颈）。

---

### 2.7 DINO Removal（P3 / 白化修复后评估）

#### 理论动机

DINO 提供 256 个 patch tokens 作为 style 条件，但：

- style_conditioner 占 1.34M 参数，是模型总参数的 86%；
- DINO 本身 frozen，信息可能冗余或与 latent 风格不完全对齐；
- Round 1 诊断认为若容量提升后仍卡，可考虑 intrinsic style（latent-based）。

#### 最小实验设计

1. 在 P0/P1 修复成功后，设置 `style_condition_source="latent"`；
2. 使用 `intrinsic_style_cnn` 从 target latent 提取 style tokens；
3. 保持其他架构不变；
4. 训练 3–8 epoch；
5. 对比 DINO vs no-DINO 的 WFI、`clip_style`、`clip_s_delta_idt`。

#### 预期成功信号

- no-DINO 与 DINO 的 `clip_style` 差异 $< 0.01$；
- WFI 不恶化；
- 训练更快、内存占用更低。

#### 失败信号 / 淘汰条件

- no-DINO 的 `clip_style` 下降 > 0.02；
- WFI 显著恶化；
- 风格细节（纹理、笔触）明显丢失。

**DINO 去留门槛**：

| 指标 | 保留 DINO | 删除 DINO |
|------|----------|----------|
| `clip_style` 差异 | $< 0.01$ | $> 0.02$ |
| WFI 差异 | no-DINO 不更差 | no-DINO 更差 |
| `clip_s_delta_idt` | no-DINO 不更低 | no-DINO 显著更低 |
| 训练效率 | 不重要 | 显著提升 |

若至少两项支持删除，则执行删除。

---

## 3. 当前优先级排序

| 优先级 | 修复路径 | 理由 |
|--------|----------|------|
| **P0** | Endpoint FiLM Head + 移除 GN | 最直接针对“style 信号存在但端点位移不足”的当前最可信诊断 |
| **P1** | Velocity Magnitude Constraint | 成本低，可作为 P0 的叠加项，防止 shrinkage |
| **P1** | NSWD ($\sigma=0.02$) | 打破 SWD 平坦区，辅助 P0；已在代码中支持 |
| **P2** | Network Capacity Upgrade + Self-Attention | Round 1 强烈建议，但应在 P0 验证后实施，避免盲目加容量 |
| **P2** | Explicit Dynamic Range Loss | 直接针对 WFI 指标，但可能破坏 style transfer，需谨慎 |
| **P2** | High-Frequency Residual / Endpoint Split | 针对高频方向错误，但当前代码已有 low/high 分离，需更强的 style 调制 |
| **P3** | DINO Removal | 仅在白化修复且 DINO 无明确增益后评估 |

---

## 4. 联合实验矩阵

建议以 P0 为基干，逐步叠加其他路径：

| 实验 ID | Endpoint FiLM | 无 GN | Velocity Scale | NSWD | Capacity | 目标 |
|---------|---------------|-------|----------------|------|----------|------|
| M1 | ✓ | ✓ | ✗ | ✗ | 64 | 验证 P0 核心假设 |
| M2 | ✓ | ✓ | ✓ | ✗ | 64 | 验证 velocity scale |
| M3 | ✓ | ✓ | ✓ | ✓ | 64 | 验证 NSWD 叠加效果 |
| M4 | ✓ | ✓ | ✓ | ✓ | 128 | 验证容量升级 |
| M5 | ✓ | ✓ | ✓ | ✓ | 128 | + DR loss |
| M6 | ✓ | ✓ | ✓ | ✓ | 128 | no-DINO 评估 |

---

## 5. 验收标准

### 5.1 白化修复通过门槛

- `wfi_score` $\leq 0.30$（本地）或 $\leq 0.20$（对齐 Seedream 评估）；
- `clip_style` $\geq 0.70$ 且不下降；
- `content_lpips` $< 0.36$；
- `latent_alpha_mean(t=0)$ $\geq 0.40$；
- `high_alpha_mean(t=0)$ $\geq 0.10$；
- `style_sensitivity_latent` 保持或提升。

### 5.2 DINO 去留门槛

- 在白化修复通过的前提下，no-DINO 与 DINO 的 `clip_style` 差异 $< 0.01$；
- no-DINO 的 WFI 不高于 DINO；
- 训练速度或内存有明显收益。

---

## 6. 结论

1. **P0（Endpoint FiLM Head + 无 GN）是当前唯一优先级**：它直接针对 Round A 最强证据——style 信号进入网络但未转化为端点位移。
2. **P1（Velocity Scale + NSWD）是低成本叠加项**：可加速跳出 shrinkage basin，但不应单独作为主攻方向。
3. **P2（容量升级 + DR loss + HF residual）是后续强化**：在 P0/P1 验证有效后再实施，避免无效增容。
4. **DINO 删除必须延后到白化修复后**：当前证据不支持删除 DINO；删除门槛需同时满足 style 质量、WFI、效率三个维度。
