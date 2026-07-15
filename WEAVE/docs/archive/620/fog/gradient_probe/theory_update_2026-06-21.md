# Round E1.3 理论修正：基于 E1.1/E1.2 更新 Round M 预测

**日期**: 2026-06-21  
**依据**: 
- `docs/620/fog/baseline_audit/static_diagnosis_2026-06-21.md`
- `docs/620/fog/gradient_probe/probe_report_2026-06-21.md`
- `docs/620/fog/gradient_probe/internal_probe_epoch_0001.json`

---

## 1. 核心结论：从“整体 Endpoint Shrinkage”到“高频选择性塌缩”

Round M 的理论框架假设 620 白化的主要表现是 **整体端点 shrinkage**（`α(t=0) ≪ 1`）。E1.2 内部 probe 显示，当前最优基线 `620_film_v5_gated_local_smoke` 的 **整体 shrinkage 因子已提升到约 0.62**：

$$
c = \frac{\text{endpoint_alpha}}{1-t} \approx 0.617 \quad (t=0,0.5,0.875)
$$

但 **高频 shrinkage 因子仅约 0.08**：

$$
c_\text{high} = \frac{\text{endpoint_high_alpha}}{1-t} \approx 0.078
$$

因此，白化/雾化的直接原因不是端点整体不朝目标移动，而是 **高频分量（纹理、边缘、饱和度、对比度）几乎没有被迁移**。理论叙事需要从“整体 shrinkage”修正为 **“低频尚可、高频塌缩的选择性塌缩”**。

---

## 2. Round M 预测状态标记

### 2.1 `trivial_solution.md`

| 编号 | 原预测 | 新判定 | 理由 |
|---|---|---|---|
| P-L1 | 纯 FM loss 下 α≈1 | 待补证 | 未做纯 FM ablation |
| P-L2 | 增大 lowfreq/edge 惩罚会降 α | **支持（需细化）** | `lowfreqfix` 分支支持；当前 `single_step_edge_weight=0.1`、`kinetic_lambda_high=0.02` 可能同时抑制高频，是高频率塌缩的潜在 loss-driven 来源 |
| P-N1 | 移除 endpoint head GN 升 α | **部分支持** | 当前 `velocity` head 已无 GN，整体 α=0.62；但 `endpoint_lowhigh` head 仍有 GN，需单独验证 |
| P-N2 | 替换 trunk GN 增强 style 敏感度 | **待补证/次要** | Trunk block std 逐层放大，无 trunk 级统计塌缩；GN 不是当前主因 |
| P-A1 | gate=0.05 时 style velocity 余弦≈1 | **支持** | `cross_attn_entropy ≈ ln(256)`，attention 仍接近均匀，条件期望坍缩成立 |
| P-A2 | gate=0.3 降低 style velocity 余弦 | **部分支持** | gate 值升至 0.294，但 entropy 仍接近均匀；FiLM 部分绕过了 attention，但未完全恢复高频 style 区分度 |
| P-A3 | FiLM 绕过 attention 后 style sensitivity 上升 | **部分支持** | 整体 α 从旧基线 0.16 升到 0.62，但高频 α_high 仅 0.08；FiLM 提升了低频迁移，未解决高频 |
| P-E1 | 大 std init 升 α | **部分支持** | `velocity` head std=0.02 使整体 α 改善，但高频仍塌缩；说明 init 只能解决幅度，不能解决方向/谱分布 |
| P-E2 | endpoint head 直接预测减轻 shrinkage | **否证** | 当前 `velocity` head（α=0.62）优于旧 `endpoint_lowhigh` 无 FiLM（α≈-0.04）；直接预测 endpoint 若无强 style 调制反而更差 |
| P-E3 | endpoint head + style-FiLM 恢复 style sensitivity | **待补证/高度优先** | 当前 `endpoint_film_enabled=false`；这是下一轮最需验证的假设 |
| P-S1 | solver 是白化来源 | **否证** | 白化在 endpoint 预测已存在 |
| P-S2 | NFE 改变显著影响 WFI | **否证** | 旧 fog probe 显示 nfe 变化不改变 img_std |
| P-S3 | noise 改变显著影响结果 | **否证** | solver trace 显示 solver 不 source-seeking |

### 2.2 `stat_collapse.md`

| 预测 | 原内容 | 新判定 | 理由 |
|---|---|---|---|
| 预测 1 | GN 后振幅压缩 | **情境依赖** | Trunk 中 block output std 逐层放大（0.65→1.13），无压缩；但 endpoint head 内的 GN 仍可能压缩，需针对 `endpoint_lowhigh` 验证 |
| 预测 2 | AdaLN gate 饱和会压制 style | **当前否证** | `gate_mean=0.484`，未饱和；当前 checkpoint 中 AdaLN gate 不是瓶颈 |
| 预测 3 | FiLM 后 GN 洗掉通道调制 | **部分支持** | `pre_film_gamma_abs=0.20 > film_gamma_abs=0.15`，FiLM 引入的通道差异被后续 GN 部分洗掉；但 trunk 整体仍在放大方差，说明“洗掉”不等于“塌缩” |
| attention 平均化 | softmax 输出接近均匀 | **支持** | `cross_attn_entropy ≈ 5.53 ≈ ln(256)`，条件期望坍缩成立 |

### 2.3 `train_infer_mismatch.md`

| 命题/判据 | 原内容 | 新判定 | 理由 |
|---|---|---|---|
| A1：白化起源于 t=0 端点预测 | 核心假设 | **部分修正** | t=0 整体端点已相对健康（c=0.62）；白化主要来自 t=0 端点的**高频分量**缺失 |
| A2：训练 loss 下降不能排除推理白化 | 核心假设 | **支持** | 1-epoch smoke 的 loss 与 WFI 关系仍未收敛；需持续监控 |
| A3：style 信号进入网络但未转化为端点位移 | 核心假设 | **部分修正** | Style 信号已进入网络并转化为**低频**端点位移，但未转化为**高频**端点位移 |
| Late-stage mismatch 判据 | M_img > 0.5 | **当前不适用** | 1-epoch smoke 未进入晚期；当前问题是 endpoint 高频预测不足，而非 solver 补偿失控 |

---

## 3. 理论叙事修正

### 3.1 旧的单一叙事

> “620 白化是 endpoint shrinkage 导致的整体端点收缩：模型预测的端点几乎不朝目标方向移动。”

### 3.2 新的分层叙事

> “620 白化是 **分层选择性塌缩**：
> 1. Cross-attention 将 style 信号平均化为条件期望（attention-driven 条件期望坍缩）；
> 2. Block 内 StyleFiLM 部分绕过 attention，使 **低频/整体 latent 方向** 的端点位移恢复到较健康水平（c≈0.62）；
> 3. 但 **高频/细节方向** 的端点位移仍严重缺失（c_high≈0.08），因为 style 信号未直接进入 endpoint head，且 loss/head 参数化对高频位移有额外抑制；
> 4. 解码后高频缺失表现为低对比度、低饱和度、低动态范围，即 WFI 上升。”

### 3.3 机制贡献度更新

| 机制 | Round M 排序 | Round E1 更新 | 说明 |
|---|---|---|---|
| Attention-driven | 高 | 高 | 仍是 style-specific 信号弱化的起点，但 FiLM 已部分补偿 |
| Endpoint-driven | 高 | **高频极高 / 整体中等** | 整体 α 改善，但高频 α_high 塌缩；head 缺少 style-FiLM 是主因 |
| Norm-driven | 中—高 | **中 / 情境依赖** | Trunk 无塌缩；endpoint head GN 仍可疑，但在 velocity head 中不是主因 |
| Loss-driven | 中 | **中 / 针对高频** | Edge/kinetic 高频惩罚可能抑制高频位移 |
| Solver-driven | 低（已否证） | 低（已否证） | 保持否证 |

---

## 4. 下一轮只允许验证的候选方案列表

基于 E1.1/E1.2，以下方案与证据链直接对应，优先级如下：

### P0：Endpoint Head Style-FiLM + Low/High 分解

- **改动**: `endpoint_head_mode=endpoint_lowhigh`，`endpoint_film_enabled=true`。
- **理论预测**: style 信号直接调制高频 endpoint delta，使 `c_high` 从 0.08 提升到 ≥0.3。
- **成功门槛**: `wfi_score < 0.40`，`clip_style ≥ 0.695`，`content_lpips < 0.36`。
- **失败信号**: `c_high` 未提升，WFI 不变或恶化。

### P1：降低高频 Loss 惩罚

- **改动**: 减小 `single_step_edge_weight`、提高 `kinetic_lambda_high`，或加入高频鼓励项。
- **理论预测**: 当前 loss 对高频位移过度惩罚，导致 `c_high` 低；减轻后高频迁移改善。
- **成功门槛**: `c_high` 提升 ≥50%，WFI 下降。
- **失败信号**: WFI 上升或 content LPIPS 显著恶化。

### P2：Attention 稀疏化/结构化

- **改动**: 在 `gated` 基础上引入 style-dependent top-k 或 per-style token 选择。
- **理论预测**: 进一步降低 `cross_attn_entropy`，减少条件期望坍缩。
- **成功门槛**: `cross_attn_entropy` 下降 >0.1，同时 `c_high` 提升。
- **失败信号**: attention 熵不变，WFI 不变。

### P3：Endpoint Head 容量/初始化

- **改动**: 增大 `endpoint_style_hidden_dim`，或对 endpoint head 最后一层使用更大 std 初始化。
- **理论预测**: 更高容量/更大 init 使 head 能表示高频目标位移。
- **成功门槛**: `c_high` 提升，WFI 下降。
- **失败信号**: 容量提升无效果。

**禁止并行堆叠**：下一轮每次只改动一个候选，避免无法归因。

---

## 5. 已否证/降级的方案

| 方案 | 否证原因 | 状态 |
|---|---|---|
| 仅切换 attention 模式（relu2/gated_raw/style_select） | 四种模式 WFI 均高，clip_style 几乎不变 | 已否证作为独立修复 |
| 仅增大 gate_init | gate=0.3 已打开，但高频仍塌缩 | 已降级为必要条件而非充分条件 |
| 仅 block 内 StyleFiLM | 已使整体 α 提升，但高频未解决 | 已降级为部分措施 |
| Solver 相关修复 | 白化在 endpoint 预测已存在 | 已否证 |

---

## 6. 需要补证的开放问题

1. **纯 FM loss 下 α 是否接近 1**（P-L1）：若成立，说明当前辅助 loss 是高频塌缩的主要来源；若不成立，说明架构本身限制更大。
2. **`endpoint_lowhigh` 无 GN 的效果**（P-N1 延伸）：当前 velocity head 无 GN 已使整体 α=0.62；需验证 endpoint_lowhigh 去掉 GN 后是否也能避免旧基线的 α=-0.04。
3. **跨样本稳定性**：当前 probe 仅 5 样本，需扩展样本数以确认 `c_high` 的跨样本方差。
4. **多 epoch 演化**：1-epoch smoke 中 `c=0.62` 已不错，但 3–8 epoch 后高频是否改善或退化未知。

---

## 7. 总结

Round E1 的核心理论修正是：620 白化已从“整体端点 shrinkage”转变为 **“低频尚可、高频塌缩的选择性塌缩”**。当前 `gated` 基线通过打开 gate 和 block 内 FiLM 解决了大部分整体位移问题，但高频风格细节仍严重缺失。下一轮实验应优先验证 **endpoint head 内的 style-FiLM 调制** 和 **高频 loss 惩罚的调整**，并严格禁止多方案并行堆叠。
