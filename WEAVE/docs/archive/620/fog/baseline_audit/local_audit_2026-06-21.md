# 620 本地基线审计报告

**日期**: 2026-06-21  
**审计范围**: `exp/620_spatial_bridge/620_film_v5_*_local_smoke/`（共 4 个本地 smoke 实验）  
**数据集**: `F:/wikiart_distinct5_samam_512_latents_ema`（5 styles × 1000 images/style）  
**基础模型**: SD 1.5，配置族 `620_spatial_bridge`，solver `solver_i2sb`

---

## 1. 当前最可信问题定义

620 风格迁移模型的核心病症是 **endpoint shrinkage 导致的系统性白化/雾化**。证据链如下：

- 在 `targetlinear` 探针中，模型在 `t=0` 时预测端点只向目标方向移动了约 **16%**（`latent_alpha ≈ 0.163`），高频分量甚至出现错误方向（`high_alpha ≈ -0.050`）。
- 图像空间表现为：生成图对比度低、饱和度低、亮度偏高，WFI 指标显著高于 Seedream 健康参考（参考 `wfi_score ≈ 0.158`，而 620 本地 smoke 在 `0.49–0.64` 区间）。
- 白化集中在 source 端（`t ≈ 0`），而 `t` 较大时 endpoint 方向更接近真实目标，说明问题出在 **endpoint 预测** 而非 solver 积分或 VAE decode。

本次本地 smoke 实验在已施加 `gate_init=0.3`、`StyleFiLM`（block 内）、`NSWD σ=0.02`、`target_linear` 垂直路径的前提下，仍观察到明显的 WFI 上升（`ΔWFI = +0.17 ~ +0.32`）。因此，当前最可信的诊断是：**block 级的 StyleFiLM 与 gate 放大不足以克服 velocity 参数化和 endpoint head 的 shrinkage basin；style 信号虽存在，但未能充分转化为 target-facing 的 endpoint 位移。**

---

## 2. 已失败 / 已否证分支列表

以下分支在本次 4 个本地 smoke 实验中 **未能降低白化**，或产生了不利的权衡。

### 2.1 `gated_raw` / `relu2` / `style_select` attention 变体

| 实验 | `style_attn_mode` | `clip_style` | `content_lpips` | `transfer_clip_style` | `wfi_score` | `ΔWFI` |
|---|---:|---:|---:|---:|---:|---:|
| `620_film_v5_gated_local_smoke` | `gated` | 0.6987 | **0.3300** | 0.6646 | **0.4902** | **+0.1685** |
| `620_film_v5_gated_raw_local_smoke` | `gated_raw` | 0.6987 | **0.2973** | 0.6634 | 0.6435 | +0.3218 |
| `620_film_v5_relu2_local_smoke` | `relu2` | 0.6964 | 0.3102 | 0.6619 | 0.5340 | +0.2123 |
| `620_film_v5_style_select_local_smoke` | `style_select` | 0.6982 | 0.3331 | 0.6642 | 0.5005 | +0.1788 |

- **否证结论**: 移除 softmax 重归一化（`gated_raw`）、使用 ReLU²（`relu2`）或 top-k style token 选择（`style_select`）**均未降低 WFI**；相反，`gated_raw` 使 WFI 升至 0.64，白化最严重。
- **副作用**: 这些变体虽然能得到更低的 `content_lpips`（`gated_raw` 低至 0.297），但代价是更高的 WFI，说明它们在保留内容的同时把风格差异也“漂白”了。

### 2.2 仅提升 `style_cross_attn_gate_init` 到 0.3

- 在 `gated` 配置下，gate 从早期的 0.05 提升到 0.3 确实让 style 信号进入模型（runtime observability 中 `model_style_gate_value ≈ 0.294`）。
- 但 WFI 仍高达 0.49，距离 Seedream 参考（0.16）差距明显；同时 `content_lpips` 恶化到 0.33。
- **否证结论**: gate 放大 alone 不能解决 endpoint shrinkage，只是让 style 信号到达 block 而已。

### 2.3 block 级 StyleFiLM + velocity endpoint head

- 四个实验均启用 `style_film_enabled=true`，但 endpoint head 仍为 `endpoint_head_mode=velocity` 且 `endpoint_film_enabled=false`。
- 结果显示 style 敏感度存在，但 endpoint 位移不足；WFI 与 `clip_style` 之间没有同步提升（`clip_style` 都在 0.696–0.699 窄幅波动）。
- **否证结论**: 仅把 FiLM 放在 cross-attention block 内，而不直接调制 endpoint head，无法打破 shrinkage basin。

### 2.4 1-epoch smoke 作为完整结论的可靠性

- 所有实验仅训练 1 epoch；`clip_style` 在 0.70 目标以下，且不同 attention 模式的差异可能被训练噪声淹没。
- 因此，本次审计 **不将任何单一变体视为已收敛解**，只作为“哪条分支值得继续”的筛选依据。

---

## 3. 当前最优基线及其指标

**当前最优基线**: `620_film_v5_gated_local_smoke`

理由：在四个本地 smoke 中，它的 **WFI 最低**（0.4902），**WFI 上升最小**（+0.1685），`clip_style` 持平（0.6987），`transfer_clip_style` 最高（0.6646）。

关键指标：

- `clip_style` (all pairs): **0.6987**
- `content_lpips` (all pairs): **0.3300**
- `clip_s_delta_idt`: **0.0588**
- `transfer_clip_style`: **0.6646**
- `wfi_score` (generated): **0.4902**
- `source_wfi_score`: **0.3217**
- `ΔWFI = generated − source`: **+0.1685**
- WFI 分量：contrast_ratio=2.40, dynamic_range=36.95, saturation=0.139, brightness=0.569, entropy=6.71

**注意事项**:

- 该实验的 `train.log` 中记录了一次 `run_evaluation.py` 子进程返回非零退出码（标准 `full_eval` 失败），但 `full_eval_wfi/epoch_0001/summary.json` 已存在且包含完整 WFI 指标，说明 WFI 评估成功，标准评估可能为后续重跑失败或环境偶发问题。
- `content_lpips=0.3300` 是四个实验中最高的，说明“最少白化”与“最好内容保留”之间存在冲突。

---

## 4. 待验证假设列表（按优先级排序）

### P0: Endpoint-First + Style-FiLM Head 直接修复 shrinkage

- **假设**: 将 FiLM 放到 endpoint head 内部（`endpoint_film_enabled=true`，`endpoint_head_mode=endpoint_lowhigh`），并去掉 GroupNorm、使用非零初始化，能直接让 style 信号调制 endpoint 输出，从而把 `latent_alpha(t=0)` 从 0.16 提升到 0.5 以上，并显著降低 WFI。
- **对应配置**: `configs/620_spatial_bridge_film_formal.json`（基于 `targetlinear`）。
- **验证方式**: 在本地 RTX 4070 上跑 1–3 epoch smoke，对比 `gated` 基线的 WFI、`clip_style`、`content_lpips` 和 endpoint probe。

### P1: Velocity Magnitude Constraint 防止 shrinkage

- **假设**: 在 loss 中加入 `velocity_target_ratio` MSE 约束（权重约 0.1），强制 learned velocity 幅度接近 target velocity，可抑制优化选择“小位移”局部最优。
- **验证方式**: 在 P0 基础上叠加该 loss，观察 `velocity_abs` 是否上升、`wfi_score` 是否下降。

### P2: 网络容量升级 + Self-Attention

- **假设**: 当前 `base_dim=64`、无 self-attention 的瓶颈导致 style 梯度无法有效传播；升级到 `base_dim=128`、加入 self-attention 后，block 级 StyleFiLM 或 endpoint FiLM 才能真正发挥作用。
- **验证方式**: 在 P0/P1 验证有效后，做 dim128 正式 run，观察 `clip_style` 能否突破 0.72 且 WFI 继续下降。

### P3: 多 epoch 稳定性

- **假设**: 1-epoch smoke 排名可能受初始化/噪声影响；`gated` 基线在 3–8 epoch 后仍能维持最低 WFI，且 `clip_style` 继续上升。
- **验证方式**: 对 P0 最优组合跑 8 epoch，每 epoch 评估 WFI。

---

## 5. 下一步唯一优先级建议

**立即执行**: 以 `configs/620_spatial_bridge_film_formal.json` 为模板，在本地 RTX 4070 上启动 `620_film_v5_endpoint_film_local_smoke`，关键参数为：

- `_base`: `620_spatial_bridge_targetlinear.json`
- `model.endpoint_film_enabled=true`
- `model.endpoint_head_mode=endpoint_lowhigh`
- `model.style_attn_mode=gated`（基于当前最优基线）
- `model.style_cross_attn_gate_init=0.3`
- 训练 1–3 epoch，每 epoch 跑 WFI 评估

**验收标准**: 与当前最优基线 `620_film_v5_gated_local_smoke` 对比，要求：

- `wfi_score` 显著下降（目标 < 0.40，即 ΔWFI < +0.08）
- `clip_style` 不下降（≥ 0.695）
- `content_lpips` 不显著恶化（< 0.36）

若通过，则进入 8-epoch formal；若未通过，需回头检查 endpoint head 初始化与 GroupNorm 移除是否真正生效。

---

## 6. 证据冲突点与未决问题

### 6.1 冲突：WFI 与 content LPIPS 的反向关系

- `gated` 白化最少但内容失真最大（`content_lpips=0.330`）；`gated_raw` 内容保留最好但白化最严重（`wfi_score=0.644`）。
- **冲突含义**: 当前 attention 机制下，“更像目标风格”与“更像原图”的优化方向似乎不兼容；需要 endpoint-level 的 style 调制才能同时提升两者。

### 6.2 冲突：style 信号存在 vs endpoint 位移不足

- Runtime observability 显示 `model_style_gate_value ≈ 0.294`、`model_film_gamma_abs ≈ 0.122`，说明 style 信号已进入网络。
- 但 prior probe 显示 `latent_alpha(t=0) ≈ 0.16`，即信号未转化为足够位移。
- **冲突含义**: style 注入路径的“幅度”足够，但“方向/结构”不对；可能是 velocity 参数化或 endpoint head 容量/初始化问题。

### 6.3 未决：`gated` 实验的标准 eval 子进程失败

- `620_film_v5_gated_local_smoke/train.log` 末尾记录了 `run_evaluation.py` 返回 exit status 1。
- `full_eval_wfi/epoch_0001/summary.json` 存在且完整，说明 WFI 评估成功；失败的是标准 `full_eval`。
- **需确认**: 这是偶发环境问题，还是 `gated` 配置在标准 eval 路径上有 reproducibility 问题。建议在下一轮实验中同步验证标准 eval 是否稳定。

### 6.4 未决：1-epoch smoke 的排名可信度

- 四个变体的 `clip_style` 差异极小（0.696–0.699），但 WFI 差异较大。
- **需确认**: attention 模式对白化的影响在 3–8 epoch 后是否保持一致，还是会被训练后期动态抹平。

### 6.5 未决：WFI 参考基准的适用性

- Seedream 参考 `wfi_score ≈ 0.158` 来自 repaired750 的 30 张样本，而 620 评估使用 150 source + 750 generated 的本地测试集。
- **需确认**: 两套评估在分布/数量上的差异是否导致 WFI 不可直接比较；建议后续用同一 WFI 脚本在 620 生成图上重新计算并与 Seedream 样本对齐。

---

## 附录：被审计实验文件清单

| 实验目录 | config.json | train.log | epoch_0001.pt | wfi_eval_report.json | summary.json |
|---|---|---|---|---|---|
| `620_film_v5_gated_local_smoke` | ✓ | ✓（含标准 eval 失败记录） | ✓ | ✗ | ✓ |
| `620_film_v5_gated_raw_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `620_film_v5_relu2_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `620_film_v5_style_select_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ |

所有实验共用同一数据路径、同一训练超参（batch=4, accum=16, lr=2e-4, 1 epoch），唯一变量为 `model.style_attn_mode`。
