# Round E3 理论更新：Endpoint-FiLM 容量瓶颈与过训练风险

**日期**: 2026-06-21  
**依据**:
- `docs/620/fog/round_e3/acceptance_report_2026-06-21.md`
- `exp/620_spatial_bridge/620_film_v5_endpoint_film_3ep_local/full_eval_wfi/epoch_000{1,2,3}/wfi_eval_report.json`
- `exp/620_spatial_bridge/620_film_v5_endpoint_film_init02_local_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
- `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`

---

## 1. Round E2 理论回顾

Round E2 将白化叙事从 **"高频 RMS 位移不足"** 修正为 **"style 信号未能有效调制 endpoint 的低频/高频方向"**。Endpoint-FiLM Head 验证了这一方向：

- WFI 从 0.4902 降至 0.4283
- clip_style 从 0.6987 升至 0.7066
- content_lpips 从 0.3300 降至 0.3226

但 WFI 仍未达到 0.40 放行门，因此 Round E3 继续验证如何增强 endpoint 的 style 调制能力。

---

## 2. Round E3 实验结果

### 2.1 3 epoch 训练：更多 epoch 反而加剧白化

| epoch | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI |
|---:|---:|---:|---:|---:|
| 1 | 0.7067 | 0.3236 | 0.4271 | +0.1054 |
| 2 | 0.7095 | 0.3505 | 0.4532 | +0.1315 |
| 3 | 0.7099 | 0.3768 | 0.4680 | +0.1463 |

在 lr=2e-4、当前架构下，WFI 随 epoch **单调上升**。这说明：

> 当前优化目标与优化器会在训练后期滑向统计上更"安全"但更白化的 basin。单纯增加 epoch 不能解决问题，反而可能损害动态范围。

### 2.2 H1 / H2 最小修复

| 实验 | 改动 | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI |
|---|---|---:|---:|---:|---:|
| 原始 Endpoint-FiLM | — | 0.7066 | 0.3226 | 0.4283 | +0.1066 |
| H1 | `endpoint_film_init_std=0.02` | 0.7044 | 0.3217 | 0.4022 | +0.0805 |
| H2 | `endpoint_style_hidden_dim=512` | 0.7015 | 0.3382 | **0.3906** | +0.0689 |

- **H1** 验证非零初始化有助于 style 调制早期生效。
- **H2** 直接通过放行门，成为新的当前最优。

---

## 3. Round E3 理论修正

### 3.1 style→endpoint 的 FiLM 映射存在容量瓶颈

Round E2 提出 "style 信号未能有效调制 endpoint 方向"，Round E3 进一步明确：

> **瓶颈在于 style→endpoint 的 FiLM 映射容量不足**。当 `endpoint_style_hidden_dim=128` 时，style embedding 到 FiLM gamma/beta 的映射无法充分学习不同风格的复杂调制，导致 modulation 信号被压缩到接近零的无效区域，endpoint 输出退化为与 style 弱相关的"安全"均值。

证据：
- 将 hidden_dim 从 128 提升到 512（H2），WFI 从 0.4283 降到 0.3906。
- 单纯非零初始化（H1）也有帮助，但效果弱于容量提升，说明初始化只是让训练启动更好，真正的表达瓶颈是网络容量。

### 3.2 过训练会放大白化

3 epoch 实验显示：
- content_lpips 从 0.3236 升至 0.3768
- WFI 从 0.4271 升至 0.4680

这说明在当前目标函数和 lr 下，优化器会逐渐找到一个 **内容保持更好、但动态范围更差** 的 basin。该 basin 可能对应于：
- endpoint 输出向 source latent 的统计均值收缩
- 高频分支被进一步抑制
- style-specific 调制被平滑化为对所有风格都"不犯错"的均值响应

因此：
> **白化问题不仅是"学得不够"，也可能是"学得太过"导致的统计塌缩**。在解决容量瓶颈后，需要配合 early stopping 或学习率调整防止过训练。

### 3.3 初始化与容量的交互

- H1（init std=0.02）+ H2（hd=512）尚未组合测试，但理论上可能互补：
  - 容量提升提供足够的表达能力
  - 非零初始化避免从零开始的冷启动
- 这是 Round E4 可选验证方向之一。

---

## 4. Round M / E 预测状态更新

| 预测 | 原判定 | E3 更新 | 理由 |
|---|---|---|---|
| style 信号未能有效调制 endpoint 方向 | 部分支持 | **支持** | H2 提升 capacity 直接过门 |
| 高频塌缩 = 高频 RMS 不足 | 否证 | 否证 | 容量提升降低 WFI，未依赖增大 RMS 位移 |
| Endpoint-FiLM 需要非零初始化 | 待验证 | **部分支持** | H1 有效果但未过门 |
| Endpoint-FiLM 需要更大 capacity | 待验证 | **强支持** | H2 过门 |
| 更多 epoch 能降低 WFI | 待验证 | **否证** | 3ep 单调恶化 |
| Attention 平均化是 style 弱化起点 | 支持 | 支持 | 未改变 attention 机制，仍依赖 FiLM 绕过 |

---

## 5. 下一阶段（Round E4）候选方向

白化放行门已通过，但 WFI 0.3906 距离 Seedream IDT（≈0.158）仍有差距。后续优化可谨慎推进：

1. **组合 H1 + H2**：`endpoint_film_init_std=0.02` + `endpoint_style_hidden_dim=512`，看是否能进一步压低 WFI。
2. **学习率 / early stopping**：鉴于 3ep 过训练，尝试 lr=1e-4 或更短的 epoch，配合 hd=512。
3. **移除 FiLMEndpointHead 内的 GroupNorm**：验证 GN 是否压缩动态范围。
4. **low/high 分解带宽**：调整 `endpoint_lowpass_kernel`。
5. **恢复 620 原计划实验**：text、cross-attn、DINO 等，但必须先过白化指标检查。

所有后续实验必须沿用 WFI 指标，确保不会以提升 clip_style 为代价重新引入雾化。

---

## 6. 总结

Round E3 的核心理论修正是：

> 620 白化的关键不是 style 信号"有没有"送到 endpoint，而是 **style→endpoint 的 FiLM 映射是否有足够容量学习有效的 per-style 调制**。当容量不足时，调制信号被压缩，endpoint 退化为与风格无关的安全均值输出，表现为白化/雾化。同时，在当前目标函数下，过训练会进一步放大这种塌缩，因此容量提升后仍需注意学习率与 early stopping。
