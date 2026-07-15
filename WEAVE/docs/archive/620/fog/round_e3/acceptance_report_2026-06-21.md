# Round E3 白化压制验收报告

**日期**: 2026-06-21  
**执行环境**: 本地 RTX 4070，Windows，PyTorch CUDA  
**最终最优 checkpoint**: `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt`  
**验收标准**: WFI < 0.40，且 clip_style ≥ 0.695，且 content_lpips < 0.36

---

## 1. 验收标准

| 指标 | 硬门槛 | 说明 |
|---|---|---|
| `wfi_score` | **< 0.40** | 核心白化放行门，参考 Seedream IDT ≈ 0.158 留有余量 |
| `clip_style` | ≥ 0.695 | 风格迁移能力不得明显退化 |
| `content_lpips` | < 0.36 | 内容保留能力不得明显退化 |

> 注：WFI 门槛 0.40 是基于 Seedream IDT 参考值 0.158 设定的保守余量；即使未达到 IDT 水平，也需先压到 0.40 以下才允许进入后续优化阶段。

---

## 2. 验收过程

### 2.1 初始最优（Endpoint-FiLM Head 1 epoch）

| 指标 | 数值 | 状态 |
|---|---|---|
| `clip_style` | **0.7066** | ✅ 通过 |
| `content_lpips` | **0.3226** | ✅ 通过 |
| `wfi_score` | **0.4283** | ❌ **未通过** |
| `wfi_delta` (gen - source) | +0.1066 | 仍显著白化 |

- **Checkpoint**: `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/epoch_0001.pt`
- **Eval 目录**: `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/full_eval_wfi/epoch_0001/`

### 2.2 3 epoch 训练验证

为验证 WFI 是否随训练继续下降，对同配置跑了 3 epoch（`620_film_v5_endpoint_film_3ep_local`），结果如下：

| epoch | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI (gen - source) |
|---:|---:|---:|---:|---:|
| 1 | 0.7067 | 0.3236 | **0.4271** | +0.1054 |
| 2 | 0.7095 | 0.3505 | **0.4532** | +0.1315 |
| 3 | 0.7099 | 0.3768 | **0.4680** | +0.1463 |

**关键发现**：在当前学习率（2e-4）与配置下，WFI 随 epoch **单调上升**，说明更多训练反而加剧白化/雾化。3 epoch 最优仍为 epoch 1，且与单 epoch 版本基本持平。

### 2.3 下一轮最小修复实验（H1 / H2）

由于初始最优未通过，按 E3.3 执行两个最小改动候选：

| 实验 | 改动 | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI |
|---|---|---:|---:|---:|---:|
| 原始 Endpoint-FiLM | — | 0.7066 | 0.3226 | 0.4283 | +0.1066 |
| **H1** | `endpoint_film_init_std=0.02` | 0.7044 | 0.3217 | 0.4022 | +0.0805 |
| **H2** | `endpoint_style_hidden_dim=512` | 0.7015 | 0.3382 | **0.3906** | +0.0689 |

- **H1** 将 WFI 从 0.4283 压到 0.4022，已非常接近放行门，验证“非零初始化让 style 调制从早期就生效”的理论。
- **H2** 将 WFI 进一步压到 **0.3906**，正式低于 0.40，且 clip_style、content_lpips 均满足门槛。

---

## 3. 与 Seedream IDT 的对比

| 指标 | Seedream IDT | Endpoint-FiLM hd512 (最终最优) | 差距 |
|---|---|---|---|
| `wfi_score` | **≈ 0.158** | 0.3906 | **+0.233**（仍有差距，但已过 0.40 门） |
| `clip_style` | — | 0.7015 | — |
| `content_lpips` | — | 0.3382 | — |

最终最优虽未达到 Seedream IDT 水平，但已通过本项目设定的 0.40 放行门，具备进入后续优化阶段的资格。

---

## 4. 与当前基线的对比

| 指标 | 基线（E1 最优 gated） | 最终最优（H2 hd512） | 变化 |
|---|---|---|---|
| `clip_style` | 0.6987 | 0.7015 | ↑ +0.0028 |
| `content_lpips` | 0.3300 | 0.3382 | ↑ +0.0082（仍在门槛内） |
| `wfi_score` | 0.4902 | 0.3906 | ↓ **-0.0996** |
| `ΔWFI` | +0.1685 | +0.0689 | ↓ **-0.0996** |

最终最优在 WFI 上相比基线下降约 20.3%，同时保持 clip_style 略升，content_lpips 略有增加但仍在放行范围内。

---

## 5. 正式判断

| 门槛 | 最终最优（H2 hd512） | 是否通过 |
|---|---|---|
| WFI < 0.40 | 0.3906 | ✅ **通过** |
| clip_style ≥ 0.695 | 0.7015 | ✅ 通过 |
| content_lpips < 0.36 | 0.3382 | ✅ 通过 |

**结论：Round E3 白化压制验收 —— 通过。**

通过依据：
1. 经 H2（`endpoint_style_hidden_dim=512`）最小改动后，WFI 降至 0.3906，低于 0.40 硬门槛。
2. 核心副作用指标 clip_style（0.7015）与 content_lpips（0.3382）均保持在可接受范围内。
3. 与 Seedream IDT 的 WFI 差距从 +0.270 缩小到 +0.233，且 ΔWFI 从 +0.1685 大幅降至 +0.0689。

---

## 6. 关键洞察与理论修正

1. **3 epoch 训练不是解药**：在 lr=2e-4 下，更多 epoch 反而使 WFI 单调恶化，说明当前优化轨迹会在后期滑向更“安全”但更白化的解。
2. **容量瓶颈比初始化更关键**：
   - H1（非零 init）带来约 6% WFI 下降（0.4283 → 0.4022）。
   - H2（hidden_dim 128 → 512）带来约 9% WFI 下降（0.4283 → 0.3906），并直接过门。
3. **理论修正**：Round E2 的“style 信号未能有效调制 endpoint 方向”应进一步细化为 **“style→endpoint 的 FiLM 映射容量不足，导致 modulation 信号被压缩到接近零的无效区域”**。增大 `endpoint_style_hidden_dim` 直接提升了 style 嵌入到 FiLM gamma/beta 的表达能力，使 endpoint 的低/高频分支都能被 style 有效驱动。

---

## 7. 产物清单

| 产物 | 路径 |
|---|---|
| 最终最优 checkpoint | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt` |
| 最终最优完整 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/` |
| 原始 Endpoint-FiLM WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/full_eval_wfi/epoch_0001/` |
| 3 epoch 训练配置 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_3ep_local/config.json` |
| 3 epoch 各 epoch WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_3ep_local/full_eval_wfi/epoch_000{1,2,3}/` |
| H1 配置 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_init02_local_smoke/config.json` |
| H1 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_init02_local_smoke/full_eval_wfi/epoch_0001/` |
| H2 配置 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json` |
| H2 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/` |
| 本验收报告 | `docs/620/fog/round_e3/acceptance_report_2026-06-21.md` |
