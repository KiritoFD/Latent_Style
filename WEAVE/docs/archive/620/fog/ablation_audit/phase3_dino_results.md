# Phase 3.3 — DINO 与条件源消融实验报告

> 实验时间：2026-06-21  
> 基线配置：`exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json`  
> 核心问题：在当前 `endpoint_film_hd512` 配置下，`style_condition_source` 使用 `latent`（intrinsic cross-attention）是否足够？`target_dino_patches` 是否仍然必要？DINO adapter 是否有效？

---

## 1. 实验设计

### 1.1 基线确认

读取基线配置后确认：

- `model.style_condition_source = "latent"`
- `model.style_dino_adapter_enabled = false`
- `model.endpoint_head_mode = "endpoint_lowhigh"`
- `model.endpoint_style_hidden_dim = 512`
- `model.endpoint_film_enabled = true`

因此基线本身就是一个 **latent intrinsic** 配置，只是未在消融矩阵中单独标记。

### 1.2 消融变体

所有变体均继承 `620_film_v5_endpoint_film_hd512_local_smoke`，仅修改 `style_condition_source` 与 `style_dino_adapter_enabled`：

| 变体 | condition_source | adapter | 说明 |
|---|---|---|---|
| `620_ablation_intrinsic_latent_smoke` | `latent` | `false` | 完全使用 intrinsic latent cross-attention，不使用 DINO patches |
| `620_ablation_dino_baseline_smoke` | `target_dino_patches` | `false` | 使用目标图 DINO patches 作为 K/V，无 adapter |
| `620_ablation_dino_adapter_smoke` | `target_dino_patches` | `true` | 使用目标图 DINO patches 作为 K/V，并启用 DINO adapter |

训练规模严格保持 smoke：

- batch_size = 4
- accumulation_steps = 16
- num_epochs = 1
- 本地 RTX 4070

评估统一使用 `tools/run_eval_with_wfi.py --force-regen`，输出 `full_eval_wfi/epoch_0001/wfi_eval_report.json`。

---

## 2. 实验结果

### 2.1 指标汇总

| 变体 | condition_source | adapter/intrinsic | WFI ↓ | CLIP-S ↑ | content LPIPS ↓ | 训练时间（约） |
|---|---|---:|---:|---:|---:|---:|
| **基线** `620_film_v5_endpoint_film_hd512_local_smoke` | `latent` | intrinsic | **0.3906** | 0.7015 | 0.3382 | — |
| `620_ablation_intrinsic_latent_smoke` | `latent` | intrinsic | **0.3842** | 0.7020 | 0.3417 | **224 s** |
| `620_ablation_dino_baseline_smoke` | `target_dino_patches` | 无 adapter | **0.6407** | 0.7097 | 0.2773 | **257 s** |
| `620_ablation_dino_adapter_smoke` | `target_dino_patches` | 有 adapter | **0.6076** | 0.7063 | 0.2618 | **267 s** |

*注：训练时间为 `config.json` 写入到 `epoch_0001.pt` 保存的 wall-clock 间隔，单位为秒，仅供相对参考。*

### 2.2 WFI 子指标对比

| 变体 | contrast_ratio | dynamic_range | saturation | brightness | entropy |
|---|---:|---:|---:|---:|---:|
| intrinsic_latent | **3.54** | **43.69** | **0.249** | 0.510 | **6.97** |
| dino_baseline | 1.70 | 28.63 | 0.115 | 0.745 | 6.06 |
| dino_adapter | 1.84 | 31.73 | 0.117 | 0.717 | 6.24 |

`target_dino_patches` 两个变体在 contrast、dynamic_range、saturation 上均显著低于 intrinsic latent，而 brightness 更高，表现为典型的“雾化/白化”特征。

---

## 3. 关键发现

### 3.1 DINO patches 在当前配置下并非必要，反而显著加剧白化

- 将 `style_condition_source` 从 `latent` 改为 `target_dino_patches` 后，WFI 从 ~0.38 飙升至 ~0.61–0.64，远超 <0.40 的验收门。
- 虽然 DINO 变体的 CLIP-S 略高（0.706–0.710）、content LPIPS 更低（0.262–0.277），但白化严重到无法接受。
- 这表明在 `endpoint_film_hd512` 已经提供足够风格调制能力时，再叠加 DINO patches 条件会导致风格/端点信号过强，模型学到“高亮度、低饱和度、低对比度”的均值解。

### 3.2 Latent intrinsic cross-attention 已能通过 WFI 门

- `620_ablation_intrinsic_latent_smoke` WFI = 0.3842，CLIP-S = 0.7020，content LPIPS = 0.3417，全部满足当前验收标准：
  - WFI < 0.40 ✓
  - CLIP-S ≥ 0.695 ✓
  - content LPIPS < 0.36 ✓
- 与基线指标几乎一致，说明本次复测稳定。

### 3.3 DINO adapter 无法修复 DINO patches 带来的白化

- 启用 adapter 后，WFI 从 0.6407 微降至 0.6076，仍然大幅超标。
- adapter 带来 content LPIPS 进一步下降（0.2773 → 0.2618），但白化问题没有实质改善。
- 因此 adapter 在当前配置下不是白化解法。

### 3.4 与历史 H6 intrinsic 结果的对照

Git 历史摘要（`docs/620/fog/ablation_audit/git_history_digest.md`）记录 H6 intrinsic 路径：

| 版本 | CLIP-S | LPIPS | 说明 |
|---|---:|---:|:---|
| 历史 H6 intrinsic | 0.6717 | 0.3678 | 未超越 DINO，风格明显偏弱 |
| 当前 intrinsic_latent | **0.7020** | **0.3417** | 通过 WFI 门，风格与内容均优于历史 H6 |

关键差异在于当前配置已加入 `endpoint_head_mode=endpoint_lowhigh` + `endpoint_style_hidden_dim=512` + `endpoint_film_enabled=true`。Endpoint-FiLM 的大容量映射补偿了 intrinsic latent 风格信号的不足，使得 **latent 条件源也能达到此前 DINO patches 才能实现的风格强度**。这改变了“必须依赖 DINO”的历史结论。

---

## 4. 设计取舍建议

| 设计 | 建议 | 理由 |
|---|---|:---|
| `style_condition_source = "latent"` | **KEEP** | 当前唯一通过 WFI 门的条件源，且 CLIP-S/LPIPS 均达标，应作为默认配置保留。 |
| `style_condition_source = "target_dino_patches"` | **REMOVE**（默认关闭） | 在当前 `endpoint_film_hd512` 配置下导致严重白化（WFI > 0.60），默认应关闭。 |
| `style_dino_adapter_enabled = true` | **REMOVE**（默认关闭） | 无法修复 DINO patches 的白化，仅轻微提升 content LPIPS，成本收益不成正比。 |
| DINO patches / adapter 在更大容量下复测 | **NEED_MORE_DATA** | 历史 Round 1 怀疑 adapter 在 dim=64 下无效是因为 Q 维度受限。后续在 dim=128 / num_res_blocks=6 升级后可重新测试，看是否能同时保留高 CLIP-S 并通过 WFI 门。 |
| intrinsic latent 路径的长期有效性 | **NEED_MORE_DATA** | 当前仅为 1 epoch smoke 结果。若训练更多 epoch 或切换更大模型后出现风格衰减，需重新评估。 |

### 4.1 对后续实验的影响

- **白化验收优先**：任何后续优化（text conditioning、cross-attn Q source、dim=128 等）都应先以 `style_condition_source=latent` 为基线，确保 WFI < 0.40 后再引入 DINO 相关改动。
- **不要默认叠加 DINO**：在 `endpoint_film_hd512` 基线上，DINO patches 与 adapter 均应视为“默认关闭、按需复测”的实验性选项，而非核心路径。

---

## 5. 原始数据路径

- 基线配置：`exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json`
- intrinsic_latent 报告：`exp/620_spatial_bridge/620_ablation_intrinsic_latent_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
- dino_baseline 报告：`exp/620_spatial_bridge/620_ablation_dino_baseline_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
- dino_adapter 报告：`exp/620_spatial_bridge/620_ablation_dino_adapter_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
- 消融配置：`configs/ablations/620_ablation_intrinsic_latent_smoke.json`、`configs/ablations/620_ablation_dino_baseline_smoke.json`、`configs/ablations/620_ablation_dino_adapter_smoke.json`
