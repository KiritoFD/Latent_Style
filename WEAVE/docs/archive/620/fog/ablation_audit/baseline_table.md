# 620 消融审计：历史基线对照表

> 生成时间：2026-06-21  
> 用途：为后续消融实验提供三条统一参照基线  
> 白化放行门：`wfi_score < 0.40`，`clip_style ≥ 0.695`，`content_lpips < 0.36`

---

## 1. 基线汇总表

| 基线名称 | 实验目录 / 来源 | 关键配置 | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI (gen − source) | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| **Round 1 base_swd8** | `docs/620/round1_diagnosis.md` | dim=64, num_res_blocks=4, SWD weight=8, 8 epoch 远程 3060 | 0.6720 | 0.2900 | — | — | 历史瓶颈基线 |
| **白化修复前 gated** | `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/` | dim=64, gated attn, style_film, endpoint=velocity | 0.6987 | 0.3300 | 0.4902 | +0.1685 | ❌ 白化严重 |
| **当前最优 endpoint_film_hd512** | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/` | dim=64, gated attn, endpoint_lowhigh + FiLM, hd=512 | 0.7015 | 0.3382 | 0.3906 | +0.0689 | ✅ 通过放行门 |

---

## 2. 每条基线的关键上下文

### 2.1 Round 1 base_swd8

- **来源**：`docs/620/round1_diagnosis.md` 中 7 变体 × 8 epoch 远程 3060 训练的主基线。
- **关键参数**：
  - `model.base_dim = 64`
  - `model.num_res_blocks = 4`
  - `bridge.single_step_swd_weight = 8`
  - `model.style_attn_mode = gated`（或早期等价实现）
  - `training.num_epochs = 8`
- **观察**：
  - `clip_style` 集中在 0.668–0.677 平台，远低于目标 0.72+。
  - `LPIPS` 极好（0.29），说明 vertical FM 有效。
  - 增加 adapter / MoE / gate12 均无收益，根因诊断为 Q 侧维度不足（dim=64）和缺少 self-attention。
- **遗留问题**：当时未引入 WFI 指标，无法评估白化程度。

### 2.2 白化修复前 gated（620_film_v5_gated_local_smoke）

- **来源**：本地 RTX 4070 smoke 实验，`epoch_0001` 完整 WFI 评估。
- **关键参数**：
  - `model.base_dim = 64`
  - `model.num_res_blocks = 4`
  - `model.style_attn_mode = gated`
  - `model.style_film_enabled = true`
  - `model.endpoint_head_mode = velocity`
  - `model.endpoint_film_enabled = false`
  - `bridge.single_step_swd_weight = 8`
  - `training.num_epochs = 1`
- **评估指标**（来自 `full_eval_wfi/epoch_0001/summary.json` 与 `wfi_benchmark.json`）：
  - `clip_style`（all pairs）：0.6987
  - `content_lpips`（all pairs）：0.3300
  - `wfi_score`（generated）：0.4902
  - `wfi_score`（source）：0.3217
  - `ΔWFI = 0.4902 − 0.3217 = +0.1685`
- **诊断结论**：白化严重；问题被定位为 endpoint shrinkage，即 style→endpoint 调制信号太弱。

### 2.3 当前最优 endpoint_film_hd512

- **来源**：本地 RTX 4070 smoke 实验，`620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt`。
- **关键参数**：
  - `model.base_dim = 64`
  - `model.num_res_blocks = 4`
  - `model.style_attn_mode = gated`
  - `model.style_film_enabled = true`
  - `model.endpoint_head_mode = endpoint_lowhigh`
  - `model.endpoint_film_enabled = true`
  - `model.endpoint_style_hidden_dim = 512`
  - `model.endpoint_film_init_std = 0.0`
  - `model.style_cross_attn_gate_init = 0.3`
  - `bridge.single_step_swd_weight = 8`
  - `training.num_epochs = 1`
- **评估指标**（来自 `full_eval_wfi/epoch_0001/wfi_eval_report.json`）：
  - `clip_style`：0.7015
  - `content_lpips`：0.3382
  - `wfi_score`：0.3906
  - `source_wfi_score`：0.3217
  - `ΔWFI = 0.3906 − 0.3217 = +0.0689`
- **结论**：首次通过 `wfi_score < 0.40` 放行门；但距离 Seedream IDT（wfi≈0.158）仍有差距。

---

## 3. 并排对比

| 指标 | Round 1 base_swd8 | 白化修复前 gated | 当前最优 hd512 | 变化说明 |
|---|---|---:|---:|---|
| clip_style ↑ | 0.6720 | 0.6987 | 0.7015 | 从 0.67 平台提升到 0.70 平台 |
| content_lpips ↓ | 0.2900 | 0.3300 | 0.3382 | hd512 略升，但仍低于 0.36 门 |
| wfi_score ↓ | — | 0.4902 | 0.3906 | **−0.0996**，通过放行门 |
| ΔWFI ↓ | — | +0.1685 | +0.0689 | **−0.0996**，生成图更接近 source 统计 |
| 训练 epoch | 8 | 1 | 1 | 当前最优仅需 1 epoch；3 epoch 会恶化 |
| 参数规模 | 1.55M（blocks 183K） | ~1.55M | ~1.70M（FiLM head 扩容） | 容量增加有限 |

---

## 4. 对消融实验的用法说明

1. **所有后续消融实验必须先与这三条基线并排对比**，优先保证 `wfi_score < 0.40` 不恶化。
2. **clip_style ≥ 0.695 和 content_lpips < 0.36 是第二、第三约束**；任何改善 WFI 但严重损害 style/content 的变体不采纳。
3. **ΔWFI 是更敏感的白化信号**：即使 `wfi_score` 接近，ΔWFI 仍可反映生成图相对 source 的统计偏移。
4. **Round 1 base_swd8 作为“历史天花板未破”参照**：当前最优已在其基础上提升 clip_style 约 0.03，但仍在同一数量级。
5. **后续核心目标**：在保持 WFI < 0.40 的前提下，通过容量升级（dim=128）和 Phase 4 架构进一步提升 clip_style。

---

## 5. 数据来源索引

| 基线 | 关键文件 |
|---|---|
| Round 1 base_swd8 | `docs/620/round1_diagnosis.md` |
| 白化修复前 gated | `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/full_eval_wfi/epoch_0001/summary.json` |
| 白化修复前 gated WFI | `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/full_eval_wfi/epoch_0001/wfi_benchmark.json` |
| 当前最优 hd512 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json` |
| 完整历史调研 | `docs/620/fog/ablation_audit/git_history_digest.md` |
