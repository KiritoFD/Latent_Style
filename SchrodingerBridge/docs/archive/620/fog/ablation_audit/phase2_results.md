# 620 消融审计：Phase 2 核心维度结果

> 运行时间：2026-06-21  
> 基线模板：`exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json`  
> 批量脚本：`tools/run_ablation_batch.py`  
> 实验环境：本地 RTX 4070，batch=4，accum=16，1 epoch smoke

---

## 1. 基线复测与通过标准

| 基线/复测 | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ | 状态 |
|---|---:|---:|---:|---|
| 历史最优 endpoint_film_hd512 | 0.3906 | 0.7015 | 0.3382 | ✅ 通过 |
| 本批次复测 attn_gated（同配置） | 0.3925 | 0.7020 | 0.3400 | ✅ 通过 |

硬约束：`wfi_score < 0.40`。所有 Phase 2 变体均通过。

---

## 2. Task 2.1 Attention 机制消融

固定：`endpoint_head_mode=endpoint_lowhigh`、`endpoint_film_enabled=true`、`endpoint_style_hidden_dim=512`、`style_film_enabled=true`、`style_cross_attn_gate_init=0.3`。

| 变体 | `style_attn_mode` | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ | 相对当前最优 ΔWFI |
|---|---|---:|---:|---:|---:|
| attn_softmax | softmax | **0.3736** | 0.7023 | 0.3397 | -0.0170 |
| attn_style_select | style_select | 0.3751 | 0.7015 | 0.3366 | -0.0155 |
| attn_sparsemax | sparsemax | 0.3779 | 0.7018 | 0.3354 | -0.0127 |
| attn_gated_raw | gated_raw | 0.3850 | 0.7017 | 0.3453 | -0.0056 |
| attn_relu2 | relu2 | 0.3856 | 0.7020 | 0.3434 | -0.0049 |
| attn_gated（复测） | gated | 0.3925 | 0.7020 | 0.3400 | +0.0019 |

**关键发现**：
- 在当前 `endpoint_film_hd512` 基线上，**所有 6 种 attention 模式均通过 WFI 门**。
- `softmax` 反而取得最低 WFI（0.3736），优于当前使用的 `gated`（0.3925）。
- 历史上有害的 `gated_raw`（0.6435）、`relu2`（0.5340）、`style_select`（0.5005）在 endpoint-FiLM 基线上均不再恶化。
- **结论**：attention 核函数不是当前白化的瓶颈；endpoint-FiLM/基线配置本身已提供足够鲁棒性。

---

## 3. Task 2.2 StyleFiLM 消融

固定：`endpoint_film_hd512` 基线其余参数。

| 变体 | `style_film_enabled` | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ |
|---|---|---:|---:|---:|
| stylefilm_on | true | 0.3785 | 0.7020 | 0.3321 |
| stylefilm_off | false | 0.3782 | 0.7021 | 0.3322 |

**关键发现**：
- Block 内 StyleFiLM 开/关差异极小（WFI 差 0.0003，LPIPS 差 0.0001）。
- 在已有 endpoint-FiLM 的情况下，block 内 FiLM 对最终指标几乎无独立贡献。
- **结论**：`style_film_enabled` 可关闭以简化模型，但保留也无害。

---

## 4. Task 2.3 Endpoint 结构消融

固定：`style_attn_mode=gated`、`style_film_enabled=true`、`style_cross_attn_gate_init=0.3`。

| 变体 | `endpoint_head_mode` | `endpoint_film_enabled` | `endpoint_style_hidden_dim` | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ |
|---|---|---|---|---:|---:|---:|
| endpoint_velocity | velocity | false | — | **0.3769** | 0.7020 | 0.3315 |
| endpoint_lowhigh_hd128 | endpoint_lowhigh | true | 128 | 0.3801 | 0.7023 | 0.3422 |
| endpoint_lowhigh_hd512 | endpoint_lowhigh | true | 512 | 0.3915 | 0.7019 | 0.3432 |
| endpoint_lowhigh_nofilm | endpoint_lowhigh | false | — | 0.3957 | 0.7012 | 0.3399 |
| endpoint_lowhigh_hd256 | endpoint_lowhigh | true | 256 | 0.3990 | 0.7013 | 0.3408 |

**关键发现**：
- `velocity` head 单独使用即通过 WFI（0.3769），且优于大多数 `endpoint_lowhigh` 变体。
- `endpoint_lowhigh_hd256` 接近门限（0.3990），表现反而差于 hd128 和 hd512。
- 在 endpoint 侧加入 FiLM 并未系统性地降低 WFI；`lowhigh_nofilm`（0.3957）与 `lowhigh_hd512`（0.3915）差距很小。
- **结论**：endpoint_lowhigh + FiLM 并非白化修复的唯一路径；当前基线配置（ latent 条件源 + gate_init=0.3 ）本身已稳定。

> 注：历史 `620_film_v5_gated`（velocity head，WFI=0.4902）与本批次 `endpoint_velocity`（WFI=0.3769）的差异，主要来自 `style_condition_source` 不同（历史为 `latent`，本批次基线实际也为 `latent`）。需进一步确认旧实验是否存在其他未记录的差异（如数据路径、训练步数、eval 设置）。

---

## 5. Task 2.4 Gate 初始化消融

固定：`endpoint_film_hd512` 基线其余参数。

| 变体 | `style_cross_attn_gate_init` | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ |
|---|---|---:|---:|---:|
| gate_init005 | 0.05 | **0.3757** | 0.7020 | 0.3413 |
| gate_init05 | 0.5 | 0.3833 | 0.7022 | 0.3415 |
| gate_init03 | 0.3 | 0.3908 | 0.7022 | 0.3446 |

**关键发现**：
- 默认 gate_init=0.05 取得最低 WFI；当前最优使用的 0.3 反而是三者中最差。
- 所有值均通过 WFI 门，说明 gate_init 在当前基线上不是决定性因素，但存在优化空间。
- **结论**：建议重新考虑默认 `style_cross_attn_gate_init=0.05`。

---

## 6. Phase 2 综合结论

| 维度 | 当前最优值 | 观察 | 建议 |
|---|---|---|---|
| `style_attn_mode` | gated | softmax 等模式均通过，部分更优 | **NEED_MORE_DATA**：可在 softmax 上复测多 epoch；暂时保留 gated |
| `style_film_enabled` | true | 开关差异极小 | **KEEP 或 REMOVE** 均可；若追求最小模型可关闭 |
| `endpoint_head_mode` | endpoint_lowhigh | velocity head 单独即可通过，且 WFI 更低 | **NEED_MORE_DATA**：velocity 路径更简，需验证多 epoch 稳定性 |
| `endpoint_film_enabled` | true | 对 WFI 影响不单调 | **NEED_MORE_DATA**：hd128/velocity 更简单 |
| `endpoint_style_hidden_dim` | 512 | hd128 表现优于 hd256/512 | **RESTORE/REDUCE**：hd128 可能足够 |
| `style_cross_attn_gate_init` | 0.3 | 0.05 更优 | **RESTORE**：恢复默认 0.05 |

---

## 7. 待 Phase 3 验证的问题

1. `style_condition_source` 的真实取值：当前最优基线 config 写为 `latent`，但 `ablation_matrix.md` 写为 `target_dino_patches`，需澄清。
2. 网络容量（64×4 vs 128×4）是否影响 WFI 或训练稳定性。
3. SWD 权重与 NSWD 噪声对风格/内容 trade-off 的影响。
4. DINO patches vs latent 条件源在当前基线下的差异。

---

## 8. 原始数据文件

- `results/task2_1_attention.json`
- `results/task2_2_stylefilm.json`
- `results/task2_3_endpoint.json`
- `results/task2_4_gate_init.json`
