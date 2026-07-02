# 620 消融审计：实验矩阵

> 生成时间：2026-06-21  
> 基线模板：`configs/620_spatial_bridge_targetlinear.json`  
> 当前最优：`620_film_v5_endpoint_film_hd512_local_smoke`（WFI=0.3906，clip_style=0.7015）

---

## 1. 设计维度总览

| 维度类别 | 配置键 | 当前最优值 | 候选取值 | 优先级 |
|---|---|---|---|---|
| **Attention 机制** | `model.style_attn_mode` | `gated` | `softmax` / `gated` / `gated_raw` / `relu2` / `style_select` / `sparsemax` | P0（核心） |
| **Block 内 StyleFiLM** | `model.style_film_enabled` | `true` | `true` / `false` | P0（核心） |
| **Endpoint 结构** | `model.endpoint_head_mode` | `endpoint_lowhigh` | `velocity` / `endpoint_lowhigh` | P0（核心） |
| **Endpoint FiLM** | `model.endpoint_film_enabled` | `true` | `true` / `false` | P0（核心） |
| **Endpoint FiLM 容量** | `model.endpoint_style_hidden_dim` | `512` | `128` / `256` / `512` | P0（核心） |
| **Gate 初始化** | `model.style_cross_attn_gate_init` | `0.3` | `0.05` / `0.3` / `0.5` | P0（核心） |
| **网络容量** | `model.base_dim` × `model.num_res_blocks` | `64×4` | `64×4` / `64×6` / `128×4` / `128×6` | P1（扩展） |
| **SWD 权重** | `bridge.single_step_swd_weight` | `8.0` | `0.0` / `2.0` / `8.0` / `16.0` | P1（扩展） |
| **NSWD 噪声** | `bridge.swd_noise_sigma` | `0.02` | `0.0` / `0.02` | P1（扩展） |
| **HF Residual** | `model.velocity_hf_residual_enabled` | `false` | `true` / `false` | P2（扩展） |
| **DINO Adapter** | `model.style_dino_adapter_enabled` | `false` | `true` / `false` | P2（扩展） |
| **Style MoE** | `model.style_moe_enabled` | `false` | `true` / `false` | P2（扩展） |
| **风格条件源** | `model.style_condition_source` | `target_dino_patches` | `target_dino_patches` / `latent` | P2（扩展） |
| **FiLM 初始化** | `model.endpoint_film_init_std` | `0.0` | `0.0` / `0.02` | P2（扩展） |

---

## 2. 核心维度（Phase 2）

### 2.1 Attention 机制消融（Task 2.1）

以 `endpoint_film_hd512` 为基线，固定 `endpoint_head_mode=endpoint_lowhigh`、`endpoint_film_enabled=true`、`endpoint_style_hidden_dim=512`，仅改变 attention 核函数。

| 变体名 | `style_attn_mode` | 历史证据 | 预期结果 |
|---|---|---|---|
| `attn_softmax` | `softmax` | 基础实现 | 验证默认 softmax 是否可接受 |
| `attn_gated` | `gated` | 当前最优使用 | 作为核心基线 |
| `attn_gated_raw` | `gated_raw` | WFI 0.6435，恶化 | 验证历史结论 |
| `attn_relu2` | `relu2` | WFI 0.5340，无效 | 验证历史结论 |
| `attn_style_select` | `style_select` | WFI 0.5005，无效 | 验证历史结论 |
| `attn_sparsemax` | `sparsemax` | 代码已实现，无实验 | 探索稀疏注意力 |

**关键问题**：在当前 `endpoint_film_hd512` 基线上，attention 模式是否仍对白化敏感？

### 2.2 StyleFiLM 消融（Task 2.2）

固定 `endpoint_film_hd512` 基线，切换 block 内 `style_film_enabled`。

| 变体名 | `style_film_enabled` | 说明 |
|---|---|---|
| `stylefilm_on` | `true` | 当前最优 |
| `stylefilm_off` | `false` | 移除 block 内 FiLM + Q-FiLM + style_bias |

**关键问题**：block 内 StyleFiLM 是否是 endpoint FiLM 有效的必要条件？

### 2.3 Endpoint 结构消融（Task 2.3）

固定 `endpoint_film_enabled=true`，扫描 endpoint head 结构和容量。

| 变体名 | `endpoint_head_mode` | `endpoint_style_hidden_dim` | `endpoint_film_enabled` | 说明 |
|---|---|---|---|---|
| `endpoint_velocity` | `velocity` | — | `false` | 回退到 velocity head |
| `endpoint_lowhigh_nofilm` | `endpoint_lowhigh` | — | `false` | low/high 拆分但无 FiLM |
| `endpoint_lowhigh_hd128` | `endpoint_lowhigh` | `128` | `true` | 低容量 FiLM |
| `endpoint_lowhigh_hd256` | `endpoint_lowhigh` | `256` | `true` | 中容量 FiLM |
| `endpoint_lowhigh_hd512` | `endpoint_lowhigh` | `512` | `true` | 当前最优 |

**关键问题**：`endpoint_style_hidden_dim` 是否存在阈值效应？`endpoint_lowhigh` 本身是否必要？

### 2.4 Gate 初始化消融（Task 2.4）

固定 `endpoint_film_hd512` 基线，扫描 cross-attention gate 初始值。

| 变体名 | `style_cross_attn_gate_init` | 历史证据 |
|---|---|---|
| `gate_init005` | `0.05` | 默认值 |
| `gate_init03` | `0.3` | 当前最优，velocity magnitude +16% |
| `gate_init05` | `0.5` | 更大初始 gate |

**关键问题**：gate 初始值是否影响白化？当前 0.3 是否最优？

---

## 3. 扩展维度（Phase 3）

### 3.1 网络容量消融（Task 3.1）

在 `endpoint_film_hd512` 基线上升级容量，验证 Round 1 假设。

| 变体名 | `base_dim` | `num_res_blocks` | `style_attn_num_heads` | 说明 |
|---|---|---|---|---|
| `capacity_64x4` | `64` | `4` | `4` | 当前最优 |
| `capacity_64x6` | `64` | `6` | `4` | 增加深度 |
| `capacity_128x4` | `128` | `4` | `8` | 增加宽度 |
| `capacity_128x6` | `128` | `6` | `8` | 同时增加宽度和深度 |

**约束**：dim=128 时 batch size 可能需要从 4 降到 2 或降低 accumulation。

### 3.2 Loss 消融（Task 3.2）

| 变体名 | `single_step_swd_weight` | `swd_noise_sigma` | `single_step_edge_weight` | 说明 |
|---|---|---|---|---|
| `loss_swd0` | `0.0` | `0.02` | `0.1` | 关闭 SWD |
| `loss_swd2` | `2.0` | `0.02` | `0.1` | H7 建议 |
| `loss_swd8` | `8.0` | `0.02` | `0.1` | 当前最优 |
| `loss_swd16` | `16.0` | `0.02` | `0.1` | 突破 0.70 配置 |
| `loss_nosigma` | `8.0` | `0.0` | `0.1` | 关闭 NSWD 噪声 |
| `loss_edge0` | `8.0` | `0.02` | `0.0` | 关闭 edge loss |

### 3.3 DINO 与条件源消融（Task 3.3）

| 变体名 | `style_condition_source` | `style_dino_adapter_enabled` | `style_moe_enabled` | 说明 |
|---|---|---|---|---|
| `dino_baseline` | `target_dino_patches` | `false` | `false` | 当前最优 |
| `dino_adapter` | `target_dino_patches` | `true` | `false` | Round 1 无效，dim=64 复测 |
| `dino_moe` | `target_dino_patches` | `false` | `true` | Round 1 无效，dim=64 复测 |
| `intrinsic_latent` | `latent` | `false` | `false` | H6 路径，去除 DINO |

### 3.4 其他扩展维度

| 维度 | 变体示例 | 优先级 | 备注 |
|---|---|---|---|
| HF Residual | `velocity_hf_residual_enabled: true/false` | 低 | E2 P1 已证明几乎无效 |
| FiLM Init Std | `endpoint_film_init_std: 0.0/0.02` | 低 | H1 单独有效但不过门 |
| Text Conditioning | `style_text_enabled: true` | 低 | 白化通过后可恢复，需先过 WFI 检查 |
| Top-k Attention | `style_attn_topk: 4/12/16` | 低 | 代码已实现，历史无收益 |

---

## 4. 推荐的最小实验集合

考虑到本地 RTX 4070 资源，优先执行以下核心实验（约 12–16 个 smoke）：

1. `attn_softmax`
2. `attn_gated`（当前最优复跑）
3. `stylefilm_off`
4. `endpoint_velocity`
5. `endpoint_lowhigh_hd128`
6. `endpoint_lowhigh_hd256`
7. `gate_init005`
8. `gate_init05`
9. `capacity_128x4`
10. `loss_swd2`
11. `loss_nosigma`
12. `dino_adapter`
13. `intrinsic_latent`

每个实验训练 1 epoch，跑完整 WFI 评估。所有实验必须先过 `wfi_score < 0.40` 门，再比较 `clip_style` 和 `content_lpips`。

---

## 5. 配置命名规范

```
620_ablation_<维度>_<变体>_smoke
```

例如：
- `620_ablation_attn_softmax_smoke`
- `620_ablation_stylefilm_off_smoke`
- `620_ablation_endpoint_hd128_smoke`
- `620_ablation_capacity_128x4_smoke`

---

## 6. 评估与通过标准

| 优先级 | 标准 |
|---|---|
| P0 | `wfi_score < 0.40`（硬约束） |
| P1 | `clip_style ≥ 0.695` |
| P2 | `content_lpips < 0.36` |
| P3 | `ΔWFI` 尽可能小 |

任何变体若 `wfi_score ≥ 0.40`，直接标记为不通过，无需比较其他指标。

---

## 7. 配置生成脚本接口（建议）

建议实现 `tools/generate_ablation_configs.py`，输入消融矩阵，输出一组 JSON 配置：

```python
python tools/generate_ablation_configs.py \
  --base configs/620_spatial_bridge_targetlinear.json \
  --matrix docs/620/fog/ablation_audit/ablation_matrix.md \
  --outdir configs/ablations/
```

每个生成的配置继承 `620_spatial_bridge_targetlinear.json`，并覆盖对应的维度值，同时更新 `checkpoint.save_dir` 和 `ablation.name/axis/notes`。
