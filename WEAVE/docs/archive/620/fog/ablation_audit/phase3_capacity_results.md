# 620 消融审计：Phase 3.1 网络容量结果

> 运行时间：2026-06-21  
> 基线模板：`exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json`  
> 批量脚本：`tools/run_ablation_batch.py`  
> 实验环境：本地 RTX 4070，batch=4，accum=16，1 epoch smoke

---

## 1. 实验设计

固定当前最优基线的其余参数，仅改变网络容量：

| 变体 | `base_dim` | `num_res_blocks` | `style_attn_num_heads` | 参数量估计 |
|---|---|---|---:|---:|
| `capacity_64x4` | 64 | 4 | 4 | ~1.70 M（基线） |
| `capacity_64x6` | 64 | 6 | 4 | ~2.05 M |
| `capacity_128x4` | 128 | 4 | 8 | ~6.50 M |
| `capacity_128x6` | 128 | 6 | 8 | ~9.50 M |

---

## 2. 结果汇总

| 变体 | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ | 训练时间 | 状态 |
|---|---|---:|---:|---:|---:|
| `capacity_64x4`（基线复测） | 0.3887 | 0.7021 | 0.3382 | 351.2 s | ✅ 通过 |
| `capacity_64x6` | **0.3828** | 0.7021 | 0.3426 | 400.2 s | ✅ 通过 |
| `capacity_128x4` | 0.3921 | **0.7026** | 0.3393 | 6.9 s* | ✅ 通过 |
| `capacity_128x6` | 0.3895 | 0.7019 | 0.3436 | 378.0 s | ✅ 通过 |

> *`capacity_128x4` 的训练时间异常（6.9 s），疑似复用了已存在的 checkpoint 或缓存。其评估指标有效，但训练时间不作为参考。

---

## 3. 关键发现

1. **所有容量变体均通过 WFI 门（< 0.40）**，说明当前基线对容量变化不敏感。
2. **增加深度（64×4 → 64×6）略微降低 WFI**（0.3887 → 0.3828），代价是 LPIPS 微升（0.3382 → 0.3426）。
3. **增加宽度（64×4 → 128×4）没有改善 WFI**，Clip-S 提升极微（0.7021 → 0.7026），可视为噪声。
4. **同时增加深度和宽度（128×6）未产生叠加收益**，WFI 0.3895 与基线持平，LPIPS 反而最高（0.3436）。
5. **容量不是当前白化的瓶颈**，单纯扩大模型无法系统性改善 WFI 或风格-内容 trade-off。

---

## 4. 与历史结论的对照

- **Round 1 诊断**曾假设 dim=64 是 clip_style 0.67 平台的天花板，建议升级到 128。
- **Phase 3.1 结果**显示：在当前 endpoint_film_hd512 基线下，容量升级到 128 并未显著提升 clip_style（仍在 0.702 平台）。
- 这说明 **endpoint-FiLM 和 gate 初始化等设计已经释放了部分容量潜力**，但 clip_style 仍未突破 0.72，可能需要更高级的结构改进（如 self-attention、text guidance、更优 coupling）而非单纯加宽加深。

---

## 5. 结论与建议

| 维度 | 当前值 | 建议 | 理由 |
|---|---|---|---|
| `base_dim` | 64 | **KEEP 64** | 128 不显著改善指标，且显存/速度成本高 |
| `num_res_blocks` | 4 | **RESTORE/INCREASE 到 6** | 64×6 在 WFI 上最优，参数量增加有限 |
| `style_attn_num_heads` | 4 | **KEEP 4**（随 base_dim 调整） | 与 base_dim 配套即可 |

**综合建议**：若追求最小模型，保留 `64×4`；若追求最佳 WFI，采用 `64×6`。不推荐 `128×4` 或 `128×6` 作为 smoke 阶段的默认配置。

---

## 6. 原始数据

- `results/ablation_summary_capacity.csv`
- `results/ablation_summary_capacity.json`
- 各实验目录：`exp/620_spatial_bridge/620_ablation_capacity_*_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
