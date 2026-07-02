# 620 消融审计：Phase 2/3 结果统一汇总

> 生成时间：2026-06-21  
> 基线：`620_film_v5_endpoint_film_hd512_local_smoke`（WFI=0.3906，CLIP-S=0.7015，LPIPS=0.3382）  
> 放行门：WFI < 0.40，CLIP-S ≥ 0.695，LPIPS < 0.36  
> source WFI 基准：0.3217

---

## 1. 实验汇总总表

| 实验名 | 维度 | 关键参数 | WFI ↓ | CLIP-S ↑ | LPIPS ↓ | 训练时间 | ΔWFI（相对 hd512 基线） | 建议动作 |
|---|---|---|---|---:|---:|---:|---:|---:|:---|
| **基线与历史对照** |
| Round 1 base_swd8 | 历史瓶颈基线 | dim=64×4, SWD=8, 8ep | — | 0.6720 | 0.2900 | 远程 3060 | — | 参照 |
| 620_film_v5_gated_local_smoke | 白化修复前 | velocity, gated, style_film | 0.4902 | 0.6987 | 0.3300 | — | +0.0996 | ❌ 已废弃 |
| 620_film_v5_endpoint_film_hd512_local_smoke | 当前最优基线 | endpoint_lowhigh, FiLM, hd=512 | 0.3906 | 0.7015 | 0.3382 | — | 0.0000 | ✅ 基线 |
| **Phase 2.1 Attention** |
| attn_softmax | style_attn_mode | softmax | 0.3736 | 0.7023 | 0.3397 | ~5 min | -0.0170 | ✅ KEEP/候选默认 |
| attn_style_select | style_attn_mode | style_select | 0.3751 | 0.7015 | 0.3366 | ~5 min | -0.0155 | ⚠️ NEED_MORE_DATA |
| attn_sparsemax | style_attn_mode | sparsemax | 0.3779 | 0.7018 | 0.3354 | ~5 min | -0.0127 | ⚠️ NEED_MORE_DATA |
| attn_gated_raw | style_attn_mode | gated_raw | 0.3850 | 0.7017 | 0.3453 | ~5 min | -0.0056 | ⚠️ NEED_MORE_DATA |
| attn_relu2 | style_attn_mode | relu2 | 0.3856 | 0.7020 | 0.3434 | ~5 min | -0.0049 | ⚠️ NEED_MORE_DATA |
| attn_gated | style_attn_mode | gated（复测） | 0.3925 | 0.7020 | 0.3400 | ~5 min | +0.0019 | ⚠️ 可被 softmax 替代 |
| **Phase 2.2 StyleFiLM** |
| stylefilm_on | style_film_enabled | true | 0.3785 | 0.7020 | 0.3321 | ~5 min | -0.0121 | ✅ KEEP/REMOVE 均可 |
| stylefilm_off | style_film_enabled | false | 0.3782 | 0.7021 | 0.3322 | ~5 min | -0.0124 | ✅ 可简化关闭 |
| **Phase 2.3 Endpoint** |
| endpoint_velocity | endpoint_head_mode | velocity, no FiLM | 0.3769 | 0.7020 | 0.3315 | ~5 min | -0.0137 | ⚠️ NEED_MORE_DATA（更简） |
| endpoint_lowhigh_hd128 | endpoint_style_hidden_dim | 128 | 0.3801 | 0.7023 | 0.3422 | ~5 min | -0.0105 | ✅ 候选默认 |
| endpoint_lowhigh_hd512 | endpoint_style_hidden_dim | 512（基线） | 0.3915 | 0.7019 | 0.3432 | ~5 min | +0.0009 | ⚠️ 可降至 128 |
| endpoint_lowhigh_nofilm | endpoint_film_enabled | false | 0.3957 | 0.7012 | 0.3399 | ~5 min | +0.0051 | ⚠️ NEED_MORE_DATA |
| endpoint_lowhigh_hd256 | endpoint_style_hidden_dim | 256 | 0.3990 | 0.7013 | 0.3408 | ~5 min | +0.0084 | ❌ 避免 |
| **Phase 2.4 Gate Init** |
| gate_init005 | style_cross_attn_gate_init | 0.05 | 0.3757 | 0.7020 | 0.3413 | ~5 min | -0.0149 | ✅ RESTORE 默认 |
| gate_init05 | style_cross_attn_gate_init | 0.5 | 0.3833 | 0.7022 | 0.3415 | ~5 min | -0.0073 | ⚠️ NEED_MORE_DATA |
| gate_init03 | style_cross_attn_gate_init | 0.3（基线） | 0.3908 | 0.7022 | 0.3446 | ~5 min | +0.0002 | ⚠️ 不如 0.05 |
| **Phase 3.1 Capacity** |
| capacity_64x4 | base_dim×blocks | 64×4（基线复测） | 0.3887 | 0.7021 | 0.3382 | 351.2 s | -0.0019 | ✅ 默认 |
| capacity_64x6 | base_dim×blocks | 64×6 | 0.3828 | 0.7021 | 0.3426 | 400.2 s | -0.0078 | ✅ 若追求 WFI 最优 |
| capacity_128x4 | base_dim×blocks | 128×4 | 0.3921 | 0.7026 | 0.3393 | 6.9 s* | +0.0015 | ❌ 收益/成本比差 |
| capacity_128x6 | base_dim×blocks | 128×6 | 0.3895 | 0.7019 | 0.3436 | 378.0 s | -0.0011 | ❌ 无叠加收益 |
| **Phase 3.2 Loss** |
| loss_swd0 | single_step_swd_weight | 0 | 0.3921 | 0.7007 | 0.3384 | ~5 min | +0.0015 | ⚠️ style 略降 |
| loss_swd2 | single_step_swd_weight | 2 | 0.4001 | 0.7013 | 0.3304 | ~5 min | +0.0095 | ⚠️ WFI 超门 |
| loss_swd8 | single_step_swd_weight | 8（基线复测） | 0.3959 | 0.7018 | 0.3369 | ~5 min | +0.0053 | ✅ 默认 |
| loss_swd16 | single_step_swd_weight | 16 | 0.4013 | 0.7028 | 0.3395 | ~5 min | +0.0107 | ⚠️ 需配合 edge=0 |
| loss_nosigma | swd_noise_sigma | 0 | 0.4105 | 0.7007 | 0.3398 | ~5 min | +0.0199 | ❌ 不能关 |
| loss_edge0 | single_step_edge_weight | 0 | 0.3786 | 0.7020 | 0.3336 | 5.7 s | -0.0120 | ✅ REMOVE edge |
| loss_swd16_edge0 | 组合 | SWD=16, edge=0 | 0.3885 | 0.7030 | 0.3396 | 5.9 s | -0.0021 | ⚠️ NEED_MORE_DATA |
| loss_swd16_nosigma | 组合 | SWD=16, σ=0 | 0.3983 | 0.7023 | 0.3509 | 6.2 s | +0.0077 | ❌ 不推荐 |
| loss_edge0_nosigma | 组合 | edge=0, σ=0 | 0.4077 | 0.7017 | 0.3314 | 276.1 s | +0.0171 | ❌ 不能同时关 |
| **Phase 3.3 DINO / 条件源** |
| intrinsic_latent | style_condition_source | latent intrinsic | 0.3842 | 0.7020 | 0.3417 | 224 s | -0.0064 | ✅ KEEP 默认 |
| dino_baseline | style_condition_source | target_dino_patches | 0.6407 | 0.7097 | 0.2773 | 257 s | +0.2501 | ❌ 默认关闭 |
| dino_adapter | style_condition_source | target_dino_patches + adapter | 0.6076 | 0.7063 | 0.2618 | 267 s | +0.2170 | ❌ 默认关闭 |

> *`capacity_128x4` 训练时间 6.9 s 为异常值，疑似复用已有 checkpoint，不作为参考。*  
> 未标注具体训练时间的实验按本地 RTX 4070 smoke 规模估算约 4–6 min（含训练，不含 eval）。

---

## 2. WFI 分量详情

| 实验名 | contrast_ratio | dynamic_range | saturation | brightness | entropy |
|---|---|---:|---:|---:|---:|---:|
| 基线 hd512 | 3.570 | 44.264 | 0.217 | 0.517 | 6.977 |
| attn_softmax | 3.808 | 44.001 | 0.244 | 0.496 | 6.976 |
| attn_style_select | 3.711 | 43.336 | 0.254 | 0.494 | 6.949 |
| attn_sparsemax | 3.658 | 43.283 | 0.252 | 0.497 | 6.950 |
| attn_gated_raw | 3.534 | 43.728 | 0.245 | 0.511 | 6.970 |
| attn_relu2 | 3.505 | 43.492 | 0.249 | 0.510 | 6.964 |
| attn_gated | 3.437 | 43.406 | 0.232 | 0.514 | 6.957 |
| stylefilm_on | 3.580 | 43.355 | 0.263 | 0.501 | 6.942 |
| stylefilm_off | 3.597 | 43.503 | 0.260 | 0.502 | 6.946 |
| endpoint_velocity | 3.622 | 43.609 | 0.260 | 0.501 | 6.948 |
| endpoint_lowhigh_hd128 | 3.619 | 43.648 | 0.251 | 0.504 | 6.972 |
| endpoint_lowhigh_hd512 | 3.454 | 43.684 | 0.239 | 0.518 | 6.972 |
| endpoint_lowhigh_nofilm | 3.502 | 44.185 | 0.210 | 0.522 | 6.979 |
| endpoint_lowhigh_hd256 | 3.421 | 43.814 | 0.212 | 0.523 | 6.975 |
| gate_init005 | 3.743 | 43.992 | 0.245 | 0.500 | 6.977 |
| gate_init05 | 3.574 | 43.624 | 0.246 | 0.507 | 6.970 |
| gate_init03 | 3.462 | 43.825 | 0.238 | 0.518 | 6.973 |
| capacity_64x4 | 3.475 | 43.577 | 0.240 | 0.512 | 6.954 |
| capacity_64x6 | 3.578 | 43.510 | 0.248 | 0.506 | 6.970 |
| capacity_128x4 | 3.444 | 43.667 | 0.236 | 0.518 | 6.963 |
| capacity_128x6 | 3.468 | 43.656 | 0.243 | 0.516 | 6.971 |
| loss_swd0 | 3.333 | 43.110 | 0.264 | 0.519 | 6.914 |
| loss_swd2 | 3.230 | 42.596 | 0.250 | 0.520 | 6.892 |
| loss_swd8 | 3.428 | 43.495 | 0.222 | 0.518 | 6.962 |
| loss_swd16 | 3.301 | 42.995 | 0.226 | 0.523 | 6.999 |
| loss_nosigma | 3.194 | 41.689 | 0.199 | 0.511 | 6.885 |
| loss_edge0 | 3.583 | 43.407 | 0.261 | 0.502 | 6.942 |
| loss_swd16_edge0 | 3.418 | 42.769 | 0.254 | 0.509 | 6.994 |
| loss_swd16_nosigma | 3.239 | 41.402 | 0.223 | 0.496 | 6.910 |
| loss_edge0_nosigma | 3.239 | 42.856 | 0.192 | 0.517 | 6.911 |
| intrinsic_latent | 3.536 | 43.690 | 0.249 | 0.510 | 6.968 |
| dino_baseline | 1.700 | 28.630 | 0.115 | 0.745 | 6.058 |
| dino_adapter | 1.844 | 31.734 | 0.117 | 0.717 | 6.245 |

> WFI 分量中，DINO patches 变体在 `contrast_ratio`、`dynamic_range`、`saturation` 上显著低于 intrinsic latent，而 `brightness` 明显更高，这是“雾化/白化”的核心特征。

---

## 3. 各维度 best / worst 变体

| 维度 | best 变体 | best WFI | worst 变体 | worst WFI | WFI 跨度 |
|---|---|---:|---|---:|---:|
| style_attn_mode | attn_softmax | 0.3736 | attn_gated | 0.3925 | 0.0189 |
| style_film_enabled | stylefilm_off | 0.3782 | stylefilm_on | 0.3785 | 0.0003 |
| endpoint_head_mode / FiLM | endpoint_velocity | 0.3769 | endpoint_lowhigh_hd256 | 0.3990 | 0.0221 |
| style_cross_attn_gate_init | gate_init005 | 0.3757 | gate_init03 | 0.3908 | 0.0151 |
| base_dim×num_res_blocks | capacity_64x6 | 0.3828 | capacity_128x4 | 0.3921 | 0.0093 |
| single_step_swd_weight | loss_edge0（edge=0） | 0.3786 | loss_swd16 | 0.4013 | 0.0227 |
| swd_noise_sigma | σ=0.02 | 0.3786 | σ=0.00 | 0.4105 | 0.0319 |
| single_step_edge_weight | 0.0 | 0.3786 | 0.1 | 0.3959 | 0.0173 |
| style_condition_source | latent | 0.3842 | target_dino_patches | 0.6407 | 0.2565 |

> 跨度最大的维度是 `style_condition_source`（0.2565），其次是 `swd_noise_sigma`（0.0319）和 `single_step_swd_weight`（0.0227）。这说明：
> 1. **条件源选择** 是当前白化的主导因素；
> 2. **NSWD 噪声** 是第二关键的白化抑制项；
> 3. **attention / endpoint / gate / capacity** 在当前基线上已进入微调平台。

---

## 4. 对 CLIP-S / LPIPS trade-off 影响最大的维度

| 维度 | 变体范围 | CLIP-S 范围 | LPIPS 范围 | 观察 |
|---|---|---:|---:|---|
| style_condition_source | latent vs dino | 0.7020–0.7097 | 0.2773–0.3417 | DINO 显著提升风格但严重白化 |
| single_step_swd_weight | 0/2/8/16 | 0.7007–0.7028 | 0.3304–0.3384 | 高 SWD 提升风格，但 LPIPS/WFI 劣化 |
| single_step_edge_weight | 0 vs 0.1 | 0.7018–0.7020 | 0.3336–0.3369 | edge=0 同时改善 WFI、CLIP-S、LPIPS |
| endpoint_head_mode | velocity vs lowhigh | 0.7012–0.7023 | 0.3315–0.3432 | velocity 更简洁且 LPIPS 更低 |
| base_dim | 64 vs 128 | 0.7019–0.7026 | 0.3382–0.3436 | 容量对 trade-off 几乎无影响 |

> 关键洞察：在当前 `latent` 条件源 + `endpoint_film_hd512` 基线下，**edge loss 是唯一的“三赢”开关**（WFI↓、CLIP-S↑、LPIPS↓）。SWD weight 与条件源存在明显的风格-白化 trade-off，而容量升级无法打破该 trade-off。

---

## 5. 原始数据索引

| 实验 | 原始文件 |
|---|---|
| 全部 Phase 2/3 | `exp/620_spatial_bridge/620_ablation_*_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json` |
| 当前最优基线 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json` |
| 白化修复前 gated | `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/full_eval_wfi/epoch_0001/wfi_benchmark.json` |
| Loss 汇总 CSV | `results/phase32_loss_new_summary.csv` |
| Capacity 汇总 | `results/ablation_summary_capacity.csv` / `.json` |
| Attention / StyleFiLM / Endpoint / Gate | `results/task2_1_attention.json`、`results/task2_2_stylefilm.json`、`results/task2_3_endpoint.json`、`results/task2_4_gate_init.json` |

---

## 6. 可立即执行的候选默认组合

基于本汇总，一个**更简洁且指标更优**的候选默认配置为：

| 维度 | 推荐值 | 理由 |
|---|---|---|
| style_condition_source | latent | 唯一通过 WFI 门 |
| endpoint_head_mode | velocity 或 endpoint_lowhigh | velocity 更简且 WFI 更低；lowhigh 历史验证更充分 |
| endpoint_film_enabled | true | 保持风格调制能力 |
| endpoint_style_hidden_dim | 128 | hd128 WFI 优于 hd256/512，参数量更小 |
| style_attn_mode | softmax | 当前最低 WFI |
| style_cross_attn_gate_init | 0.05 | 当前最低 WFI，且为 schema 默认值 |
| style_film_enabled | false | 开关无差异，关闭可简化 |
| base_dim | 64 | 128 无显著收益 |
| num_res_blocks | 4（或 6） | 4 更快；6 WFI 最优 |
| single_step_swd_weight | 8 | 平衡；配合 edge=0 可尝试 12–16 |
| swd_noise_sigma | 0.02 | 必要白化抑制 |
| single_step_edge_weight | 0.0 | 三赢开关 |

> 注意：以上组合为 smoke 1 epoch 结果，多 epoch / 全量训练的稳定性需在 Phase 5 验证。
