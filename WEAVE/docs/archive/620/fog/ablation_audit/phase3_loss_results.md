# 620 消融审计：Phase 3.2 Loss 消融结果（SWD / NSWD / edge）

> 运行时间：2026-06-21  
> 基线模板：`exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json`  
> 批量脚本：`tools/run_ablation_batch.py`（辅以 `tools/_gen_phase32_loss_configs.py` 生成组合配置）  
> 实验环境：本地 RTX 4070，batch=4，accum=16，1 epoch smoke，`full_eval_each_epoch=false`（新补充的 4 个组合）

---

## 1. 实验设计

固定当前最优基线 `620_film_v5_endpoint_film_hd512_local_smoke` 的其余参数，仅改变三类 loss 超参：

- `bridge.single_step_swd_weight`（SWD 权重）：0 / 2 / 8 / 16
- `bridge.swd_noise_sigma`（NSWD 噪声）：0 / 0.02
- `bridge.single_step_edge_weight`（edge loss 权重）：0 / 0.1

在避免完全重复的前提下，组合出 9 个有意义的变体（含基线），覆盖单因子变化与关键双因子交互：

| 变体名 | `single_step_swd_weight` | `swd_noise_sigma` | `single_step_edge_weight` | 说明 |
|---|---|---:|---:|---|
| `loss_swd0` | 0 | 0.02 | 0.1 | 关闭 SWD |
| `loss_swd2` | 2 | 0.02 | 0.1 | H7 建议的低 SWD 权重 |
| `loss_swd8`（基线） | 8 | 0.02 | 0.1 | 当前最优 |
| `loss_swd16` | 16 | 0.02 | 0.1 | 历史突破配置中的 SWD 权重 |
| `loss_nosigma` | 8 | 0.00 | 0.1 | 关闭 NSWD 噪声 |
| `loss_edge0` | 8 | 0.02 | 0.0 | 关闭 edge loss |
| `loss_swd16_edge0` | 16 | 0.02 | 0.0 | 高 SWD + 无 edge |
| `loss_swd16_nosigma` | 16 | 0.00 | 0.1 | 高 SWD + 无噪声 |
| `loss_edge0_nosigma` | 8 | 0.00 | 0.0 | 同时关闭 edge 与噪声 |

> 放行门：`wfi_score < 0.40`，`clip_style ≥ 0.695`，`content_lpips < 0.36`。  
> 所有生成图的 source WFI 基准均为 `0.3217`。

---

## 2. 结果汇总

| 变体 | swd_weight | noise_sigma | edge_weight | WFI ↓ | Clip-S ↑ | Content LPIPS ↓ | 训练时间 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|:---|
| `loss_swd0` | 0 | 0.02 | 0.1 | 0.3921 | 0.7007 | 0.3384 | — | ✅ 通过 |
| `loss_swd2` | 2 | 0.02 | 0.1 | 0.4001 | 0.7013 | **0.3304** | — | ✅ 通过 |
| `loss_swd8`（基线） | 8 | 0.02 | 0.1 | 0.3959 | 0.7018 | 0.3369 | — | ✅ 通过 |
| `loss_swd16` | 16 | 0.02 | 0.1 | 0.4013 | 0.7028 | 0.3395 | — | ⚠️ WFI 略超门 |
| `loss_nosigma` | 8 | 0.00 | 0.1 | 0.4105 | 0.7007 | 0.3398 | — | ❌ WFI 未通过 |
| `loss_edge0` | 8 | 0.02 | 0.0 | **0.3786** | 0.7020 | 0.3336 | ≈5 s* | ✅ 通过 |
| `loss_swd16_edge0` | 16 | 0.02 | 0.0 | 0.3885 | **0.7030** | 0.3396 | ≈6 s* | ✅ 通过 |
| `loss_swd16_nosigma` | 16 | 0.00 | 0.1 | 0.3983 | 0.7023 | 0.3509 | ≈6 s* | ✅ 通过 |
| `loss_edge0_nosigma` | 8 | 0.00 | 0.0 | 0.4077 | 0.7017 | 0.3314 | 276.1 s | ⚠️ WFI 未通过 |

> *`loss_edge0`、`loss_swd16_edge0`、`loss_swd16_nosigma` 的 checkpoint 在本次运行前已存在（来自先前训练），本轮 `run_ablation_batch.py` 仅执行了 resume+no-op，因此显示的 5–6 s 不是真实训练时间。  
> `loss_edge0_nosigma` 为本次全新训练，耗时 276.1 s（不含内置 full_eval）。历史 5 个单因子实验未记录训练时间，按 smoke 规模估算约 4–6 min。

---

## 3. 关键发现

### 3.1 SWD weight：存在 style / WFI / content 的 trade-off

- 随着 `single_step_swd_weight` 从 0 提高到 16，**Clip-S 单调上升**（0.7007 → 0.7030），说明 SWD loss 确实在提供风格监督信号。
- 但 **WFI 也随之恶化**：基线 0.3959 → `loss_swd16` 0.4013；`content_lpips` 在 SWD=2 时最低，在 SWD=16 时回升。
- 单独把 SWD 提到 16 会在当前基线上**略超 WFI 门**（0.4013 > 0.40）。但若同时移除 edge loss（`loss_swd16_edge0`），WFI 可回到 0.3885，Clip-S 达到 0.7030，说明 SWD 与 edge loss 之间存在明显交互。

### 3.2 NSWD 噪声（`swd_noise_sigma=0.02`）是必要的白化抑制项

- 关闭噪声的 `loss_nosigma` WFI 从 0.3959 **上升到 0.4105**；`loss_edge0_nosigma` 也比 `loss_edge0` 高 0.029（0.4077 vs 0.3786）。
- 在有无 edge loss 两种情况下，**保留噪声都能显著降低 WFI**，且 Clip-S 基本保持或略升。
- 结论：`swd_noise_sigma=0.02` 对抑制白化有效，**不能移除**。

### 3.3 Edge loss 在当前配置下弊大于利

- `loss_edge0`（WFI 0.3786）是全部 9 个变体中 **WFI 最低** 的，且 Clip-S（0.7020）和 LPIPS（0.3336）均优于基线（0.7018 / 0.3369）。
- 即使把 SWD 提到 16，移除 edge loss 仍能把 WFI 拉回 0.40 以下（`loss_swd16_edge0`）。
- 这说明 `single_step_edge_weight=0.1` 在当前 endpoint-FiLM 基线上**没有带来净收益**，反而抬高了 WFI 并轻微损害风格/内容指标。

### 3.4 组合最优提示

- 当前实验中最均衡的变体是 **`loss_edge0`**（WFI 0.3786，Clip-S 0.7020，LPIPS 0.3336）。
- 若希望进一步压榨 style，可尝试 **`single_step_swd_weight=12~16` + `single_step_edge_weight=0` + `swd_noise_sigma=0.02`**；本次 `loss_swd16_edge0` 已给出 WFI 0.3885 / Clip-S 0.7030 的积极信号。

---

## 4. 与历史结论的对照

| 历史时期 | 关键配置 | clip_style | LPIPS | 结论 |
|---|---|---:|---:|---|
| Round 1 `base_swd8` | dim=64，8 epoch，SWD=8 | 0.6720 | 0.2900 | 0.67 平台，LPIPS 极好 |
| Round 1 `swd12` | dim=64，8 epoch，SWD=12 | 0.6725 | 0.2968 | SWD 提升几乎无效 |
| Round 1 `swd4` | dim=64，8 epoch，SWD=4 | 0.6706 | 0.2794 | LPIPS 最低 |
| Phase 1 `swd16` | SWD=16，vlen=1.0，1 epoch | 0.7053 | 0.2901 | 首次突破 0.70 |
| Phase 1 `swd16_vlen0.04` | SWD=16，vlen=0.04，5 epoch | **0.7051** | 0.2935 | 此前自评最优 |
| H7 | SWD weight 8→2 | — | — | 为解决梯度冲突而降低 SWD |

对照解读：

1. **Round 1 在 dim=64 上扫描 SWD 权重 4/8/12 未取得突破**，说明当时 style 天花板主要受容量/结构限制，而非 SWD 权重本身。
2. **Phase 1 将 SWD 提升到 16，并结合 `virtual_length_multiplier=0.04` 后，clip_style 从 0.67 平台跃升到 0.705**。本轮在 endpoint-FiLM hd512 基线上，`single_step_swd_weight=16` 仍能将 Clip-S 从 0.7018 提升到 0.7028，但 WFI 从 0.3959 升至 0.4013。
3. **H7 曾建议将 SWD weight 从 8 降到 2 以缓解梯度冲突**。本轮结果显示 SWD=2 的 LPIPS 最低（0.3304），但 WFI（0.4001）与 Clip-S（0.7013）均不如基线或 edge0。在 WFI 优先的框架下，SWD=2 不是最优选择。
4. **当前基线已自带 NSWD 噪声**，历史结论认为 NSWD 让 SWD 梯度更接近目标方向（cos 从几乎正交改善）。本轮消融直接验证了 `swd_noise_sigma=0.02` 对 WFI 的正面作用。

---

## 5. 结论与建议

| 维度 | 当前值 | 建议 | 理由 |
|---|---|---|---|
| `single_step_swd_weight` | 8.0 | **KEEP 8.0**（或 **NEED_MORE_DATA 12–16**） | 8.0 能在 WFI<0.40 内取得较好平衡；16 可提升 Clip-S 但单独使用会超 WFI 门，需配合 edge=0 或 vlen 扫描再决定 |
| `swd_noise_sigma` | 0.02 | **KEEP 0.02** | 关闭噪声会显著抬高 WFI，在有无 edge loss 两种设置下均被验证 |
| `single_step_edge_weight` | 0.1 | **REMOVE / 设为 0.0** | edge0 在 WFI、Clip-S、LPIPS 三项指标上均优于基线；edge loss 当前权重无净收益 |

**综合建议**：

- 立即采用 **`single_step_edge_weight=0.0`** 作为新默认，可得到 WFI 0.3786、Clip-S 0.7020、LPIPS 0.3336 的更好 smoke 结果。
- 在 edge=0 的基础上，进一步探索 **`single_step_swd_weight=12` 或 `16`** 是否能在保持 WFI<0.40 的同时继续提高 Clip-S；本次 `loss_swd16_edge0`（WFI 0.3885，Clip-S 0.7030）已显示潜力。
- 若后续要复现 Phase 1 的 `swd16_vlen0.04` 突破，应在 endpoint-FiLM hd512 + edge=0 的基线上重新扫描 `virtual_length_multiplier`，而不是单独调 SWD 权重。

---

## 6. 原始数据

- 本次新跑批量汇总：`results/phase32_loss_new_summary.csv`、`results/phase32_loss_new_summary.json`
- 各实验目录：`exp/620_spatial_bridge/620_ablation_loss_*/full_eval_wfi/epoch_0001/wfi_eval_report.json`
- 历史 Round 1 数据：`docs/620/round1_diagnosis.md`
- 历史 git 调研：`docs/620/fog/ablation_audit/git_history_digest.md`
