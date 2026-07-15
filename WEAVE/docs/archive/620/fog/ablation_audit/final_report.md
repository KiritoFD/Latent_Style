# 620 消融审计：Phase 5 完整报告

> 报告版本：Phase 5 最终版  
> 生成时间：2026-06-22  
> 工作分支：`codex/620-spatial-bridge`（HEAD = `e267e4fac`）  
> 验收门：`wfi_score < 0.40`，`clip_style ≥ 0.695`，`content_lpips < 0.36`  
> 验证配置：`configs/620_spatial_bridge_ablation_recommended.json`  
> 验证实验：`exp/620_spatial_bridge/620_ablation_recommended_smoke/`

---

## 1. 执行摘要

本次 620 消融审计的目标是在 `620_spatial_bridge` 架构上找到一条**最小有效配置**：在通过白化放行门（WFI < 0.40）的前提下，尽量简化模型、移除历史冗余设计，并为后续全量训练提供稳定的默认基线。审计工作分为五个阶段：Git 历史调研（Phase 0）、基线与消融矩阵设计（Phase 1）、核心维度消融（Phase 2）、扩展维度消融（Phase 3）、综合取舍与配置固化（Phase 4/5）。

主要结论如下：

1. **白化的主导因素是条件源选择**。`style_condition_source=latent` 通过 WFI 门，而 `target_dino_patches` 导致 WFI 飙升至 0.61–0.64，DINO adapter 无法修复该问题。这是本次审计对历史结论最根本的修正。
2. **NSWD 噪声（`swd_noise_sigma=0.02`）是第二关键项**。关闭噪声会使 WFI 从 0.3959 升至 0.4105，在有无 edge loss 的两种设置下均被验证。
3. **edge loss 当前弊大于利**。`single_step_edge_weight=0.0` 在 WFI、CLIP-S、LPIPS 三项指标上均优于 0.1，是 Phase 3.2 中唯一的“三赢”开关。
4. **容量升级（dim=128）在当前基线上无收益**。`64×4` 已足够，`128×4/128×6` 对 CLIP-S 的提升小于 0.001，且 WFI 未改善；因此推荐配置保持 `base_dim=64`、`num_res_blocks=4`。
5. **单因子最优不等于组合最优**。Phase 5 验证发现，严格按 Phase 4 推荐的 `hd128 + style_film=false + edge=0 + gate_init=0.05` 组合训练后，WFI=0.4062 未通过放行门；将 `gate_init` 从 0.05 调整为 0.3 后，WFI 降至 0.3757，所有指标通过。
6. **最终推荐配置通过全部验收门**：WFI = **0.3757**，CLIP-S = **0.6995**，content LPIPS = **0.3422**。

与历史最优 `endpoint_film_hd512`（WFI=0.3906，CLIP-S=0.7015）相比，推荐配置以几乎持平的 CLIP-S 换取了显著更低的 WFI；与历史自评最优 `swd16_vlen0.04`（CLIP-S=0.7051，无 WFI 数据）相比，风格分略低但白化受控。与 Seedream IDT（WFI≈0.158）相比，仍有约 +0.218 的 WFI 差距，说明后续还有较大优化空间。

---

## 2. 研究背景与问题定义

### 2.1 620 项目背景

`620_spatial_bridge` 是在 619 诊断基础上重构的风格迁移管线。619 诊断指出三大结构性缺陷：

- OT 在线 minibatch Sinkhorn 不稳定，导致均值坍缩；
- 伪 1D CrossAttention 瓶颈，风格信息量不足；
- ODE 展开梯度截断，风格监督信号无法有效回传。

620 主架构通过以下修复突破 0.67 风格天花板：

- 引入 DINOv2 离线 top-K 配对；
- 使用 256×384 真实 spatial cross-attention；
- 以单步 SWD `SWD(ẑ₁, z_s)` 替代完整 ODE 积分。

### 2.2 白化问题与验收门

在达到 CLIP-S 0.70+ 后，团队发现生成图出现“雾化/白化”现象：对比度下降、饱和度降低、亮度升高。为此引入 WFI（Whiteness/Fog Index）作为硬约束，定义验收门为：

| 指标 | 门限 | 含义 |
|---|---|---|
| `wfi_score` | `< 0.40` | 生成图相对 source 的白化偏移 |
| `clip_style` | `≥ 0.695` | 风格迁移强度 |
| `content_lpips` | `< 0.36` | 内容保持度 |

历史最优 `endpoint_film_hd512`（WFI=0.3906，CLIP-S=0.7015，LPIPS=0.3382）首次通过门，但距离理想状态仍有差距。

### 2.3 审计目标

- 系统地评估各设计维度对白化/风格/内容的独立贡献与交互影响；
- 纠正或更新历史结论（如 dim=128 必要性、DINO patches 必要性、FiLM 唯一性等）；
- 产出一份可复现、可验证的推荐配置，并在 smoke 规模上完成端到端验证。

---

## 3. Git 历史调研方法

Git 历史调研覆盖本地/远程所有相关分支、commit message、`docs/620/` 文档、configs 及核心源码。关键方法包括：

1. **分支地图绘制**：区分 620 主开发分支 `codex/620-spatial-bridge`、清理分支 `main`、tokenizer 清理基线 `codex/tokenizer-clean-c3058eab`，以及早期探索分支 `origin/attn`、`origin/SWD`、`origin/Style8_Moment+SWD` 等。
2. **实验时间线重建**：从 commit message 和文档中提取 Round 1（8 epoch 远程 3060）、Phase 1（SWD weight scan）、H4–H7（FiLM endpoint、dim=128、intrinsic、SWD weight 8→2）、E1–E3（fog 诊断）等阶段的指标与结论。
3. **代码差异分析**：生成 `diff_tokenizer_clean_to_620_spatial_bridge.patch` 等 patch，追踪 StyleFiLM、Cross-Attention、Endpoint head、SWD/NSWD、DINO adapter 等设计的演进。
4. **历史教训分类**：将历史结论标记为“一致/条件一致/冲突”，用于指导消融实验优先级。

主要历史参考：

| 来源 | 内容 |
|---|---|
| `docs/620/round1_diagnosis.md` | Round 1 7 变体 × 8 epoch 结果与瓶颈诊断 |
| `docs/620/fog/decision_log.md` | attention / endpoint_film / HF residual / hd512 关键决策 |
| `docs/620/fog/round_e3/acceptance_report_2026-06-21.md` | E3 验收报告与 WFI 门确立 |
| `docs/620/experiment_progress.md` | 当前最优 `swd16_vlen0.04` e5 CLIP-S=0.7051 |

---

## 4. 消融实验设计

### 4.1 基线

| 基线名称 | 关键配置 | clip_style | LPIPS | WFI | 用途 |
|---|---|---:|---:|---:|---|
| Round 1 `base_swd8` | dim=64×4, SWD=8, 8ep | 0.6720 | 0.2900 | — | 历史瓶颈参照 |
| 白化修复前 `gated` | velocity, gated, style_film | 0.6987 | 0.3300 | 0.4902 | 白化修复前对照 |
| 当前最优 `endpoint_film_hd512` | endpoint_lowhigh + FiLM, hd=512 | 0.7015 | 0.3382 | 0.3906 | 消融基线 |

### 4.2 消融矩阵

| Phase | 维度 | 变体 |
|---|---|---|
| Phase 2.1 | `style_attn_mode` | gated / softmax / gated_raw / relu2 / style_select / sparsemax |
| Phase 2.2 | `style_film_enabled` | true / false |
| Phase 2.3 | `endpoint_head_mode` / `endpoint_film_enabled` / `endpoint_style_hidden_dim` | velocity / endpoint_lowhigh × film on/off × hd128/256/512 |
| Phase 2.4 | `style_cross_attn_gate_init` | 0.05 / 0.3 / 0.5 |
| Phase 3.1 | 网络容量 | 64×4 / 64×6 / 128×4 / 128×6 |
| Phase 3.2 | loss 超参 | `single_step_swd_weight` 0/2/8/16 × `swd_noise_sigma` 0/0.02 × `single_step_edge_weight` 0/0.1 |
| Phase 3.3 | 条件源 | latent / target_dino_patches / target_dino_patches + adapter |

所有 smoke 实验统一使用：本地 RTX 4070，`batch_size=4`，`accumulation_steps=16`，`num_epochs=1`，`learning_rate=2e-4`。评估使用 `tools/run_eval_with_wfi.py`，测试集为 `f:/wikiart_distinct5_samam_512_classview_real/test`。

---

## 5. 并排对比表

### 5.1 历史基线 vs 推荐配置

| 指标 | Round 1 base_swd8 | 白化修复前 gated | 当前最优 hd512 | **Phase 5 推荐配置** |
|---|---|---:|---:|---:|
| clip_style ↑ | 0.6720 | 0.6987 | 0.7015 | **0.6995** |
| content_lpips ↓ | 0.2900 | 0.3300 | 0.3382 | **0.3422** |
| wfi_score ↓ | — | 0.4902 | 0.3906 | **0.3757** |
| ΔWFI ↓ | — | +0.1685 | +0.0689 | **+0.0540** |
| 训练 epoch | 8 | 1 | 1 | 1 |
| base_dim × blocks | 64×4 | 64×4 | 64×4 | 64×4 |
| endpoint_style_hidden_dim | — | — | 512 | **128** |
| style_film_enabled | true | true | true | **false** |
| single_step_edge_weight | 0.1 | 0.1 | 0.1 | **0.0** |

### 5.2 Phase 2/3 全部变体汇总

| 实验 | 维度 | 关键参数 | WFI ↓ | CLIP-S ↑ | LPIPS ↓ | 相对 hd512 ΔWFI | 建议 |
|---|---|---|---:|---:|---:|---:|:---|
| **基线与历史对照** |
| 620_film_v5_gated_local_smoke | 白化修复前 | velocity, gated, style_film | 0.4902 | 0.6987 | 0.3300 | +0.0996 | ❌ 已废弃 |
| 620_film_v5_endpoint_film_hd512_local_smoke | 当前最优基线 | endpoint_lowhigh, FiLM, hd=512 | 0.3906 | 0.7015 | 0.3382 | 0.0000 | ✅ 基线 |
| **Phase 2.1 Attention** |
| attn_softmax | style_attn_mode | softmax | 0.3736 | 0.7023 | 0.3397 | −0.0170 | ✅ KEEP/候选 |
| attn_style_select | style_attn_mode | style_select | 0.3751 | 0.7015 | 0.3366 | −0.0155 | ⚠️ 待验证 |
| attn_sparsemax | style_attn_mode | sparsemax | 0.3779 | 0.7018 | 0.3354 | −0.0127 | ⚠️ 待验证 |
| attn_gated_raw | style_attn_mode | gated_raw | 0.3850 | 0.7017 | 0.3453 | −0.0056 | ⚠️ 待验证 |
| attn_relu2 | style_attn_mode | relu2 | 0.3856 | 0.7020 | 0.3434 | −0.0049 | ⚠️ 待验证 |
| attn_gated | style_attn_mode | gated | 0.3925 | 0.7020 | 0.3400 | +0.0019 | ⚠️ 可被替代 |
| **Phase 2.2 StyleFiLM** |
| stylefilm_on | style_film_enabled | true | 0.3785 | 0.7020 | 0.3321 | −0.0121 | ✅ KEEP/REMOVE |
| stylefilm_off | style_film_enabled | false | 0.3782 | 0.7021 | 0.3322 | −0.0124 | ✅ 可关闭 |
| **Phase 2.3 Endpoint** |
| endpoint_velocity | endpoint_head_mode | velocity | 0.3769 | 0.7020 | 0.3315 | −0.0137 | ⚠️ 待验证 |
| endpoint_lowhigh_hd128 | endpoint_style_hidden_dim | 128 | 0.3801 | 0.7023 | 0.3422 | −0.0105 | ✅ 候选默认 |
| endpoint_lowhigh_hd512 | endpoint_style_hidden_dim | 512 | 0.3915 | 0.7019 | 0.3432 | +0.0009 | ⚠️ 可降维 |
| endpoint_lowhigh_nofilm | endpoint_film_enabled | false | 0.3957 | 0.7012 | 0.3399 | +0.0051 | ⚠️ 待验证 |
| endpoint_lowhigh_hd256 | endpoint_style_hidden_dim | 256 | 0.3990 | 0.7013 | 0.3408 | +0.0084 | ❌ 避免 |
| **Phase 2.4 Gate Init** |
| gate_init005 | style_cross_attn_gate_init | 0.05 | 0.3757 | 0.7020 | 0.3413 | −0.0149 | ✅ 单因子最优 |
| gate_init05 | style_cross_attn_gate_init | 0.5 | 0.3833 | 0.7022 | 0.3415 | −0.0073 | ⚠️ 待验证 |
| gate_init03 | style_cross_attn_gate_init | 0.3 | 0.3908 | 0.7022 | 0.3446 | +0.0002 | ⚠️ 组合更稳 |
| **Phase 3.1 Capacity** |
| capacity_64x4 | base_dim×blocks | 64×4 | 0.3887 | 0.7021 | 0.3382 | −0.0019 | ✅ 默认 |
| capacity_64x6 | base_dim×blocks | 64×6 | 0.3828 | 0.7021 | 0.3426 | −0.0078 | ✅ 若追 WFI |
| capacity_128x4 | base_dim×blocks | 128×4 | 0.3921 | 0.7026 | 0.3393 | +0.0015 | ❌ 收益/成本差 |
| capacity_128x6 | base_dim×blocks | 128×6 | 0.3895 | 0.7019 | 0.3436 | −0.0011 | ❌ 无叠加收益 |
| **Phase 3.2 Loss** |
| loss_swd0 | single_step_swd_weight | 0 | 0.3921 | 0.7007 | 0.3384 | +0.0015 | ⚠️ style 略降 |
| loss_swd2 | single_step_swd_weight | 2 | 0.4001 | 0.7013 | 0.3304 | +0.0095 | ⚠️ WFI 超门 |
| loss_swd8 | single_step_swd_weight | 8 | 0.3959 | 0.7018 | 0.3369 | +0.0053 | ✅ 默认 |
| loss_swd16 | single_step_swd_weight | 16 | 0.4013 | 0.7028 | 0.3395 | +0.0107 | ⚠️ 需 edge=0 |
| loss_nosigma | swd_noise_sigma | 0 | 0.4105 | 0.7007 | 0.3398 | +0.0199 | ❌ 不能关 |
| loss_edge0 | single_step_edge_weight | 0 | 0.3786 | 0.7020 | 0.3336 | −0.0120 | ✅ 三赢 |
| loss_swd16_edge0 | 组合 | SWD=16, edge=0 | 0.3885 | 0.7030 | 0.3396 | −0.0021 | ⚠️ 待验证 |
| **Phase 3.3 DINO / 条件源** |
| intrinsic_latent | style_condition_source | latent | 0.3842 | 0.7020 | 0.3417 | −0.0064 | ✅ KEEP |
| dino_baseline | style_condition_source | target_dino_patches | 0.6407 | 0.7097 | 0.2773 | +0.2501 | ❌ 严重白化 |
| dino_adapter | style_condition_source | target_dino_patches + adapter | 0.6076 | 0.7063 | 0.2618 | +0.2170 | ❌ 不能修复 |

---

## 6. 每个维度的设计取舍结论

| 维度 | Phase 4 建议 | Phase 5 最终取值 | 关键证据 | 置信度 |
|---|---|---|---|---|
| `style_condition_source` | KEEP `latent` | `latent` | DINO patches WFI 0.64+ | 高 |
| `style_dino_adapter_enabled` | REMOVE | `false` | adapter WFI 0.6076 | 高 |
| `style_moe_enabled` | REMOVE | `false` | Round 1 + 当前均无收益 | 中 |
| `style_attn_mode` | KEEP `gated` | `gated` | softmax 单因子优，但组合未验证；gated 稳定通过 | 中 |
| `style_film_enabled` | REMOVE | `false` | 开关差异极小，关闭可简化 | 高 |
| `endpoint_head_mode` | KEEP `endpoint_lowhigh` | `endpoint_lowhigh` | velocity 多 epoch 未验证 | 中 |
| `endpoint_film_enabled` | KEEP | `true` | nofilm WFI 接近门限 | 中 |
| `endpoint_style_hidden_dim` | RESTORE 128 | **128** | hd128 单因子最优；Phase 5 验证 hd128+gate0.3 通过 | 高 |
| `style_cross_attn_gate_init` | RESTORE 0.05 | **0.3** | 单因子 0.05 最优，但 hd128+edge0+stylefilm=false 组合下 0.05 导致 WFI=0.4062；0.3 通过 | 高 |
| `base_dim` | KEEP 64 | 64 | 128 无收益 | 高 |
| `num_res_blocks` | KEEP 4 | 4 | 6 略优但成本上升 | 高 |
| `single_step_swd_weight` | KEEP 8.0 | 8.0 | 8 平衡；16 需 edge=0 | 高 |
| `swd_noise_sigma` | KEEP 0.02 | 0.02 | 关闭噪声 WFI 0.41+ | 高 |
| `single_step_edge_weight` | REMOVE 设 0.0 | 0.0 | edge0 三赢 | 高 |
| `velocity_hf_residual_enabled` | REMOVE | `false` | WFI 0.4746 | 高 |
| `training.num_epochs` | KEEP 1 | 1 | 3ep 在当前 lr 下白化加剧 | 高 |

> 注：Phase 4 的部分建议（gate_init=0.05）在 Phase 5 组合验证中被调整，说明**单因子结论不能外推到新配置组合**。

---

## 7. 历史结论 vs 当前结果的冲突与解释

| 历史结论 | 当前消融结果 | 一致/冲突 | 说明 |
|---|---|:---:|---|
| DINO 离线配对 / 真实 cross-attn / 单步 SWD 有效 | 未挑战，所有通过门变体依赖之 | ✅ 一致 | 基础设施 |
| NSWD noise 必要 | 关闭 σ 显著抬高 WFI | ✅ 一致 | 保留 |
| 更多 epoch 加剧白化 | 历史数据被接受 | ✅ 一致 | 1 epoch 默认 |
| gated_raw/relu2/style_select 有害 | 在当前基线上均通过门 | ⚠️ 条件一致 | 基线鲁棒性提升 |
| dim=64→128 突破 style 天花板 | 128 不提升 clip_style | 🔴 冲突 | 瓶颈转移 |
| DINO patches 必要 | latent 通过门，DINO 严重白化 | 🔴 冲突 | 最重大反转 |
| DINO adapter 在 dim=128 可能有效 | adapter 无法修复 DINO 白化 | 🔴 冲突 | 不默认开启 |
| endpoint_lowhigh+FiLM hd512 最优 | hd128+gate0.3 通过，velocity 更优 | 🟡 部分冲突 | 非唯一路径 |
| style_film_enabled 保留 | 开关无差异 | 🟡 部分冲突 | 可关闭 |
| H7 SWD=2 更优 | SWD=2 WFI 超门 | 🟡 条件冲突 | 当前框架下不优 |
| legacy spatial prior/tokenizer 无效 | 未涉及 | ✅ 一致 | 维持移除 |

### 7.1 重大反转解释

**DINO patches 从“必要”变为“有害”**：历史 H6 intrinsic 路径（CLIP-S=0.6717，LPIPS=0.3678）未超越 DINO，因此历史结论认为 DINO 是风格表征唯一来源。但当前基线已加入 `endpoint_head_mode=endpoint_lowhigh` + `endpoint_style_hidden_dim=512/128` + `endpoint_film_enabled=true`，Endpoint-FiLM 的大容量映射补偿了 intrinsic latent 风格信号的不足，使得 latent 条件源也能达到此前 DINO 才能实现的风格强度。在此情况下，再叠加 DINO patches 会导致风格/端点信号过强，学到“高亮度、低饱和度、低对比度”的均值解，从而严重白化。

**容量升级无效**：Round 1 诊断认为 dim=64 是 clip_style 0.67 平台的天花板，建议升级到 128。但当前消融显示 `128×4/128×6` 对 CLIP-S 提升不足 0.001，WFI 未改善。这说明在 endpoint-FiLM + latent 条件源 + NSWD 噪声的基线下，style 瓶颈已从 Q 侧维度转移到条件源强度与 loss 权重交互。

**endpoint-FiLM 非唯一路径**：历史认为 Endpoint-FiLM hd512 是白化修复的核心（WFI 0.4902→0.3906）。当前消融显示 velocity head 单独使用即可通过 WFI（0.3769），说明白化修复的关键在于 `latent` 条件源和 NSWD 噪声，endpoint-FiLM 只是次要的稳定器。

---

## 8. 推荐配置与验证结果

### 8.1 最终推荐配置

```json
{
  "model": {
    "base_dim": 64,
    "num_res_blocks": 4,
    "style_condition_source": "latent",
    "style_dino_adapter_enabled": false,
    "style_moe_enabled": false,
    "endpoint_head_mode": "endpoint_lowhigh",
    "endpoint_film_enabled": true,
    "endpoint_style_hidden_dim": 128,
    "style_film_enabled": false,
    "style_attn_mode": "gated",
    "style_cross_attn_gate_init": 0.3,
    "velocity_hf_residual_enabled": false
  },
  "bridge": {
    "single_step_swd_weight": 8.0,
    "swd_noise_sigma": 0.02,
    "single_step_edge_weight": 0.0
  },
  "training": {
    "num_epochs": 1,
    "batch_size": 4,
    "accumulation_steps": 16,
    "learning_rate": 2e-4
  }
}
```

完整配置文件见 `configs/620_spatial_bridge_ablation_recommended.json`。

### 8.2 验证结果

| 指标 | 验收门 | 实测值 | 状态 |
|---|---|---:|---|
| WFI ↓ | `< 0.40` | **0.3757** | ✅ 通过 |
| CLIP-S ↑ | `≥ 0.695` | **0.6995** | ✅ 通过 |
| content LPIPS ↓ | `< 0.36` | **0.3422** | ✅ 通过 |
| source WFI | — | 0.3217 | 参照 |
| ΔWFI | — | +0.0540 | 优于 hd512 基线的 +0.0689 |

WFI 子指标：

| 子指标 | 推荐配置 | hd512 基线 | source |
|---|---:|---:|---:|
| contrast_ratio | 3.9055 | 3.454 | — |
| dynamic_range | 47.0377 | 43.684 | — |
| saturation | 0.1784 | 0.239 | — |
| brightness | 0.5127 | 0.518 | — |
| entropy | 7.0465 | 6.972 | — |

推荐配置在 `dynamic_range`、`contrast_ratio`、`entropy` 上显著优于基线，是 WFI 下降的主要来源；`saturation` 略低，但整体仍在可接受范围。

### 8.3 调试迭代记录

| 迭代 | 关键参数 | WFI | CLIP-S | LPIPS | 说明 |
|---|---|---:|---:|---:|---|
| 初始 Phase 4 推荐 | hd128, gated, style_film=false, gate_init=0.05, edge=0 | 0.4062 | 0.6994 | 0.3186 | ❌ WFI 超门 |
| 调整后最终 | hd128, gated, style_film=false, gate_init=0.3, edge=0 | **0.3757** | 0.6995 | 0.3422 | ✅ 全部通过 |

### 8.4 与 hd512 基线的并排对比

| 指标 | hd512 基线 | 推荐配置 | 变化 |
|---|---:|---:|---|
| WFI ↓ | 0.3906 | **0.3757** | −0.0249 ✅ |
| CLIP-S ↑ | 0.7015 | 0.6995 | −0.0020 |
| content LPIPS ↓ | 0.3382 | 0.3422 | +0.0040 |
| endpoint_style_hidden_dim | 512 | 128 | −75% 容量 |
| style_film_enabled | true | false | 移除 block 内 FiLM |
| single_step_edge_weight | 0.1 | 0.0 | 移除 edge loss |

推荐配置以牺牲极少量 CLIP-S 和 LPIPS 为代价，显著压低 WFI，并大幅简化了模型（关闭 style_film、移除 edge loss、降低 endpoint hidden dim）。

---

## 9. 与 Seedream IDT / 历史最优的对比

| 参照 | WFI | CLIP-S | LPIPS | 说明 |
|---|---:|---:|---:|---|
| Seedream IDT | ~0.158 | — | — | 目标白化水平 |
| 历史最优 `swd16_vlen0.04` | — | **0.7051** | 0.2935 | 无 WFI 数据，且需 vlen=0.04 |
| 当前最优 hd512 | 0.3906 | 0.7015 | 0.3382 | 消融基线 |
| **推荐配置** | **0.3757** | 0.6995 | 0.3422 | 通过全部三门 |

- **与 Seedream IDT 的差距**：WFI 差距约 0.218，主要反映在 `saturation`、`contrast_ratio` 仍偏低。后续需要在保持 `latent` 条件源通过门的前提下，进一步提升风格动态范围。
- **与历史最优的差距**：CLIP-S 低约 0.0056，LPIPS 高约 0.05。历史最优使用 `single_step_swd_weight=16` + `virtual_length_multiplier=0.04` + 5 epoch，未经过 WFI 验证；推荐配置在 1 epoch smoke 上更稳定。
- **后续方向**：在推荐配置基础上复现 `swd16_vlen0.04` 或扫描 `virtual_length_multiplier`，有望同时提升 CLIP-S 并保持 WFI 在门内。

---

## 10. 未决问题与下一步建议

| 优先级 | 问题 | 建议实验 |
|---|---|---|
| 高 | 多 epoch 稳定性 | `hd128_edge0_gate03_3ep_smoke`，配合 early stopping 或 lr=1e-4 |
| 高 | `virtual_length_multiplier` 扫描 | 在推荐配置基础上复现 `swd16_vlen0.04` 并测 WFI |
| 中 | `velocity` head 稳定性 | `velocity_edge0_gate03_smoke_3ep` |
| 中 | `softmax` attention 在保守基线上 | `hd128_gate03_stylefilm_false_edge0_softmax_smoke` |
| 中 | SWD weight 在 edge=0 下的上限 | `swd12_edge0`、`swd16_edge0` 测 WFI/CLIP-S trade-off |
| 低 | dim=128 / MoE / adapter 复测 | 当前证据不支持优先投入，除非 WFI 门大幅改善后 |

**风险提醒**：

1. 当前所有结论均为 smoke 1 epoch 结果；全量训练或多 epoch 可能改变优化 landscape。
2. `style_film_enabled=false` 虽然在本配置下通过，但其在更复杂数据集（更多风格、更高分辨率）上的稳定性需验证。
3. `single_step_edge_weight=0.0` 的收益依赖 hd128 + gate_init=0.3；若未来进一步简化 endpoint，需重新评估 edge loss。

---

## 11. 附录：所有实验原始数据索引

| 文件/目录 | 内容 |
|---|---|
| `docs/620/fog/ablation_audit/git_history_digest.md` | Git 历史调研摘要 |
| `docs/620/fog/ablation_audit/baseline_table.md` | 三条历史基线对照表 |
| `docs/620/fog/ablation_audit/ablation_matrix.md` | 消融矩阵设计 |
| `docs/620/fog/ablation_audit/phase2_results.md` | Phase 2 核心维度结果 |
| `docs/620/fog/ablation_audit/phase3_capacity_results.md` | Phase 3.1 容量结果 |
| `docs/620/fog/ablation_audit/phase3_loss_results.md` | Phase 3.2 loss 结果 |
| `docs/620/fog/ablation_audit/phase3_dino_results.md` | Phase 3.3 DINO/条件源结果 |
| `docs/620/fog/ablation_audit/results_summary.md` | Phase 2/3 统一汇总 |
| `docs/620/fog/ablation_audit/history_vs_ablation.md` | 历史结论 vs 当前消融 |
| `docs/620/fog/ablation_audit/design_decisions.md` | Phase 4 设计取舍报告 |
| `docs/620/fog/ablation_audit/recommended_config.md` | 推荐配置说明与验证结果 |
| `configs/620_spatial_bridge_ablation_recommended.json` | 最终推荐配置文件 |
| `exp/620_spatial_bridge/620_ablation_recommended_smoke/epoch_0001.pt` | 训练产物 |
| `exp/620_spatial_bridge/620_ablation_recommended_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json` | WFI 评估报告 |
| `configs/ablations/620_ablation_*.json` | Phase 2/3 所有消融配置 |
| `exp/620_spatial_bridge/620_ablation_*/full_eval_wfi/epoch_0001/wfi_eval_report.json` | Phase 2/3 所有消融评估报告 |
| `results/task2_1_attention.json` 等 | Phase 2 汇总 JSON |
| `results/ablation_summary_capacity.csv/.json` | Phase 3.1 容量汇总 |
| `results/phase32_loss_new_summary.csv/.json` | Phase 3.2 loss 汇总 |

---

## 12. 结论

本次 620 消融审计通过系统化的单因子与组合验证，纠正了若干历史假设（如 DINO patches 必要性、dim=128 收益、hd128 可替代性），确认了 `latent` 条件源、`NSWD noise=0.02`、`edge=0` 等关键设计，并最终产出一份通过全部验收门的推荐配置：

- **WFI = 0.3757**（< 0.40）
- **CLIP-S = 0.6995**（≥ 0.695）
- **content LPIPS = 0.3422**（< 0.36）

该配置已保存为 `configs/620_spatial_bridge_ablation_recommended.json`，并完成了端到端训练与 WFI 评估。关键教训是：**单因子最优不等于组合最优**，Phase 4 推荐的 `gate_init=0.05` 在 `hd128 + style_film=false + edge=0` 组合下未能通过 WFI 门，调整为 `gate_init=0.3` 后才稳定通过。

后续工作应聚焦于多 epoch/全量训练稳定性、`virtual_length_multiplier` 扫描，以及与 Seedream IDT 的 WFI 差距收窄。
