# 620 消融审计：设计取舍报告

> 生成时间：2026-06-21  
> 依据：`docs/620/fog/ablation_audit/results_summary.md`、`docs/620/fog/ablation_audit/history_vs_ablation.md`、`docs/620/fog/ablation_audit/git_history_digest.md`  
> 放行门：WFI < 0.40，CLIP-S ≥ 0.695，LPIPS < 0.36

---

## 1. 决策总览

| 维度 | 决策 | 推荐值 | 置信度 |
|---|---|---|---|
| `style_attn_mode` | KEEP（可迁移到 softmax） | `gated` 或 `softmax` | 中 |
| `style_film_enabled` | REMOVE（默认关闭） | `false` | 高 |
| `endpoint_head_mode` | NEED_MORE_DATA（velocity 有潜力） | `endpoint_lowhigh`（保守）/ `velocity`（激进） | 中 |
| `endpoint_film_enabled` | KEEP | `true` | 中 |
| `endpoint_style_hidden_dim` | RESTORE/降低 | `128` | 高 |
| `style_cross_attn_gate_init` | RESTORE | `0.05` | 高 |
| `base_dim` | KEEP | `64` | 高 |
| `num_res_blocks` | KEEP（或微调到 6） | `4`（默认）/ `6`（WFI 优先） | 高 |
| `single_step_swd_weight` | KEEP（配合 edge=0 可提升） | `8.0` | 高 |
| `swd_noise_sigma` | KEEP | `0.02` | 高 |
| `single_step_edge_weight` | REMOVE | `0.0` | 高 |
| `style_condition_source` | KEEP | `latent` | 高 |
| `style_dino_adapter_enabled` | REMOVE | `false` | 高 |
| `style_moe_enabled` | REMOVE | `false` | 中 |
| `training.num_epochs` | KEEP（默认 1） | `1`（或带 early stopping 的 3） | 高 |
| `velocity_hf_residual_enabled` | REMOVE | `false` | 高 |
| `low_anchor` / lowmix | REMOVE | 不恢复 | 高 |
| `style_attn_mode=gated_raw/relu2/style_select` | REMOVE（默认） | 不作为默认 | 中 |
| legacy spatial prior / tokenizer | REMOVE | 已移除，不恢复 | 高 |

---

## 2. 逐项设计决策

### 2.1 `style_attn_mode`

**决策**：KEEP（当前 `gated`），但 `softmax` 可作为更优候选。

- **独立贡献**：在当前 `latent` + `endpoint_film_hd512` 基线上，6 种 attention 模式 WFI 均通过门。`softmax` 取得最低 WFI（0.3736），`gated` 为 0.3925，差距 0.0189。
- **交互影响**：attention 核函数的差异被 endpoint-FiLM / latent 条件源 / NSWD 噪声所缓冲；修改 attention 本身不是白化主因。
- **证据**：`attn_softmax` WFI=0.3736/CLIP-S=0.7023/LPIPS=0.3397；`attn_gated` WFI=0.3925/CLIP-S=0.7020/LPIPS=0.3400。
- **置信度**：中（attention 模式在历史上差异巨大，但在当前基线上差异缩小，多 epoch 稳定性待验证）。
- **推荐值**：保守保持 `gated`；若追求更优 WFI 可切换到 `softmax`，需在 Phase 5 做完整验证。

---

### 2.2 `style_film_enabled`

**决策**：REMOVE（默认关闭）。

- **独立贡献**：开关差异极小。`stylefilm_on` WFI=0.3785，`stylefilm_off` WFI=0.3782；CLIP-S 差 0.0001，LPIPS 差 0.0001。
- **交互影响**：在已有 `endpoint_film_enabled=true` 的情况下，block 内 StyleFiLM 对最终指标几乎无独立贡献。
- **证据**：`stylefilm_on` WFI=0.3785/CLIP-S=0.7020/LPIPS=0.3321；`stylefilm_off` WFI=0.3782/CLIP-S=0.7021/LPIPS=0.3322。
- **置信度**：高。
- **推荐值**：`false`。可简化模型并减少一条风格路径，避免与 endpoint-FiLM 的信号叠加。

---

### 2.3 `endpoint_head_mode`

**决策**：NEED_MORE_DATA。

- **独立贡献**：`velocity` head 单独使用即可通过 WFI 门（WFI=0.3769，CLIP-S=0.7020，LPIPS=0.3315），且优于大多数 `endpoint_lowhigh` 变体。
- **交互影响**：历史 `620_film_v5_gated`（velocity）WFI=0.4902 与本批次 `endpoint_velocity` WFI=0.3769 的差异主要来自 `style_condition_source`（历史为某种配置，当前为 `latent`）。这说明 velocity head 在白化抑制上依赖条件源选择。
- **证据**：`endpoint_velocity` WFI=0.3769/CLIP-S=0.7020/LPIPS=0.3315；`endpoint_lowhigh_hd512` WFI=0.3915/CLIP-S=0.7019/LPIPS=0.3432。
- **置信度**：中（velocity 在 smoke 1 epoch 表现更好，但 `endpoint_lowhigh` 经过更多历史验证，多 epoch / 全量训练稳定性未知）。
- **推荐值**：Phase 5 前保守保持 `endpoint_lowhigh`；同步测试 `velocity` 多 epoch 稳定性，若通过可作为更简洁默认。

---

### 2.4 `endpoint_film_enabled`

**决策**：KEEP。

- **独立贡献**：在 `endpoint_lowhigh` 路径上，关闭 FiLM 的 `endpoint_lowhigh_nofilm` WFI=0.3957，接近 hd512 的 0.3915；打开 FiLM 并非 WFI 的决定性因素。
- **交互影响**：FiLM 为 style→endpoint 提供直接调制路径；即使当前基线下 WFI 不敏感，移除 FiLM 可能削弱风格表达上限，尤其在更复杂数据或多 epoch 场景。
- **证据**：`endpoint_lowhigh_nofilm` WFI=0.3957/CLIP-S=0.7012/LPIPS=0.3399；`endpoint_lowhigh_hd128` WFI=0.3801/CLIP-S=0.7023/LPIPS=0.3422。
- **置信度**：中。
- **推荐值**：`true`。

---

### 2.5 `endpoint_style_hidden_dim`

**决策**：RESTORE/降低至 128。

- **独立贡献**：hd128 取得 WFI=0.3801，优于 hd256（0.3990）和 hd512（0.3915）。hd256 反而接近门限，表现最差。
- **交互影响**：历史认为 hd512 是“关键容量突破”，但该结论依赖于 DINO patches 条件源；在 `latent` 条件源下，hd128 已足够，甚至过大会导致 WFI 劣化。
- **证据**：`endpoint_lowhigh_hd128` WFI=0.3801/CLIP-S=0.7023/LPIPS=0.3422；`endpoint_lowhigh_hd256` WFI=0.3990/CLIP-S=0.7013/LPIPS=0.3408；`endpoint_lowhigh_hd512` WFI=0.3915/CLIP-S=0.7019/LPIPS=0.3432。
- **置信度**：高。
- **推荐值**：`128`。

---

### 2.6 `style_cross_attn_gate_init`

**决策**：RESTORE 至 0.05。

- **独立贡献**：gate_init=0.05 取得最低 WFI（0.3757），0.5 次之（0.3833），0.3 最差（0.3908）。
- **交互影响**：gate_init 影响 cross-attention 的初始激活强度；0.05 更接近 schema 默认值，可能是更稳定的起点。
- **证据**：`gate_init005` WFI=0.3757/CLIP-S=0.7020/LPIPS=0.3413；`gate_init05` WFI=0.3833/CLIP-S=0.7022/LPIPS=0.3415；`gate_init03` WFI=0.3908/CLIP-S=0.7022/LPIPS=0.3446。
- **置信度**：高。
- **推荐值**：`0.05`。

---

### 2.7 `base_dim`

**决策**：KEEP 64。

- **独立贡献**：128×4 的 CLIP-S 仅比 64×4 高 0.0005，WFI 反而更差；128×6 无叠加收益。
- **交互影响**：历史假设 dim=64 是 style 0.67 天花板的结论在当前基线上不成立；瓶颈已转移至条件源和 loss 权重。
- **证据**：`capacity_64x4` WFI=0.3887/CLIP-S=0.7021；`capacity_128x4` WFI=0.3921/CLIP-S=0.7026；`capacity_128x6` WFI=0.3895/CLIP-S=0.7019。
- **置信度**：高。
- **推荐值**：`64`。

---

### 2.8 `num_res_blocks`

**决策**：KEEP 4（或微调到 6 若 WFI 优先）。

- **独立贡献**：64×6 的 WFI 最优（0.3828），但 LPIPS 略升（0.3426）；参数量和训练时间增加约 14%。
- **交互影响**：深度增加对 WFI 有轻微正面作用，但对 CLIP-S 无影响。
- **证据**：`capacity_64x6` WFI=0.3828/CLIP-S=0.7021/LPIPS=0.3426；`capacity_64x4` WFI=0.3887/CLIP-S=0.7021/LPIPS=0.3382。
- **置信度**：高。
- **推荐值**：默认 `4`；若后续实验以 WFI 为首要目标，可采用 `6`。

---

### 2.9 `single_step_swd_weight`

**决策**：KEEP 8.0。

- **独立贡献**：SWD=0 时 CLIP-S 降至 0.7007；SWD=16 时 CLIP-S 升至 0.7028 但 WFI 超门（0.4013）。SWD=8 在 WFI 门内取得较好平衡。
- **交互影响**：SWD 与 edge loss 存在明显交互：SWD=16 + edge=0 时 WFI 回到 0.3885 且 CLIP-S=0.7030。
- **证据**：`loss_swd8` WFI=0.3959/CLIP-S=0.7018/LPIPS=0.3369；`loss_swd16` WFI=0.4013/CLIP-S=0.7028/LPIPS=0.3395；`loss_swd16_edge0` WFI=0.3885/CLIP-S=0.7030/LPIPS=0.3396。
- **置信度**：高。
- **推荐值**：`8.0`；在 `single_step_edge_weight=0.0` 基础上可进一步探索 12–16。

---

### 2.10 `swd_noise_sigma`

**决策**：KEEP 0.02。

- **独立贡献**：关闭 noise 在基线设置下 WFI 从 0.3959 升至 0.4105；在 edge=0 设置下从 0.3786 升至 0.4077。
- **交互影响**：noise 是 SWD 白化抑制的基础项，与 edge weight、SWD weight 均有交互。
- **证据**：`loss_nosigma` WFI=0.4105；`loss_edge0_nosigma` WFI=0.4077；对应含 noise 版本为 0.3959 和 0.3786。
- **置信度**：高。
- **推荐值**：`0.02`。

---

### 2.11 `single_step_edge_weight`

**决策**：REMOVE（设为 0.0）。

- **独立贡献**：`loss_edge0` 是全部 9 个 loss 变体中 WFI 最低（0.3786），且 CLIP-S（0.7020）和 LPIPS（0.3336）均优于基线。
- **交互影响**：edge=0 与高 SWD 组合（SWD=16）仍能通过 WFI 门；edge=0.1 会抬高 WFI 并轻微损害风格/内容。
- **证据**：`loss_edge0` WFI=0.3786/CLIP-S=0.7020/LPIPS=0.3336；基线 `loss_swd8` WFI=0.3959/CLIP-S=0.7018/LPIPS=0.3369。
- **置信度**：高。
- **推荐值**：`0.0`。

---

### 2.12 `style_condition_source`

**决策**：KEEP `latent`。

- **独立贡献**：`latent` 变体 WFI=0.3842，CLIP-S=0.7020，LPIPS=0.3417，全部满足放行门；`target_dino_patches` WFI=0.6407，严重白化。
- **交互影响**：DINO patches 在当前 `endpoint_film_hd512` 基线上导致风格/端点信号过强，学到“高亮度、低饱和度、低对比度”的均值解。
- **证据**：`intrinsic_latent` WFI=0.3842/CLIP-S=0.7020/LPIPS=0.3417；`dino_baseline` WFI=0.6407/CLIP-S=0.7097/LPIPS=0.2773。
- **置信度**：高。
- **推荐值**：`latent`。

---

### 2.13 DINO adapter / intrinsic cross-attention

**决策**：DINO adapter REMOVE；intrinsic cross-attention KEEP。

- **DINO adapter 独立贡献**：启用 adapter 后 WFI 从 0.6407 微降至 0.6076，仍严重超标；仅轻微提升 content LPIPS（0.2773→0.2618）。
- **intrinsic cross-attention 独立贡献**：`intrinsic_latent` 通过 WFI 门，且优于历史 H6 intrinsic（CLIP-S 0.7020 vs 0.6717，LPIPS 0.3417 vs 0.3678）。
- **交互影响**：adapter 无法修复 DINO patches 的白化；intrinsic latent 在 endpoint-FiLM 辅助下已足够强。
- **证据**：`dino_adapter` WFI=0.6076/CLIP-S=0.7063/LPIPS=0.2618；`intrinsic_latent` WFI=0.3842/CLIP-S=0.7020/LPIPS=0.3417。
- **置信度**：高（对 adapter 默认关闭）；中（对 intrinsic latent 的长期有效性，需多 epoch 验证）。
- **推荐值**：`style_dino_adapter_enabled=false`，`style_condition_source=latent`。

---

### 2.14 3-epoch training（历史 over-train 现象）

**决策**：KEEP 默认 1 epoch；多 epoch 需带 early stopping / 低学习率。

- **独立贡献**：历史 E3 显示 `endpoint_film_hd512` 3 epoch WFI 从 e1 的 0.4271 升至 e3 的 0.4680，CLIP-S 提升但 LPIPS 大幅恶化。
- **交互影响**：当前 lr=2e-4 下，更多 epoch 使优化落入“安全但白化”的 basin。
- **证据**：历史 `endpoint_film_hd512` e1 WFI=0.3906，e3 WFI=0.4680。
- **置信度**：高。
- **推荐值**：默认 `training.num_epochs=1`；若需长训练，尝试 `lr=1e-4` 或 early stopping（WFI 或 LPIPS 作为停止指标）。

---

### 2.15 其他历史设计

#### 2.15.1 Style MoE

**决策**：REMOVE（默认关闭）。

- **理由**：Round 1 显示 MoE 在 dim=64 下无收益；当前消融显示 dim=128 也无显著收益，说明瓶颈不在 Q 侧容量。
- **推荐值**：`style_moe_enabled=false`。
- **置信度**：中。

#### 2.15.2 `velocity_hf_residual_enabled`

**决策**：REMOVE。

- **理由**：历史 `620_film_v5_hf_residual_local_smoke` WFI=0.4746，网络主动弱化该路径，独立无效。
- **推荐值**：`false`。
- **置信度**：高。

#### 2.15.3 `lowmix` / `low_anchor`

**决策**：REMOVE，不恢复。

- **理由**：历史 `lowmix05` 导致 LPIPS 崩坏（0.3492）并引发水平泄漏；style 微升但不可接受。
- **推荐值**：不恢复相关参数。
- **置信度**：高。

#### 2.15.4 `style_attn_mode=gated_raw/relu2/style_select`

**决策**：REMOVE 作为默认。

- **理由**：虽然在当前基线上均通过 WFI 门，但历史上曾导致严重白化；无证据表明它们优于 softmax/gated。
- **推荐值**：不作为默认，仅在特定研究中按需开启。
- **置信度**：中。

#### 2.15.5 Legacy spatial prior / tokenizer

**决策**：REMOVE，不恢复。

- **理由**：历史已证 zero ROI，相关代码已从 main 移除。
- **推荐值**：维持移除。
- **置信度**：高。

---

## 3. 推荐默认配置（Phase 5 验证基线）

基于以上决策，建议 Phase 5 验证以下最小有效配置：

```json
{
  "model": {
    "base_dim": 64,
    "num_res_blocks": 4,
    "style_attn_mode": "gated",
    "style_film_enabled": false,
    "endpoint_head_mode": "endpoint_lowhigh",
    "endpoint_film_enabled": true,
    "endpoint_style_hidden_dim": 128,
    "endpoint_film_init_std": 0.0,
    "style_cross_attn_gate_init": 0.05,
    "style_condition_source": "latent",
    "style_dino_adapter_enabled": false,
    "style_moe_enabled": false,
    "velocity_hf_residual_enabled": false
  },
  "bridge": {
    "single_step_swd_weight": 8.0,
    "swd_noise_sigma": 0.02,
    "single_step_edge_weight": 0.0
  },
  "training": {
    "num_epochs": 1,
    "learning_rate": 2e-4
  }
}
```

> **保守路径**：保持 `endpoint_head_mode=endpoint_lowhigh`，等待 velocity 多 epoch 验证。  
> **激进路径**：若 velocity 验证通过，可进一步简化为 `endpoint_head_mode=velocity`、`endpoint_film_enabled=true/false` 待测。

---

## 4. 未决问题与后续实验

| 问题 | 优先级 | 建议实验 |
|---|---|---|
| velocity head 多 epoch 稳定性 | 高 | `620_film_v5_velocity_3ep_smoke` |
| softmax attention 多 epoch 稳定性 | 中 | `620_ablation_attn_softmax_3ep_smoke` |
| SWD=12–16 + edge=0 组合 | 中 | `620_ablation_loss_swd12_edge0_smoke`、`loss_swd16_edge0` 已显示潜力 |
| 低学习率 + 3 epoch | 中 | `lr=1e-4` 或 `1.5e-4`，early stopping on WFI |
| intrinsic latent 多 epoch 稳定性 | 中 | `620_ablation_intrinsic_latent_3ep_smoke` |
| dim=128 + adapter/MoE 在 latent+edge=0 基线上 | 低 | 当前证据不支持优先投入 |
| multi-scale DINO / cross-attn Q source | 低 | 白化门通过后再引入 |

---

## 5. 原始数据索引

| 文件 | 内容 |
|---|---|
| `docs/620/fog/ablation_audit/results_summary.md` | 统一结果汇总 |
| `docs/620/fog/ablation_audit/history_vs_ablation.md` | Git 历史对照 |
| `docs/620/fog/ablation_audit/phase2_results.md` | Phase 2 核心维度 |
| `docs/620/fog/ablation_audit/phase3_capacity_results.md` | Phase 3.1 容量 |
| `docs/620/fog/ablation_audit/phase3_loss_results.md` | Phase 3.2 loss |
| `docs/620/fog/ablation_audit/phase3_dino_results.md` | Phase 3.3 DINO |
