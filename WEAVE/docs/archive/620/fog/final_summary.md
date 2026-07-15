# 620 白化问题诊断最终总结

**日期**: 2026-06-21  
**状态**: ✅ Round E3 白化压制验收通过  
**文档版本**: Round D 闭环

---

## 1. 问题定义

620 风格迁移模型在生成迁移图像时呈现系统性白化/雾化（whitening / fog）：输出图像对比度低、饱和度低、亮度偏高，视觉上如同覆盖了一层薄雾。该问题同时影响 `identity` 重建与 `style_transfer` 生成，说明白化并非风格迁移本身的副产物，而是模型输出的系统性统计缺陷。

项目以 Seedream repaired750 的图像空间指标作为健康参考（`wfi_score ≈ 0.158`），并设定本轮放行门为 `wfi_score < 0.40`。通过从图像空间 WFI 到潜空间 endpoint probe 的多轮诊断，最终确认白化起源于 `predict_endpoint(t=0)`：模型在 source 端预测的端点仅向目标方向移动了约 16%，高频分量甚至出现负方向，导致生成图像的动态范围与饱和度被压缩。

---

## 2. 执行摘要（Round A → Round E3）

### Round A: 基线审计与问题重述
- 梳理 `docs/620/`、`configs/620*.json`、`src/*620*` 与最近实验产物。
- 明确当前最优基线为 `620_film_v5_gated_local_smoke`，`wfi_score = 0.4902`。
- 建立统一问题定义：620 白化是 endpoint shrinkage 导致的图像空间统计塌缩，而非 solver 或 VAE decode 问题。

### Round M: 整体动力学理论工作
- 建立整体状态变量与训练/推理路径（`theory/overall_dynamics.md`）。
- 推导平凡解进入条件（`theory/trivial_solution.md`）。
- 分析训练—推理失配（`theory/train_infer_mismatch.md`）。
- 分析 GroupNorm/LayerNorm/AdaLN 导致的统计塌缩（`theory/stat_collapse.md`）。
- 绘制干预方案地图（`theory/intervention_map.md`）。

### Round P: 指标与探针体系
- 定义图像空间 WFI 指标（contrast_ratio、dynamic_range、saturation、wfi_score 等）。
- 定义潜空间指标（endpoint alpha、high-frequency alpha、style_sensitivity 等）。
- 建立本地与远程 probe runbook（`probe_system/`）。

### Round E1: 当前模型整体诊断
- 跑白化基线评测，确认 `620_film_v5_gated_local_smoke` WFI = 0.4902，`clip_style = 0.6987`。
- 梯度探针发现 SWD 梯度非零但基本正交于 `v_target`（`cos ≈ -0.024`），排除 SWD 梯度平坦导致平凡解的假设。
- 理论修正为：style 信号太弱 + endpoint head 容量不足，导致模型学到条件期望 `E[v_target | x, t]` 而非 style-specific velocity。

### Round E2: 最小必要修复实验
- 在本地 RTX 4070 上对比两个候选：Endpoint-FiLM Head（P0）与 High-Frequency Residual（P1）。
- **P0 Endpoint-FiLM Head** 将 WFI 从 0.4902 降至 0.4283，`clip_style` 从 0.6987 升至 0.7066，content_lpips 从 0.3300 降至 0.3226。
- **P1 HF Residual** 几乎无效（WFI = 0.4746）。
- 理论修正：白化改善不依赖 RMS 高频位移增大，而依赖 style 对 endpoint 方向的更准确调制。

### Round E3: 白化压制验收
- 对 Endpoint-FiLM Head 做 3 epoch 验证：WFI 单调上升（0.4271 → 0.4532 → 0.4680），发现过训练加剧白化。
- 测试两个最小改动：
  - **H1** `endpoint_film_init_std=0.02`：WFI = 0.4022。
  - **H2** `endpoint_style_hidden_dim=512`：WFI = 0.3906，正式低于 0.40 放行门。
- 最终最优：`620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt`。

---

## 3. 最终最优方案

### 3.1 配置参数

| 参数 | 最终值 |
|---|---|
| `model.endpoint_head_mode` | `endpoint_lowhigh` |
| `model.endpoint_film_enabled` | `true` |
| `model.endpoint_style_hidden_dim` | `512` |
| `model.endpoint_film_init_std` | `0.0`（最终最优未启用 H1 组合） |
| `model.style_attn_mode` | `gated` |
| `model.style_cross_attn_gate_init` | `0.3` |
| `model.style_film_enabled` | `true` |
| `bridge.training_target_projection_low_mode` | `target_linear` |
| `bridge.swd_noise_sigma` | `0.02` |
| `bridge.single_step_swd_weight` | `8.0` |
| `bridge.single_step_edge_weight` | `0.1` |

### 3.2 训练设置

| 设置 | 值 |
|---|---|
| 环境 | 本地 RTX 4070，Windows，PyTorch CUDA |
| batch_size | 4 |
| accumulation_steps | 16 |
| num_epochs | 1（最优） |
| learning_rate | 2e-4 |
| amp_dtype | bf16 |

### 3.3 评估指标

| 指标 | 最终最优值 | 放行门 | 状态 |
|---|---:|---|---|
| `wfi_score` | **0.3906** | < 0.40 | ✅ 通过 |
| `clip_style` | **0.7015** | ≥ 0.695 | ✅ 通过 |
| `content_lpips` | **0.3382** | < 0.36 | ✅ 通过 |
| `ΔWFI (gen - source)` | **+0.0689** | — | — |
| `clip_s_delta_idt` | — | — | — |

### 3.4 Checkpoint 路径

```
exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt
```

对应完整 WFI 评估目录：

```
exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/
```

---

## 4. 并排对比表

| 方案 | checkpoint | WFI ↓ | CLIP-S ↑ | LPIPS ↓ | ΔWFI |
|---|---|---:|---:|---:|---:|
| 基线 gated | `620_film_v5_gated_local_smoke` | **0.4902** | 0.6987 | 0.3300 | +0.1685 |
| gated_raw | `620_film_v5_gated_raw_local_smoke` | 0.6435 | 0.6987 | **0.2973** | +0.3218 |
| relu2 | `620_film_v5_relu2_local_smoke` | 0.5340 | 0.6964 | 0.3102 | +0.2123 |
| style_select | `620_film_v5_style_select_local_smoke` | 0.5005 | 0.6982 | 0.3331 | +0.1788 |
| endpoint_film (hd128) | `620_film_v5_endpoint_film_local_smoke` | 0.4283 | 0.7066 | 0.3226 | +0.1066 |
| **endpoint_film_hd512（最终最优）** | **`620_film_v5_endpoint_film_hd512_local_smoke`** | **0.3906** | **0.7015** | 0.3382 | **+0.0689** |

> 注：所有 attention 变体（gated_raw / relu2 / style_select）均未降低 WFI，反而多数恶化；Endpoint-FiLM hd128 显著改善但未过门；hd512 容量提升后直接过门。

---

## 5. 核心代码改动清单

| 文件 | 改动 | 说明 |
|---|---|---|
| `src/config_schema.py` | 新增 `endpoint_film_init_std: float = 0.0` | 支持配置 FiLM 投影层最后一层的初始化标准差 |
| `src/config_schema.py` | 已存在 `endpoint_style_hidden_dim: int = 128` | 默认值 128，最终最优通过配置覆盖为 512 |
| `src/model620.py` | `FiLMEndpointHead.__init__` 新增 `film_init_std: float = 0.0` 参数 | 支持可配置初始化 |
| `src/model620.py` | `FiLMEndpointHead` 中根据 `film_init_std > 0.0` 决定 FiLM 投影层 weight 初始化 | `>0` 时使用 `nn.init.normal_(..., std=film_init_std)`，否则 zero-init |
| `src/model620.py` | `SpatialBridge620.__init__` 读取 `endpoint_style_hidden_dim` 与 `endpoint_film_init_std` 并传递给 `FiLMEndpointHead` | 使配置生效 |

关键代码片段（`src/model620.py`）：

```python
class FiLMEndpointHead(nn.Module):
    def __init__(self, dim, latent_channels, style_dim, style_hidden_dim, film_init_std=0.0):
        ...
        if film_init_std > 0.0:
            nn.init.normal_(self.film_proj[-1].weight, mean=0.0, std=film_init_std)
        else:
            nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)
```

---

## 6. 理论结论

### 6.1 白化主因：style→endpoint 的 FiLM 映射容量不足

经过多轮 probe 与实验，白化的根因被精确为：

> **style→endpoint 的 FiLM 映射容量不足，导致 modulation 信号被压缩到接近零的无效区域，endpoint 输出退化为与风格无关的“安全”均值。**

证据链：
1. `endpoint_style_hidden_dim=128` 时 WFI = 0.4283，仍高于放行门。
2. 将 hidden_dim 提升到 512 后，WFI 降至 0.3906，直接过门。
3. 3 epoch 训练使 WFI 单调恶化，说明优化 landscape 存在“安全但白化”的 basin，容量不足时模型更容易滑入该 basin。

### 6.2 为什么 attention 改造无效

`gated_raw`、`relu2`、`style_select` 三种 attention 变体均未降低 WFI，其中 `gated_raw` 使 WFI 升至 0.6435。原因：

1. **cross-attention 的平均化瓶颈**：attention weights 由 content-dependent 的 `Q(x)` 决定，不同 style 的 style tokens 产生相似的 softmax 分布，导致 `V(S)` 加权和在不同 style 间几乎不变。
2. **attention 模式改动只改变 token 聚合方式**，未能把 style 信号直接、强有力地送入 endpoint head；style 信号在 block 内被进一步 norm/平均化后，仍无法转化为 target-facing 的 endpoint 位移。
3. 实验上，四种 attention 变体的 `clip_style` 几乎相同（0.696–0.699），说明 attention 内部改动对高层风格对齐影响极小，无法打破 endpoint shrinkage。

### 6.3 为什么 endpoint_film_hd512 有效

Endpoint-FiLM Head 绕过 cross-attention，直接将 style 全局嵌入通过 MLP 映射为通道级 gamma/beta，调制 endpoint head 的 feature map。当 `endpoint_style_hidden_dim=128` 时，该 MLP 容量不足，无法学习复杂的 per-style 调制，gamma/beta 信号被压缩；提升到 512 后：

1. **表达能力提升**：style embedding 到 FiLM 参数的映射可以区分更细粒度的风格特征。
2. **调制信号不再被压缩**：gamma/beta 能真正驱动 endpoint 的低频与高频分支朝目标风格移动。
3. **WFI 下降不依赖 RMS 位移增大**：H2 的 `c_high` 未显著增加，但生成图像的对比度、饱和度与动态范围更接近健康分布，说明方向/结构调制比幅度更重要。

---

## 7. 未决问题与下一步建议

### 7.1 是否继续压低 WFI 到 Seedream IDT 水平（0.158）

当前最优 WFI = 0.3906，距离 Seedream IDT ≈ 0.158 仍有 +0.233 的差距。虽然已过本轮 0.40 放行门，但若要接近 Seedream 水平，需进一步验证：
- 组合 H1 + H2：`endpoint_film_init_std=0.02` + `endpoint_style_hidden_dim=512`。
- 移除 `FiLMEndpointHead` 内的 `GroupNorm`，避免动态范围被压缩。
- 调整 `endpoint_lowpass_kernel`，优化 low/high 分解带宽。

### 7.2 Round E4 候选方向

白化放行门通过后，可谨慎恢复 620 原计划实验，但所有实验必须先过白化指标检查：
1. **text 条件引入**：复用 WFI 指标，验证 text 条件不会重新引入雾化。
2. **cross-attn 改造**：验证不会重新引入 attention 平均化导致的白化。
3. **DINO 去留评估**：先做无 DINO 对照，收益不显著则砍掉；仅在显著收益时保留或扩展多尺度 DINO。
4. **skip 比例、Per-Region SWD、注意力稀疏化、OT 配对等**：按优先级恢复，均需先过白化检查。

### 7.3 是否需要调低学习率 / early stopping

3 epoch 训练结果显示：在 lr=2e-4 下，WFI 随 epoch 单调上升（0.4271 → 0.4532 → 0.4680），`content_lpips` 也单调上升（0.3236 → 0.3768）。这表明当前目标函数与优化器会在后期滑向“内容保留更好、但动态范围更差”的白化 basin。建议：
- 对 hd512 配置尝试 lr=1e-4 或更短的 early stopping。
- 监控每 epoch 的 WFI，选择 WFI 最低且 `clip_style` 不下降的 checkpoint。
- 考虑加入 endpoint 动态范围正则，防止后期统计塌缩。

---

## 8. 产物清单

| 产物 | 路径 |
|---|---|
| 最终最优 checkpoint | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt` |
| 最终最优完整 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/` |
| E2 实验报告 | `docs/620/fog/round_e2/experiment_report_2026-06-21.md` |
| E3 验收报告 | `docs/620/fog/round_e3/acceptance_report_2026-06-21.md` |
| E2 理论更新 | `docs/620/fog/gradient_probe/theory_update_round_e2.md` |
| E3 理论更新 | `docs/620/fog/gradient_probe/theory_update_round_e3.md` |
| 基线审计 | `docs/620/fog/baseline_audit/local_audit_2026-06-21.md` |
| 决策台账 | `docs/620/fog/decision_log.md` |
| 本最终总结 | `docs/620/fog/final_summary.md` |
