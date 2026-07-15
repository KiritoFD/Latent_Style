# Round E2 最小修复实验报告

**日期**: 2026-06-21  
**执行环境**: 本地 RTX 4070，Windows，PyTorch CUDA  
**基线**: `620_film_v5_gated_local_smoke`（E1 最优）  
**实验目录**: `exp/620_spatial_bridge/`

---

## 1. 实验设计

### 1.1 理论背景（来自 Round E1）

Round E1 内部探针显示，当前最优基线 `620_film_v5_gated_local_smoke` 的白化主因是 **高频选择性塌缩**：

- 整体 endpoint shrinkage 因子：$c = \text{endpoint_alpha}/(1-t) \approx 0.62$
- 高频 endpoint shrinkage 因子：$c_\text{high} \approx 0.08$

即 style 信号已能驱动 latent 整体朝目标移动约 62%，但高频分量（纹理、边缘、饱和度、对比度）仅移动约 8%。

### 1.2 候选修复

| 编号 | 名称 | 理论动机 | 最小改动 |
|---|---|---|---|
| P0 | **Endpoint-FiLM Head** | 把 style 信号直接送到 endpoint head，绕过 block 级 shrinkage，让 endpoint 的低频/高频分支都被 style 调制 | `endpoint_head_mode=endpoint_lowhigh` + `endpoint_film_enabled=true` |
| P1 | **High-Frequency Residual** | 在 velocity head 输出层显式保留 source latent 的高频成分，防止高频被洗掉 | `output = predicted_velocity + w \cdot (x - \text{avg_pool}(x))`，$w$ 可学习、初始 0.1 |

### 1.3 成功门槛与失败信号

- **成功门槛**: `wfi_score < 0.40` 且 `clip_style >= 0.695` 且 `content_lpips < 0.36`
- **关键探针信号**: $c_\text{high}$ 从基线 ~0.08 提升到 ≥0.3
- **失败信号**: WFI 未下降、clip_style 未提升、$c_\text{high}$ 不变

---

## 2. 训练配置关键参数

两个实验均基于 `620_film_v5_gated_local_smoke`（intrinsic_v2 + gated attention + gate_init=0.3 + style_film），保持 smoke 规模：

```json
{
  "training": {
    "batch_size": 4,
    "accumulation_steps": 16,
    "num_epochs": 1,
    "num_workers": 0,
    "learning_rate": 0.0002,
    "use_amp": true,
    "amp_dtype": "bf16"
  },
  "model": {
    "style_attn_mode": "gated",
    "style_cross_attn_gate_init": 0.3,
    "style_film_enabled": true,
    "style_attn_temperature": 1.0
  },
  "bridge": {
    "swd_noise_sigma": 0.02,
    "single_step_swd_weight": 8.0,
    "single_step_edge_weight": 0.1
  }
}
```

### 2.1 Experiment 1: Endpoint-FiLM Head

- **配置路径**: `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/config.json`
- **关键增量参数**:
  - `endpoint_head_mode`: `"endpoint_lowhigh"`
  - `endpoint_film_enabled`: `true`
  - `endpoint_lowpass_kernel`: `5`
  - `endpoint_high_scale`: `1.0`
  - `endpoint_velocity_floor`: `0.05`
  - `endpoint_style_hidden_dim`: `128`

### 2.2 Experiment 2: High-Frequency Residual

- **配置路径**: `exp/620_spatial_bridge/620_film_v5_hf_residual_local_smoke/config.json`
- **关键增量参数**:
  - `velocity_hf_residual_enabled`: `true`
  - `velocity_hf_residual_init`: `0.1`
  - `velocity_hf_residual_kernel`: `5`
- **代码改动**: `src/model620.py` velocity head 分支加入可学习高频残差

---

## 3. 实验结果

### 3.1 WFI / CLIP-S / LPIPS 指标

| 实验 | clip_style ↑ | content_lpips ↓ | wfi_score ↓ | ΔWFI (gen - source) | clip_s_delta_idt |
|---|---:|---:|---:|---:|---:|
| **Baseline (E1 最优)** | 0.6987 | 0.3300 | **0.4902** | **+0.1685** | - |
| **P0 Endpoint-FiLM Head** | **0.7066** | **0.3226** | **0.4283** | **+0.1066** | 0.0667 |
| **P1 HF Residual** | 0.7020 | 0.3263 | 0.4746 | +0.1529 | 0.0621 |

> 注：Baseline 指标来自 `docs/620/fog/baseline_audit/local_audit_2026-06-21.md` / E1 probe 报告；P0/P1 指标由本次 `tools/run_eval_with_wfi.py` 测得。

### 3.2 内部探针指标（t=0.0）

| 实验 | $c$ | $c_\text{high}$ | style_gate | cross_attn_entropy | block std 演化 |
|---|---:|---:|---:|---:|---|
| **Baseline (E1)** | 0.617 | 0.078 | 0.294 | 5.53 | 0.65→0.77→0.94→1.13 |
| **P0 Endpoint-FiLM** | 0.290 | 0.053 | 0.292 | 5.54 | 0.62→0.74→0.85→1.05 |
| **P1 HF Residual** | 0.596 | 0.081 | 0.291 | 5.53 | 待补 |

> $c = \text{endpoint_alpha}/(1-t)$，$c_\text{high} = \text{endpoint_high_alpha}/(1-t)$。

### 3.3 关键观察

1. **Endpoint-FiLM Head 明显改善 WFI 与 clip_style**：
   - WFI 从 0.4902 降至 0.4283（↓12.6%）
   - ΔWFI 从 +0.1685 降至 +0.1066
   - clip_style 从 0.6987 升至 0.7066
   - content_lpips 从 0.3300 降至 0.3226
   - **所有三个核心指标同时改善**，说明 style-FiLM 确实让 endpoint 输出更风格化、同时保留内容。

2. **但 Endpoint-FiLM 的 RMS shrinkage 反而下降**：
   - 整体 $c$ 从 0.62 降至 0.29
   - 高频 $c_\text{high}$ 从 0.08 降至 0.05
   - 这表明 **RMS 位移不是 WFI 的唯一决定因素**。Endpoint-low/high 分解 + FiLM 让模型用更小的 RMS 位移实现了更好的风格化质量（更准确的位移方向/谱分布）。

3. **HF Residual 几乎无效果**：
   - WFI 0.4746，略好于基线 0.4902，但远差于 Endpoint-FiLM。
   - $c$ 和 $c_\text{high}$ 与基线几乎相同（0.596 / 0.081 vs 0.617 / 0.078）。
   - 学到的残差权重从 0.1 降至 0.089，说明网络倾向于弱化该残差。
   - 简单的 source 高频保留并不能解决风格高频迁移不足的问题。

---

## 4. 与 Round M/E1 理论的对照

| 理论/预测 | Round E1 状态 | E2 结果 | 判定 |
|---|---|---|---|
| P-E3: endpoint head + style-FiLM 恢复 style sensitivity | 待补证/高度优先 | WFI↓、clip_style↑，但 $c_\text{high}$ 未达 ≥0.3 | **部分支持** |
| 高频塌缩是白化主因 | 支持 | Endpoint-FiLM 改善 WFI，但 RMS 高频位移未显著增加 | **部分支持/需细化** |
| 增大 endpoint 高频位移即可降 WFI | 隐含假设 | Endpoint-FiLM 降 WFI 但 $c_\text{high}$↓，HF Residual $c_\text{high}$≈不变但 WFI 微降 | **否证简单版** |
| Attention 平均化是 style 弱化的起点 | 支持 | 三个实验 entropy 均 ~5.53，无改善 | **支持** |
| HF Residual 保留 source 高频可防白化 | 新假设 | 效果微弱 | **不支持作为独立修复** |

**核心修正**：

> WFI 的下降不一定需要 RMS 高频位移的增大。Endpoint-FiLM 通过 **更准确的低频/高频位移方向**（由 style 直接调制），在整体 RMS 位移减小的情况下同时提升了风格迁移质量和图像动态范围。
>
> 因此，"高频选择性塌缩"的叙事需要从 **"高频 RMS 位移不足"** 细化为 **"style 信号未有效调制高频/细节方向"**。

---

## 5. 结论与下一步建议

### 5.1 是否通过 Round E2 放行门槛

| 门槛 | P0 Endpoint-FiLM | P1 HF Residual |
|---|---|---|
| WFI < 0.40 | ❌ 0.4283 | ❌ 0.4746 |
| clip_style ≥ 0.695 | ✅ 0.7066 | ✅ 0.7020 |
| content_lpips < 0.36 | ✅ 0.3226 | ✅ 0.3263 |

**结论**：P0 Endpoint-FiLM Head **显著改善但未完全过门**（WFI 仍 > 0.40）。P1 HF Residual 效果有限。

### 5.2 是否进入 Round E3 正式验收

不建议直接进入 E3 正式验收，因为 WFI 0.4283 仍未达到 <0.40 的硬门槛。但 P0 是 **当前最有希望的修复方向**，建议在 E2 内再做一轮最小改进后复测。

### 5.3 下一轮假设（Round E2.5 / E3 前）

基于 P0 的部分成功，提出以下可验证假设：

1. **增大 endpoint high-scale 或降低 velocity floor**：P0 的 $c_\text{high}$ 仅 0.053，说明高频分支仍被抑制。尝试 `endpoint_high_scale=2.0` 或 `endpoint_velocity_floor=0.01`。
2. **Endpoint-FiLM + 更大的 high-pass 带宽**：当前 lowpass kernel=5，高频分支覆盖的频段可能过窄；尝试 kernel=3 或 7。
3. **去掉 endpoint head 内的 GroupNorm**：FiLMEndpointHead 仍含 GN，可能压缩动态范围；尝试无 GN 的 FiLM head。
4. **结合 P0 与 P1**：在 endpoint_lowhigh 模式下，对 high_delta 加入 source 高频残差，可能同时获得 style 调制与 source 细节保留。

**禁止并行堆叠**：下一轮每次只改动一个参数，避免无法归因。

---

## 6. 产物清单

| 产物 | 路径 |
|---|---|
| P0 配置 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/config.json` |
| P0 checkpoint | `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/epoch_0001.pt` |
| P0 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/full_eval_wfi/epoch_0001/` |
| P0 内部探针 | `docs/620/fog/round_e2/internal_probe_endpoint_film_epoch_0001.json` |
| P1 配置 | `exp/620_spatial_bridge/620_film_v5_hf_residual_local_smoke/config.json` |
| P1 checkpoint | `exp/620_spatial_bridge/620_film_v5_hf_residual_local_smoke/epoch_0001.pt` |
| P1 WFI 评估 | `exp/620_spatial_bridge/620_film_v5_hf_residual_local_smoke/full_eval_wfi/epoch_0001/` |
| P1 内部探针 | `docs/620/fog/round_e2/internal_probe_hf_residual_epoch_0001.json` |
| 配置生成脚本 | `tools/create_round_e2_configs.py` |
| 代码改动 | `src/model620.py`（velocity HF residual）, `src/config_schema.py`, `src/utils/run_evaluation.py`（NameError 修复） |
