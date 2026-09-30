# Round E1.2 内部 Probe 报告：`620_film_v5_gated_local_smoke`

**日期**: 2026-06-21  
**Checkpoint**: `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/epoch_0001.pt`  
**Config**: `exp/620_spatial_bridge/620_film_v5_gated_local_smoke/config.json`  
**探针脚本**: `tools/run_internal_probe.py`  
**样本**: 5 styles × 1 sample/style = 5 source-target 对  
**时间点**: `t = 0.0, 0.5, 0.875`  
**输出文件**: `docs/620/fog/gradient_probe/internal_probe_epoch_0001.json`

---

## 1. Probe 设置与限制

### 1.1 运行方式

探针脚本直接复用 `LGTInference` 的模型加载路径，在 `model.eval()` 下对固定样本做单次前向传播，记录：

- 模型级统计：`style_gate_value`、`cross_attn_entropy`、`gate_mean`、`film_gamma_abs` 等
- 每 block 统计：`block{i}_output_std`、`block{i}_film_gamma_abs`、`block{i}_cross_attn_entropy` 等
- Endpoint 统计：`endpoint_alpha`、`endpoint_high_alpha`、`velocity_abs`、`endpoint_pred_abs` 等
- 输入/目标 latent 统计：`latent_input_std`、`target_stats` 等

### 1.2 数据限制

- 由于 `num_samples_per_style=1`，聚合结果中大量 `std=0`（不同样本被 batch 平均或模型统计为标量）。
- 本次 probe 主要反映**模型级常数**和**单一样本下的 batch 平均行为**，不用于估计跨样本方差。
- 时间点 `t` 的输入均为 source latent（脚本未显式构造 `x_t`），因此 `endpoint_alpha` 自然随 `(1-t)` 缩放。下文统一将原始 `endpoint_alpha` 除以 `(1-t)` 得到**速度 shrinkage 因子** `c`。

---

## 2. 关键聚合结果

### 2.1 模型级常数（不随 t 变化）

| 指标 | 值 | 解读 |
|---|---|---|
| `style_gate_value` | 0.294 | gate 成功打开（init=0.3），style 信号进入 trunk |
| `cross_attn_entropy` | 5.528 | 接近 `ln(256)=5.545`，attention 仍接近均匀分布 |
| `actual_attn_entropy` | 5.537 | 实际注意力分布与理论均匀上限几乎一致 |
| `gate_mean` | 0.484 | time-AdaLN gate 居中，未饱和到 0 或 1 |
| `gate_std` | 0.078 | gate 在空间/通道上有一定变化 |
| `film_gamma_abs` | 0.150 | block 内 FiLM gamma 幅度 moderate |
| `film_beta_abs` | 0.135 | block 内 FiLM beta 幅度 moderate |
| `pre_film_gamma_abs` | 0.200 | pre-Film gamma 略大于 post，说明 FiLM 后信号被轻微压缩 |
| `style_dino_active` | 0.0 | DINO adapter 未启用 |
| `endpoint_head_mode_lowhigh` | 0.0 | 当前为 `velocity` head，非 `endpoint_lowhigh` |
| `endpoint_film_enabled` | 0.0 | endpoint head 内无 FiLM 调制 |

### 2.2 各时间点 Endpoint 行为

| 指标 | t=0.0 | t=0.5 | t=0.875 | 趋势 |
|---|---:|---:|---:|---|
| `endpoint_alpha` | 0.617 | 0.309 | 0.077 | 随 (1-t) 线性下降 |
| `endpoint_alpha / (1-t)` | **0.617** | **0.618** | **0.616** | **几乎恒定** |
| `endpoint_high_alpha` | 0.078 | 0.039 | 0.010 | 随 (1-t) 线性下降 |
| `endpoint_high_alpha / (1-t)` | **0.078** | **0.078** | **0.080** | **几乎恒定** |
| `velocity_abs` | 0.618 | 0.619 | 0.620 | 基本不变 |
| `endpoint_pred_abs` | 0.698 | 0.636 | 0.692 | 稳定 |
| `endpoint_output_std` | 0.849 | 0.781 | 0.843 | 稳定 |
| `endpoint_output_mean` | -0.106 | -0.053 | -0.013 | 随 t 接近 target 而漂移 |

**核心发现**：

1. **整体端点 shrinkage 已大幅改善**：`c = endpoint_alpha/(1-t) ≈ 0.62`，远高于旧 baseline 的 0.16。这说明 `gated` + `gate_init=0.3` + block 内 StyleFiLM 已经让模型在**整体 latent 方向**上朝目标移动了约 62%。
2. **高频 shrinkage 仍是主要瓶颈**：`c_high = endpoint_high_alpha/(1-t) ≈ 0.078`，即高频分量仅朝目标方向移动约 8%。这是当前白化/雾化的直接原因。
3. **Velocity 幅度稳定但方向不对**：`velocity_abs ≈ 0.62` 且跨 t 稳定，说明网络输出能量不弱；问题是能量主要分布在低频/均值方向，高频细节方向缺失。

### 2.3 Block 级统计演化

在 `t=0.0` 时，从 block0 到 block3 的输出标准差：

| Block | output_std | output_channel_std | per_sample_dynamic_range | film_gamma_abs | cross_attn_entropy |
|---|---:|---:|---:|---:|---:|
| block0 | 0.649 | 0.411 | 8.60 | 0.150 | 5.536 |
| block1 | 0.774 | 0.434 | 10.26 | 0.151 | 5.524 |
| block2 | 0.941 | 0.464 | 12.39 | 0.139 | 5.530 |
| block3 | 1.129 | 0.482 | 14.17 | 0.160 | 5.521 |

**解读**：

- Block 输出方差逐层放大（0.65 → 1.13），说明 trunk **没有发生层内统计塌缩**；相反，网络在放大信号。
- 动态范围也逐层增大，与方差放大一致。
- 各 block 的 `cross_attn_entropy` 都接近 5.53（均匀上限），说明 attention 在每个 block 都接近平均化。
- `film_gamma_abs` 在 0.14–0.16 之间，跨 block 稳定，说明 FiLM 调制存在但幅度不足以产生方向性高频位移。

---

## 3. 白化首次出现的位置与主要塌缩类型

### 3.1 首次出现位置

- **Trunk 不是塌缩起点**：block 输出方差逐层增长，无早期压缩。
- **Cross-attention 是条件期望坍缩点**：`cross_attn_entropy ≈ 5.53` 表明 attention 权重几乎均匀，style tokens 被平均化为边缘期望，导致 style-specific 信号在 block 输入端就已被削弱。
- **Endpoint head 是高频塌缩点**：模型能把低频/整体结构方向推进 62%，但高频方向仅推进 8%。由于 `endpoint_head_mode=velocity` 且 `endpoint_film_enabled=false`，style 信号未直接进入 endpoint head；head 只能依赖 trunk 输出的隐式 style 表示，无法独立恢复高频风格细节。

### 3.2 主要统计塌缩类型

| 塌缩类型 | 证据 | 严重程度 |
|---|---|---|
| **Attention 平均化** | `cross_attn_entropy ≈ ln(256)` | 高 |
| **FiLM 后 GN 洗掉通道调制** | `pre_film_gamma_abs=0.20 > film_gamma_abs=0.15` | 中 |
| **Endpoint head 高频缺失** | `c_high/c ≈ 0.13` | **极高** |
| **Trunk 方差压缩** | 不存在（block std 增长） | 低 |

**结论**：当前模型不存在传统意义上的“整体端点 shrinkage”（α 已提升到 0.62），而是进入了 **“高频选择性塌缩”**  regime：低频迁移尚可，高频纹理/对比度/饱和度迁移严重不足。

---

## 4. 与运行时可观测性（runtime observability）的对照

`full_eval_wfi/epoch_0001/summary.json` 中的 runtime observability 与本 probe 一致：

| 指标 | Runtime Observability | Probe (t=0.0) | 差异原因 |
|---|---:|---:|---|
| `model_style_gate_value` | 0.294 | 0.294 | 一致 |
| `model_cross_attn_entropy` | 5.531 | 5.528 | 样本不同，一致 |
| `model_film_gamma_abs` | 0.122 | 0.150 | eval 与 probe 的统计方式不同 |
| `model_endpoint_pred_abs` | 0.522 | 0.698 | eval 使用 `endpoint_lowhigh` 相关统计，probe 为 velocity 模式 |
| `model_velocity_abs` | 0.289 | 0.618 | eval 平均包含多种 t，probe 固定样本 |

两者共同确认：style gate 已打开，attention 接近均匀，endpoint head 为 velocity 模式且无 FiLM。

---

## 5. Probe 结论

1. **`gated` 基线已修复整体端点 shrinkage**：`endpoint_alpha/(1-t) ≈ 0.62`，说明 style 信号已能驱动 latent 整体朝目标方向移动。
2. **高频端点 shrinkage 是剩余白化的主因**：`endpoint_high_alpha/(1-t) ≈ 0.078`，高频分量几乎未迁移。
3. **Trunk 无层内统计塌缩**：block 输出方差逐层放大，问题不在 GroupNorm 的逐层压缩。
4. **Attention 仍接近均匀**：`cross_attn_entropy ≈ ln(256)`，条件期望坍缩仍然存在，但 block 内 FiLM 已部分绕过它。
5. **Endpoint head 缺少 style 调制**：`endpoint_film_enabled=false`、`endpoint_head_mode=velocity`，style 无法直接调制 velocity 的高频分量。

**下一优先实验**：启用 `endpoint_head_mode=endpoint_lowhigh` 并打开 `endpoint_film_enabled=true`，让 style 信号直接调制 endpoint 的低频与高频分量，重点验证 `endpoint_high_alpha/(1-t)` 是否能从 0.08 提升到 0.3 以上。
