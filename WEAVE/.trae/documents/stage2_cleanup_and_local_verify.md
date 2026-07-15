# Stage 2 清理 + 本地验证 + Infra 优化计划

## 摘要

系统性删除 14 项确认无效的 decorative_arch baseline 分支代码，硬编码 prune_to 值。清理后本地训练+评估验证性能无下降。最后做基本 infra 优化（清理临时脚本、创建本地一键脚本）。

## 当前状态分析

### 本地 baseline（已确立）
- **allpairs clip_style = 0.7293**（远程 0.7299，Δ=0.0006 噪声范围）
- **allpairs content_lpips = 0.3203**（远程 0.3420）
- 训练：3 分钟/10epoch，显存峰值 0.36 GB
- Eval：约 3 分钟，750 张 allpairs
- 数据：5000 张 latent 已编码（与远程完全一致 mean=0.0737 std=0.9602）

### 14 项确认无效 decorative_arch（来自 prune_manifest.json）

| # | 配置键 | baseline | prune_to | 所在文件 |
|---|--------|----------|----------|----------|
| D13 | tokenizer_global_gate_scale | 1.0 | 0 | config_schema.py:267, lancet_backbone.py:307 |
| D14 | tokenizer_residual_gain | 0.5 | 0 | config_schema.py:232, lancet_backbone.py:289 |
| D15 | style_attn_sharpen_scale | 2.5 | 0 | config_schema.py:294, lancet_backbone.py:96, lancet_blocks.py:208,218,228,339,352,666,679,688 |
| D16 | endpoint_high_scale | 1.0 | 0 | config_schema.py:321, model620.py:91,458,461 |
| D17 | skip_residual_weight | 0.1 | 0 | config_schema.py:364, lancet_backbone.py:161 |
| D18 | kinetic_penalty_mode | "global_l2" | "off" | config_schema.py:634 |
| D19 | style_attn_mode | "softmax" | "relu2" | config_schema.py:316, model620.py:113,212 |
| D23 | endpoint_head_mode | "velocity" | "endpoint_lowhigh" | config_schema.py:318, model620.py:83-85,221,453,531,536 |
| D24 | transport_prediction_mode | "velocity" | "endpoint" | config_schema.py:351,979, model.py:48-50,77,1102,1112,1892,1939,1962,1982, model620.py:82 |
| D25 | training_target_projection_mode | "legacy" | "dwt" | config_schema.py:576, losses620.py:59-70,242-315 |
| D27 | terminal_swd_mode | "standard" | "high_freq" | config_schema.py:652 |
| D28 | bridge_path_mode | "vertical" | "tri_band" | config_schema.py:566, losses620.py:134-136,339 |
| D29 | swd_distance_mode | "cdf" | "squared" | config_schema.py:658 |
| D30 | t_sampling_mode | "uniform_power" | "logit_normal" | config_schema.py:529, losses620.py:81,214 |

### 关键架构发现

- **SpectralODEBridge620**（clean_base_v2 实际使用的模型）继承 `nn.Module`，**不继承** TimeConditionedLANCETBridge/SpatialBridge620
- 它**不读取**这 14 项配置中的任何一项
- 14 项 baseline 分支只在**遗留代码路径**上（model620.py, model.py, lancet_*.py, losses620.py）
- trainer.py 分发：`620_spectral_ode` → SpectralODEObjective620（不是 SpatialBridgeObjective620）
- **结论：清理这 14 项 baseline 分支不会影响 clean_base_v2 的性能**（因为不在其代码路径上）

## 提议变更

### Stage 2A: config_schema.py 默认值更新

**文件**: `src/config_schema.py`

将 14 项配置的默认值从 baseline 改为 prune_to：

```python
# 行 232: tokenizer_residual_gain: float = 0.5 → 0
tokenizer_residual_gain: float = 0

# 行 267: tokenizer_global_gate_scale: float = 1.0 → 0
tokenizer_global_gate_scale: float = 0

# 行 294: style_attn_sharpen_scale: float = 2.5 → 0
style_attn_sharpen_scale: float = 0

# 行 316: style_attn_mode: str = "softmax" → "relu2"
style_attn_mode: str = "relu2"

# 行 318: endpoint_head_mode: str = "velocity" → "endpoint_lowhigh"
endpoint_head_mode: str = "endpoint_lowhigh"

# 行 321: endpoint_high_scale: float = 1.0 → 0
endpoint_high_scale: float = 0

# 行 351: transport_prediction_mode: str = "velocity" → "endpoint"
transport_prediction_mode: str = "endpoint"

# 行 364: skip_residual_weight: float = 0.1 → 0
skip_residual_weight: float = 0

# 行 529: t_sampling_mode: str = "uniform_power" → "logit_normal"
t_sampling_mode: str = "logit_normal"

# 行 566: bridge_path_mode: str = "linear" → "tri_band"
bridge_path_mode: str = "tri_band"

# 行 576: training_target_projection_mode: str = "legacy" → "dwt"
training_target_projection_mode: str = "dwt"

# 行 634: kinetic_penalty_mode: str = "global_l2" → "off"
kinetic_penalty_mode: str = "off"

# 行 652: terminal_swd_mode: str = "standard" → "high_freq"
terminal_swd_mode: str = "high_freq"

# 行 658: swd_distance_mode: str = "cdf" → "squared"
swd_distance_mode: str = "squared"
```

**原因**: 确保即使用户不设置这些字段，默认值也是确认有效的 prune_to 值。

### Stage 2B: 遗留代码 baseline 分支删除

**策略**: 删除 baseline 分支代码，硬编码 prune_to 值。保留类结构和接口，只删分支逻辑。

#### 文件 1: `src/model620.py` (SpatialBridge620)

- 行 82-85: `transport_prediction_mode` / `endpoint_head_mode` 默认值和校验 → 硬编码为 "endpoint" / "endpoint_lowhigh"
- 行 91: `endpoint_high_scale` 读取 → 硬编码为 0
- 行 113: `style_attn_mode` 读取 → 硬编码为 "relu2"
- 行 221, 453, 531, 536: `endpoint_head_mode == "endpoint_lowhigh"` 分支变为唯一路径，删除条件判断
- 行 458, 461: `endpoint_high_scale` 乘法变为 `* 0`，简化为直接返回 0 或删除该路径
- 行 212: `attn_mode=self.style_attn_mode` → 直接传 "relu2"

#### 文件 2: `src/model.py` (TimeConditionedLANCETBridge)

- 行 48-50: `transport_prediction_mode` 默认值和校验 → 硬编码为 "endpoint"
- 行 77: `transport_prediction_mode` 传参 → 直接传 "endpoint"
- 行 1102, 1112, 1892, 1939, 1962, 1982: `transport_prediction_mode == "endpoint"` 分支变为唯一路径，删除条件判断

#### 文件 3: `src/lancet_backbone.py` (LatentAdaCUT)

- 行 96: `style_attn_sharpen_scale` 读取 → 硬编码为 0
- 行 161: `skip_residual_weight` 读取 → 硬编码为 0
- 行 289: `tokenizer_residual_gain` 传参 → 直接传 0
- 行 307: `tokenizer_global_gate_scale` 读取 → 硬编码为 0
- 行 441: `style_attn_sharpen_scale` 传参 → 直接传 0

#### 文件 4: `src/lancet_blocks.py` (StyleMaps)

- 行 208, 339: `style_attn_sharpen_scale: float = 2.0` 默认值 → 0
- 行 218, 228, 352, 679, 688: `attn_sharpen_scale=style_attn_sharpen_scale` 传参 → 直接传 0 或保留参数但默认 0

#### 文件 5: `src/losses620.py` (SpatialBridgeObjective620)

- 行 59-70: `training_target_projection_mode` 默认值和校验 → 硬编码为 "dwt"
- 行 81: `t_sampling_mode` 读取 → 硬编码为 "logit_normal"
- 行 134-136: `bridge_path_mode` 默认值和校验 → 硬编码为 "tri_band"
- 行 214: `t_sampling_mode == "logit_normal"` 分支变为唯一路径
- 行 242-315: `training_target_projection_mode` 各分支简化为 "dwt" 路径
- 行 339: `bridge_path_mode == "spherical_vp"` 分支删除（baseline "linear" 已改为 "tri_band"）

#### 文件 6: `src/trainer.py`

- 行 1670-1673: `training_target_projection_mode_*` metrics 默认值 → 保留（不影响功能，仅是 metrics 占位）

### Stage 2C: 本地验证

1. **Smoke test**: `python _remote_smoke_train.py`（forward+backward+optimizer）
2. **训练**: `python src\run.py --config configs\clean_base_v2_local.json`（10 epoch，约 3 分钟）
3. **评估**: `python src\utils\run_evaluation.py --checkpoint ... --output ... --eval_only_lpips_clip_style`（约 3 分钟）
4. **对比 baseline**: allpairs clip_style ≥ 0.7293 - 0.005 = 0.7243（5σ 噪声阈值）

### Stage 3: tri_band_inference_lock 推理分支清理

- 搜索 `tri_band_inference_lock` 在 src/ 中的位置
- 删除推理分支代码（628 I8 验证无效）
- 本地验证

### Infra 优化

1. **清理临时脚本**: 搜索并删除 `_probe_*.py`, `_dump_*.py`, `_compare_*.py`, `_remote_smoke_train.py`, `_make_local_config.py`, `_run_*.ps1`, `_start_*.ps1`, `_parse2.py` 等本次会话创建的临时文件
2. **创建本地一键脚本**: `tools\local_train_and_eval.py` - 封装训练+评估流程
3. **保留配置文件**:
   - `configs/clean_base_v2.json` - 远程参考配置（保留）
   - `configs/clean_base_v2_local.json` - 本地配置（保留）
4. **清理 docs**: 确认 `docs/CLEAN_BASE_V2.md` 仍准确

## 假设与决策

1. **假设**: 14 项 baseline 分支不在 SpectralODEBridge620 代码路径上（已通过继承关系确认）
2. **决策**: 保留遗留文件（model620.py, losses620.py 等），只删除 baseline 分支代码 - 因为 build_model_from_config 入口点仍需要保留
3. **决策**: 不删除 TimeConditionedLANCETBridge 和 SpatialBridge620 类定义 - 它们可能被其他实验配置引用
4. **决策**: trainer.py 的 metrics 占位行保留 - 不影响功能
5. **验证标准**: allpairs clip_style ≥ 0.7243（baseline 0.7293 - 5σ）

## 验证步骤

1. ✅ Stage 2A: config_schema.py 默认值更新后，smoke test 通过
2. ✅ Stage 2B: 遗留代码清理后，smoke test 通过
3. ✅ Stage 2C: 本地训练 10 epoch + full_eval，allpairs clip_style ≥ 0.7243
4. ✅ Stage 3: tri_band_inference_lock 清理后，本地训练+eval 验证
5. ✅ Infra 优化: 临时脚本清理，一键脚本可用

## 执行顺序

1. Stage 2A: config_schema.py 默认值更新
2. Stage 2B: 6 个遗留文件 baseline 分支删除
3. Stage 2 本地验证（smoke + train + eval）
4. Stage 3: tri_band_inference_lock 清理 + 验证
5. Infra 优化（清理临时脚本 + 创建一键脚本）
6. git commit
