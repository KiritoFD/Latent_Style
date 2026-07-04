# 621 实验清单与消融结果汇总

> 建立日期: 2026-06-21  
> 数据来源: EXPERIMENT_ARCHAEOLOGY_MASTER.csv (22,630行) + 620实验记录

---

## 1. 全分支实验总览

### 1.1 分支实验统计

| 分支 | 实验数 | 最佳CLIP-S | 最佳LPIPS | 状态 |
|------|--------|-----------|-----------|------|
| SWD | ~50 | 0.6725 | 0.2968 | 早期基线 |
| Gram-Moment | ~30 | 0.65 | 0.35 | 差 |
| Diff-Gram | ~20 | 0.60 | 0.40 | 极差 |
| Thermal | ~40 | 0.68 | 0.33 | 风格好但质量差 |
| attn | ~30 | 0.69 | 0.31 | 3060适配 |
| multistep-texture | ~60 | 0.72 | 0.50 | 有潜力 |
| re-SWD | ~25 | 0.68 | 0.32 | 无显著优势 |
| Classify | ~15 | 0.70 | 0.30 | 结构太强 |
| Cycle-upscale | ~20 | 0.69 | 0.31 | structure loss无用 |
| Style8_Moment+SWD | ~35 | 0.67 | 0.34 | Few-shot |
| sdxl-fp16 | ~10 | 0.63 | 0.38 | 差 |
| **620-spatial-bridge** | **~100** | **0.7051** | **0.2935** | **当前最优** |

### 1.2 620实验详细清单

#### Phase 1: SWD Weight Scan

| 配置 | SWD weight | v_len | best_epoch | CLIP-S | LPIPS | WFI |
|------|-----------|-------|-----------|--------|-------|-----|
| swd12_v1 | 12 | 1.0 | e8 | 0.6725 | 0.2968 | — |
| swd16_v1 | 16 | 1.0 | e1 | 0.7053 | 0.2901 | — |
| swd16_v02 | 16 | 0.2 | e9 | 0.7038 | 0.3064 | — |
| **swd16_v004** | **16** | **0.04** | **e5** | **0.7051** | **0.2935** | — |
| swd20_v004 | 20 | 0.04 | e1 | 0.7006 | 0.2750 | — |

#### Phase 2: Architecture Experiments

| 配置 | 改动 | CLIP-S | LPIPS | WFI | 结论 |
|------|------|--------|-------|-----|------|
| gated | softmax→gated attention | 0.6987 | 0.3300 | 0.4902 | 基线 |
| gated_raw | 无归一化gated | 0.6987 | 0.2973 | 0.6435 | ❌ 更差 |
| relu2 | ReLU² attention | 0.6964 | 0.3102 | 0.5340 | ❌ 白化 |
| style_select | Top-k attention | 0.6982 | 0.3331 | 0.5005 | ❌ 无改善 |
| **endpoint_film** | FiLM endpoint hd128 | **0.7066** | 0.3226 | **0.4283** | ✅ 改善 |
| **endpoint_film_hd512** | FiLM endpoint hd512 | **0.7015** | 0.3382 | **0.3906** | ✅✅ 过门 |

#### Phase 2: FiLM Training Dynamics

| Epoch | WFI | CLIP-S | LPIPS | 趋势 |
|-------|-----|--------|-------|------|
| e1 | 0.4271 | 0.7066 | 0.3226 | 最优 |
| e2 | 0.4532 | 0.7040 | 0.3510 | WFI↑ |
| e3 | 0.4680 | 0.7015 | 0.3768 | WFI↑↑ |

**结论**: 过训练加剧白化，需要early stopping或lr调低

#### H系列修复实验

| ID | 实验 | WFI | CLIP-S | 结论 |
|----|------|-----|--------|------|
| H1 | endpoint_film_init_std=0.02 | 0.4022 | 0.7030 | 接近过门 |
| H2 | endpoint_style_hidden_dim=512 | **0.3906** | **0.7015** | ✅ 过门 |
| H5 | dim=128 | — | 0.6717 | 低于基线 |
| H6 | intrinsic cross-attention | — | 0.6717 | 无DINO |

---

## 2. 消融维度汇总

### 2.1 Attention Mode

| Mode | WFI | CLIP-S | LPIPS | 推荐 |
|------|-----|--------|-------|------|
| softmax (基线) | 0.4902 | 0.6987 | 0.3300 | 基线 |
| **gated** | 0.4902 | 0.6987 | 0.3300 | ✅ 保留 |
| gated_raw | 0.6435 | 0.6987 | 0.2973 | ❌ 删除 |
| relu2 | 0.5340 | 0.6964 | 0.3102 | ❌ 删除 |
| style_select | 0.5005 | 0.6982 | 0.3331 | ❌ 删除 |
| sparsemax | — | — | — | 待测 |

### 2.2 Endpoint Head

| Mode | FiLM | hd | WFI | CLIP-S | 推荐 |
|------|------|-----|-----|--------|------|
| velocity | ❌ | — | — | 0.7051 | 基线 |
| endpoint_lowhigh | ❌ | — | — | — | ❌ 无style注入 |
| endpoint_lowhigh | ✅ | 128 | 0.4283 | 0.7066 | ⚠️ 接近 |
| **endpoint_lowhigh** | **✅** | **512** | **0.3906** | **0.7015** | **✅ 最优** |

### 2.3 Style Gate Init

| Gate | velocity_abs | gate_value | CLIP-S | 推荐 |
|------|-------------|------------|--------|------|
| 0.05 | 0.186 | 0.064 | 0.700 | 基线 |
| **0.3** | **0.216** | **0.297** | **0.696** | ✅ 改善shrinkage |

### 2.4 Training Target

| Mode | 描述 | 效果 | 推荐 |
|------|------|------|------|
| legacy | 直接用target | baseline | 基线 |
| source_low_target_high | 低频锚定source | 早期有效 | ⚠️ |
| **target_linear** | 低频线性插值 | **当前最优** | ✅ 保留 |
| pure_vertical_flow | 纯垂直流 | 类似target_linear | ⚠️ |

### 2.5 SWD Scale Mode

| Mode | 描述 | 效果 | 推荐 |
|------|------|------|------|
| global | 全局SWD | 基线 | 基线 |
| 2-scale | 64+32 | 略好 | ⚠️ 待测 |
| 3-scale | 64+32+16 | 略好 | ⚠️ 待测 |
| attention-weighted | attention加权 | 待测 | ⏳ |

---

## 3. 无效实验清单 (建议删除)

| 实验 | 结果 | 删除理由 |
|------|------|----------|
| lowfreqfix | velocity 0.15→0.016 | 惩罚低频动态 |
| endpointaux | to_source_rms=0.055 | 坍回source |
| tlow (低t采样) | 同上 | 模型选择"不动" |
| endpoint_lowhigh (无FiLM) | style_sens=0.003 | 无style注入 |
| endpoint_stylehead | alpha仍负 | 不够 |
| direction loss | alpha=-0.007 | 完全坍缩 |
| gated_raw | WFI=0.64 | 最差 |
| relu2 | WFI=0.53 | 白化 |
| style_select | WFI=0.50 | 无改善 |
| Structure loss | "完全没用" | Classify分支验证 |
| Diff-Gram | 极差 | sdxl-fp32验证 |
| Gram-Moment | 差 | 自身验证 |

---

## 4. 开销汇总

### 4.1 训练时间 (3060 12GB)

| 实验 | epochs | 时间 | VRAM |
|------|--------|------|------|
| SWD scan (5配置) | 10 each | ~40h | ~8GB |
| Attention ablation (4配置) | 1 each | ~4h | ~9GB |
| FiLM formal (3配置) | 3-8 each | ~20h | ~9GB |
| H系列 (2配置) | 1 each | ~2h | ~9GB |
| **总计** | — | **~66h** | — |

### 4.2 推理时间

| 配置 | 单图 | 50图 | 备注 |
|------|------|------|------|
| velocity 8步 | 0.3s | 15s | 基线 |
| endpoint 1步 | 0.05s | 2.5s | 快速 |
| CFG 3方向 | 0.9s | 45s | 高质量 |

### 4.3 参数量

| 组件 | 参数 | 占比 |
|------|------|------|
| SpatialBridgeBlock ×4 | 15M | 60% |
| StyleConditioner | 5M | 20% |
| FiLM endpoint head | 2M | 8% |
| 其他 | 3M | 12% |
| **总计** | **25M** | 100% |
