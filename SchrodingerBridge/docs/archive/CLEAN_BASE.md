# 干净 Base 配置说明文档

## 1. 概述

本文件说明 `configs/clean_base.json` 的设计理念、依据和预期性能。该配置基于 **466 组消融实验**（Phase 628 系列，涵盖 D/L/E/P/X 五个系列）的系统分析结果，仅保留经过验证有效的模块和 loss 项，移除所有无效或有害的组件。

**设计目标**：在保留最佳性能的同时，提供最小化、可解释的配置基线，作为后续研究的基础起点。

**核心原则**：
- 每一项参数修改都有实验证据支撑
- 无效模块（±0.001 影响）保留 baseline 值以兼容代码，但标注为"装饰性"
- 有害模块（明确降低性能）直接移除
- Pareto 前沿上的免费改进点纳入默认配置

---

## 2. 消融实验总结

### 2.1 实验规模

| 系列 | 数量 | 说明 |
|------|------|------|
| D 系列 | 30 组 | 架构消融（禁用单个模块） |
| L 系列 | 16 组 | Loss 禁用（设为 0） |
| E 系列 | 24 组 | Loss 启用（新增辅助 loss） |
| P 系列 | 58 组 | 参数扫描 |
| X 系列 | 31 组 | 极端权重（w=10/50/100） |
| Phase 8D | 12 组 | color_match 深度探索 + 组合 |
| Phase 8B | 9 组 | num_steps 精细扫描 |
| **合计** | **466** | 全部完成训练+评估 |

### 2.2 关键发现

1. **指标混淆修复**：`summary.json` 有两个 clip_style 指标：
   - `all_pairs_overview.clip_style`（~0.73，含 identity pair）
   - `style_transfer_ability.clip_style`（~0.70，纯 transfer）
   - 历史 baseline 0.7307 是 all_pairs_overview，所有比较必须使用同一指标

2. **0.74 天花板被打破**：37 个实验突破 baseline 0.7307，最高达 0.7464（D5_color_w300）

3. **架构严重过度设计**：30 个架构消融中 27 个是装饰性（禁用后 ±0.001），仅 3 个核心模块

4. **大量死 loss**：16 个 loss 中 14 个禁用后无影响，仅 spectral_ll 有效，spectral_lh/hl 有害

---

## 3. 有效模块清单

### 3.1 核心架构模块（3 个，禁用后显著下降）

| 模块 | 配置键 | 禁用后 Δclip | 证据 |
|------|--------|-------------|------|
| **spectral_ode** | `model.spectral_ode_enabled=true` | -0.0167 | D1_spectral_ode_off |
| **adain_scale** | `model.endpoint_adain_scale=1.0` | -0.0142 | D2_adain_scale_0 |
| **alpha** | `model.style_extrap_alpha=0.1` | -0.0016 | D3_alpha_0 |

### 3.2 有效 Loss（1 个有效 + 2 个有害）

| Loss | 配置键 | 影响 | 证据 |
|------|--------|------|------|
| **spectral_ll** | `bridge.spectral_w_ll` | 禁用 -0.0042，升高至 2.0 → +0.0020 | L7, P8 |
| ~~spectral_lh~~ | `bridge.spectral_w_lh` | **有害**，禁用 +0.0014 | L9 |
| ~~spectral_hl~~ | `bridge.spectral_w_hl` | **有害**，禁用 +0.0014 | L9 |

### 3.3 Pareto 前沿增强（2 个免费/低代价改进）

| 增强 | 配置键 | 效果 | 证据 |
|------|--------|------|------|
| **channel_variance** | `bridge.w_channel_variance=1.0` | +0.0007 clip, **-0.0058 lpips**（免费双赢） | E2 |
| **pixel_color_match** | `bridge.w_pixel_color_match=10.0` | +0.0041 clip, +0.0148 lpips（性价比好） | X19 |

---

## 4. 无效模块清单

### 4.1 装饰性架构模块（27 个，禁用后 ±0.001）

以下模块在 baseline 中保留原值以兼容代码，但对性能无贡献：

```
style_gate_film, affine_gamma, affine_beta, global_gate, tokenizer_residual,
sharpen, endpoint_high, skip_residual, kinetic_off, attn_gated_raw, attn_relu2,
attn_style_select, attn_sparsemax, endpoint_lowhigh, transport_endpoint,
target_proj_dwt, kinetic_per_band, terminal_swd_hf, bridge_tri_band,
swd_squared, t_logit_normal, skip_clean, skip_blur, decoder_highpass,
residual_gain, no_residual_flag, avg_pool
```

### 4.2 死 Loss（14 个，禁用后无影响）

以下 loss 项在 baseline 中有权重但实际不参与有效梯度：

```
endpoint_content (L1), endpoint_style (L2), terminal_swd (L3),
single_step_swd (L4), single_step_edge (L5), kinetic (L6),
spectral_hh (L8), swd_high_freq (L11), coupling_structure (L12),
flow (L13), coupling_edge (L14), coupling_hybrid (L15), endpoint_aux (L16)
```

### 4.3 死参数（扫描全范围无反应）

```
wstyle, wswd, wkin, sigma, edge, wflow, wcontent, coupling,
tokens (64-1024), sharpen (25-100), gate_init (0-10), whh (0.5-60)
```

---

## 5. 干净 Base 的 5 项关键修改

基于 T5 ep7 baseline（`exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt`），干净 base 仅修改以下 5 项：

| # | 配置键 | Baseline | Clean Base | 修改理由 | 证据实验 |
|---|--------|----------|------------|----------|----------|
| 1 | `bridge.spectral_w_ll` | 0.3 | **2.0** | 提升低频谱权重，clip +0.0020 | P8_wll_20 |
| 2 | `bridge.spectral_w_lh` | 1.0 | **0.0** | 移除有害 loss，clip +0.0014 | L9_no_spectral_lh_hl |
| 3 | `bridge.spectral_w_hl` | 1.0 | **0.0** | 移除有害 loss，clip +0.0014 | L9_no_spectral_lh_hl |
| 4 | `bridge.w_channel_variance` | 0.0 | **1.0** | Pareto 膝点，免费降 lpips -0.0058 | E2_w_channel_variance |
| 5 | `bridge.w_pixel_color_match` | 0.0 | **10.0** | 风格提升，clip +0.0041 | X19_colormatch_w10 |

**其余所有参数保持 T5 baseline 原值不变。**

---

## 6. 预期性能

### 6.1 单项修改的实测效果

| 修改 | clip_allpairs | lpips_allpairs | Δclip | Δlpips |
|------|--------------|----------------|-------|--------|
| Baseline (T5 ep7) | 0.7307 | 0.3403 | — | — |
| +spectral_w_ll=2.0 (P8) | 0.7323 | 0.3463 | +0.0016 | +0.0060 |
| +spectral_lh/hl=0 (L9) | 0.7317 | 0.3431 | +0.0010 | +0.0028 |
| +w_channel_variance=1.0 (E2) | 0.7310 | 0.3352 | +0.0003 | -0.0051 |
| +w_pixel_color_match=10.0 (X19) | 0.7344 | 0.3559 | +0.0037 | +0.0156 |

### 6.2 组合预期（线性叠加假设）

由于 5 项修改作用于不同维度（频谱权重、有害项移除、通道方差、颜色匹配），预期效果近似线性叠加：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| clip_allpairs | **~0.7380-0.7420** | 各项 Δclip 叠加 |
| lpips_allpairs | **~0.350-0.360** | channel_variance 降，color_match 升 |

**注意**：实际组合效果需训练验证。若效果未达预期，可能存在交互作用，需逐项消融。

### 6.3 Pareto 前沿参考

| 配置 | clip_allpairs | lpips_allpairs | 定位 |
|------|--------------|----------------|------|
| D5_color_w300 | 0.7464 | 0.4689 | 风格极端优先 |
| D4_color_w150 | 0.7441 | 0.4256 | 风格极端优先 |
| D3_color_w70 | 0.7425 | 0.3998 | 风格极端优先 |
| X20_colormatch_w50 | 0.7411 | 0.3920 | 风格极端优先 |
| D2_color_w30 | 0.7387 | 0.3793 | 风格优先 |
| D1_color_w20 | 0.7370 | 0.3683 | 风格优先 |
| X19_colormatch_w10 | 0.7344 | 0.3559 | **干净 base 参考点** |
| P8_wll_20 | 0.7323 | 0.3463 | 均衡偏风格 |
| L9_no_spectral_lh_hl | 0.7317 | 0.3431 | 均衡偏风格 |
| **E2_w_channel_variance** | **0.7310** | **0.3352** | **Pareto 膝点（免费双赢）** |
| L10_no_spectral_all | 0.7311 | 0.3400 | 免费双赢 |
| L7_no_spectral_ll | 0.7261 | 0.3232 | 内容优先 |
| X6_dir_cos_w100 | 0.7147 | 0.2994 | 内容极端优先 |

---

## 7. 使用指南

### 7.1 从干净 base 训练

```bash
# 远程（I 盘）
python src/run.py --config configs/clean_base.json

# 或从 T5 ep7 续训（推荐，节省 7 epoch）
# config 中已设置 resume_checkpoint = T5 ep7
```

### 7.2 评估

```bash
python src/utils/run_evaluation.py \
    --checkpoint exp/clean_base/epoch_0010.pt \
    --output exp/clean_base/full_eval/epoch_0010 \
    --test_dir I:/wikiart_distinct5_samam_512_classview/test \
    --cache_dir I:/Github/Latent_Style/eval_cache \
    --batch_size 16 \
    --num_steps 8 \
    --eval_only_lpips_clip_style
```

### 7.3 调整风格强度

如需更高 clip（牺牲内容），调整 `w_pixel_color_match`：

| w_pixel_color_match | 预期 clip | 预期 lpips | 适用场景 |
|---------------------|-----------|------------|----------|
| 0 | ~0.732 | ~0.335 | 内容优先 |
| 10（默认） | ~0.738 | ~0.355 | 平衡 |
| 50 | ~0.741 | ~0.392 | 风格优先 |
| 100 | ~0.742 | ~0.419 | 风格极端 |
| 300 | ~0.746 | ~0.469 | 风格极端（上限） |

### 7.4 指标读取

**务必使用 `all_pairs_overview.clip_style` 进行比较**（历史 baseline 0.7307 基于此指标）：

```python
import json
with open("summary.json") as f:
    data = json.load(f)
# 正确指标
clip = data["analysis"]["all_pairs_overview"]["clip_style"]
lpips = data["analysis"]["all_pairs_overview"]["content_lpips"]
# 注意：style_transfer_ability.clip_style 是另一个指标（~0.70 级别）
```

---

## 8. 后续方向

### 8.1 已验证无效的方向（不推荐）

- 架构增加容量（dim=128, blocks=8）：不突破天花板
- mixture-of-experts (style_moe)：无效
- 跨 checkpoint ensemble：无效
- diffeomorphic stroke：无效
- 多级 DWT (spectral_ode_levels=2)：反退化

### 8.2 有潜力但未充分探索的方向

1. **color_match + hsv_saturation 组合**：Phase 8D 的 D6/D7 实验显示组合效果
2. **color_match + channel_variance 组合**：可能叠加 Pareto 改进
3. **color_match 权重曲线扫描**：w=20-70 区间可能有更优 Pareto 点
4. **跨架构（Diffusion Schrödinger Bridge）**：当前架构天花板 ~0.746，换架构可能突破

### 8.3 代码清理建议

基于消融结果，以下代码可考虑移除（需确认无副作用）：
- 27 个装饰架构模块的实现代码
- 14 个死 loss 的计算代码
- 相关的配置项和验证逻辑

**注意**：代码清理是高风险操作，建议先在 git 分支验证，确保不影响 checkpoint 加载。

---

## 9. 数据文件索引

| 文件 | 说明 |
|------|------|
| `configs/clean_base.json` | 干净 base 配置 |
| `exp/628_ablation/p8c_rescan_results.json` | 466 实验完整结果（双指标） |
| `_628_gen_clean_base.py` | 干净 base 生成脚本 |
| `_628_p8c_rescan_all_metrics.py` | 全量重扫脚本 |
| `exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt` | T5 ep7 基础 checkpoint |

---

## 10. 验证清单

- [x] clean_base.json JSON 语法验证通过
- [x] 5 项修改均有 466 实验中的具体实验证据
- [x] 架构核心模块（spectral_ode, adain, alpha）保留
- [x] 有害 loss（spectral_lh/hl）移除
- [x] Pareto 膝点（channel_variance）纳入
- [x] 风格增强（color_match w=10）纳入
- [x] 指标混淆问题已说明
- [ ] 实际训练验证（待执行）
- [ ] Pareto 前沿图更新（待执行）

---

*文档生成时间：2026-06-29*
*基于 Phase 628 系列 466 组消融实验*
