# 模型开销诊断 — 哪些模块值、哪些不值

> 目标: 判明 PureLatentSpatial + TopoGate + manifold_kinetic 等新增模块
> 的显存/计算开销，对照历史实验数据判断 ROI。

## 验证结论 (2026-06-18)

**PureLatentSpatial tokenizer 已确认无收益**。SMoE translator e8 (0.670/0.318) vs 旧 baseline F_e1 (0.697/0.319 on 1000per) — 复杂 tokenizer 的 tokenizer 风格增益 ≈ 0，但 VRAM 开销 ~1.2GB，导致 batch 从 44 降到 12-16。

**决策：回退至 legacy_factorized tokenizer + ablation_disable_spatial_prior=true**。
代码修复: `_prepare_style_maps` 现在检查 `ablation_disable_spatial_prior` 标志直接返回空 StyleMaps，
不再需要 `style_spatial_id_16`。

**current running batch**: `exp/20250618_lite_ot_vertical/` — b24, vl=0.1, legacy_factorized, spatial_prior off, pairing_cache off. 7 experiments.

## 发现的 infra bug

| Bug | 文件 | 修复 |
|-----|------|------|
| `virtual_length_multiplier` 写到 `training` 段, 代码从 `data` 段读 | gen_lite_batch.py | 改为 `c["data"]["virtual_length_multiplier"]` |
| `_prepare_style_maps` 不检查 `ablation_disable_spatial_prior` | lancet_runtime.py:167 | 加 `if getattr(...) return StyleMaps()` |
| `pairing_cache_path` 写到 `data` 段 (正确, 但之前也在 training) | gen_lite_batch.py | 已修正 |
| WSL 后台进程在 SSH 断开后被杀 | — | 用户在 WSL 交互终端跑 `bash launch_all.sh` |

---

## 一、吞吐量变化

| 配置 | 数据集 | batch | 样本/epoch | 步/epoch | 时间/epoch |
|------|--------|:---:|:---:|:---:|:---:|
| 旧 baseline (F/H/K) | distinct5-1000per | 44 | 5000 | 114 | **1.2 min** |
| topogate_appalign | distinct5-full | 12 | 18888 | 1574 | **25 min** |
| 当前 H0 (b16, vl=0.1) | distinct5-full | 16 | 18888 | 1181 | **~2.5 min** × vl=0.1 |

**吞吐量下降的根本原因**:

| 因素 | 倍数 | 解释 |
|------|:---:|------|
| 数据集变大 | 3.8× | 5000 → 18888 |
| batch 减小 | 2.75× | b44 → b16 (VRAM 约束) |
| 每步变慢 | ~2× | tokenizer + topogate 增加计算 |
| **综合** | **~21×** | 1.2 min → 25 min |

---

## 二、逐模块开销分析

基于 topogate b12 的训练日志数据 (samples_per_sec=12.3, cuda_peak=7.6GB):

### 2.1 PureLatentSpatial Tokenizer

**增加的开销**:
- 5 层 ResBlock query_extractor (vs 2 层 Conv)
- 32 clusters × 96 dim universal_keys + style_values
- PE 计算 (`_add_position_embedding`)
- global_gate MLP

**VRAM**: ~1.2 GB (keys/values embedding + intermediate activations)

**效果**: 
- SMoE translator e8: 0.670/0.318
- pure_latent topogate e2: 0.671/0.314
- 差距: **几乎相同** — 复杂 tokenizer 没有带来可见的风格增益

**判断**: ❌ 不值得。应降级为 lighter tokenizer (2 层 Conv + PE only)

### 2.2 TopoGate

**增加的开销**:
- 每层 Attention 额外计算 self-attention content affinity
- `semantic_self_topology_blend=1.0` 的混合计算

**效果**:
- 不带 topogate 的 safe_rescan_r2 e4: LPIPS=0.367
- 带 topogate 的 appalign e2: LPIPS=0.312
- **LPIPS 改善: -0.055 (15%)** — 这是所有模块中最大的单一增益

**判断**: ✅ 值得。TopoGate 是结构保持的核心。

### 2.3 Manifold Adaptive Kinetic

**增加的开销**:
- 低频/高频 split 的额外计算 (`_kinetic_lowpass`)

**效果**: 消融实验 (H_base path_kinetic):
- k=1.0 (base): LPIPS 0.427
- k=0.25: LPIPS 0.460
- k=0 (no kinetic): LPIPS 0.507

**判断**: ⚠️ 需要但可简化。Manifold split 比 global L2 好，但开销不大。

### 2.4 Cross-Attention Texture (proximal)

**当前状态**: 在 topogate 实验中 `proximal_mode="crossattn_texture"` 

**开销**: 在重构阶段额外一次 cross-attention pass

**效果**: 没有直接消融数据。从 xpred 系列看，pattn proximal 提升 style 约 0.003。

**判断**: ⚠️ 边际收益小，可能需要

---

## 三、历史数据对照: 模块是否有 ROI

### 表格: 相同数据集上的性能对比

| 模型 | all-pairs style | LPIPS | epoch/min | 评价 |
|------|:---:|:---:|:---:|------|
| old_baseline F_e1 (1000per) | 0.697 | 0.319 | 1.2 | 基准线 |
| safe_rescan_r2 e4 (full) | 0.701 | 0.367 | ~21 | 新 tokenizer, LPIPS 反而差 |
| topogate_appalign e2 (full) | 0.703 | 0.312 | ~25 | +topogate, LPIPS 好转 |
| **净增益** | +0.006 | +0.007→-0.055 | **×21** | LPIPS 改善主要来自 topogate |

### 结论

**PureLatentSpatial tokenizer**: 对 style/LPIPS 几乎无增益，承担 ~6GB 显存中的 ~1.2GB。**建议降级或删除**，换回 legacy tokenizer + 小 PE。

**TopoGate**: 唯一明确有收益的模块 (-0.055 LPIPS)。**强烈保留**。

**Manifold kinetic**: 需要但 global L2 可能就够了。

**Proximal cross-attn texture**: 边际收益不确定。

---

## 四、推荐消融计划

在 H0-H6 跑完之后，取最佳 checkpoint，做以下消融:

| # | 名称 | 改动 | 预计 epoch/min | 测试 |
|---|------|------|:---:|------|
| A1 | 降级 tokenizer | pure_latent → legacy_factorized + PE only | ~8 | tokenizer 是否拖慢训练 |
| A2 | 关 TopoGate | semantic_self_topology_gate=false | ~10 | TopoGate 的 LPIPS 贡献 |
| A3 | 简化为 global L2 | manifold_adaptive → global_l2 | ~12 | kinetic 模式影响 |
| A4 | 关 proximal | proximal_mode=off | ~12 | proximal 的 style 贡献 |
| A5 | A1+A2+A3+A4 全部关 | 回归最简模型 | ~15 | vs 当前 topogate 的性能差 |

**每个消融 8 个小 epoch (vl=0.1) ≈ ~20min**。
