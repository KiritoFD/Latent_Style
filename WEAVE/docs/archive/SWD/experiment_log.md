# Semantic SWD 实验记录

## 阶段一: k-means 区域划分路线 (S1-S4) — 已废弃

### 实验环境
- GPU: RTX 3060 12GB (本地)
- 数据集: distinct5-512 (5风格, 512分辨率)
- 训练: batch_size=48, num_epochs=10, lr=2e-4, Patience=2
- 评估: 750张生成图 (5风格 × 150源图), batch_size=2

### S1: sem_region (k-means 区域匹配)
- **配置**: `configs/musiq_s1_sem_region.json`
- **机制**: k-means 在 content latent 上聚类 8 区域, blend=0.7
- **训练时间**: ~3.5min (10 epochs, 33 it/s)
- **显存**: 6.35GB/12GB
- **结果**: MUSIQ=41.59, CLIP-S=0.7245, LPIPS=0.5067
- **分析**: MUSIQ 微升 (+0.48), 但 LPIPS 恶化 (+0.072)。k-means 区域匹配扭曲了内容。
- **结论**: 失败

### S2: sem_patch (k-means + 多尺度 patch)
- **配置**: `configs/musiq_s2_sem_patch.json`
- **机制**: k-means 6区域 + patch_sizes=[1,3,5], blend=0.7
- **结果**: MUSIQ=38.89, CLIP-S=0.7047, LPIPS=0.5311
- **分析**: MUSIQ 下降 (-2.22)。区域内 patch 匹配过于激进，破坏了全局纹理。
- **结论**: 失败

### S3: sem_band (k-means per DWT subband)
- **配置**: `configs/musiq_s3_sem_band.json`
- **机制**: k-means 在每个 DWT 子带内独立聚类, w_hh=2.0
- **结果**: MUSIQ=40.92, CLIP-S=0.6976, LPIPS=0.4240
- **分析**: CLIP-S 下降 (-0.030)。频段分解 + k-means 组合不稳定。
- **结论**: 失败

### S4: sem_xattn (cross-attn guidance + k-means)
- **配置**: `configs/musiq_s4_sem_xattn.json`
- **机制**: cross-attn entropy 作为 importance sampling + k-means 区域匹配, blend=0.7
- **结果**: MUSIQ=42.01, CLIP-S=0.7095, LPIPS=0.5320
- **分析**: MUSIQ 最高 (+0.90), 但 LPIPS 失控 (+0.097 > 0.48 目标)。
  - cross-attn guidance 有效（S4 > S1 证明）
  - k-means 仍然伤害 LPIPS（S4 LPIPS=0.5320 >> baseline 0.4347）
- **结论**: guidance 方向正确，区域划分必须去掉

### 阶段一总结

| 方向 | MUSIQ | CLIP-S | LPIPS | MUSIQΔ | LPIPSΔ |
|------|-------|--------|-------|--------|--------|
| Baseline | 41.11 | 0.7275 | 0.4347 | — | — |
| S1 | 41.59 | 0.7245 | 0.5067 | +0.48 | +0.072 |
| S2 | 38.89 | 0.7047 | 0.5311 | -2.22 | +0.096 |
| S3 | 40.92 | 0.6976 | 0.4240 | -0.19 | -0.011 |
| S4 | 42.01 | 0.7095 | 0.5320 | +0.90 | +0.097 |

**关键洞察**: 
1. k-means 区域划分一致地恶化 LPIPS（+0.07~+0.10）
2. cross-attn guidance 是唯一有效的 MUSIQ 提升信号（S4 > S1）
3. 正确方向: 去掉 k-means，保留 guidance，用全局重要性采样

**代码清理**: 删除 `_kmeans_labels`, `_semantic_region_swd`, `_semantic_patch_swd`, `_semantic_band_swd` 四个死函数和对应的 3 个 dispatch 分支。

---

## 阶段二: Guidance-based Semantic SWD (S5-S8) — 进行中

### 设计原理

从 region-based 转向 guidance-based:
- **Region-based (S1-S4)**: 划分区域 → 区域内独立匹配 → 破坏全局约束
- **Guidance-based (S5-S8)**: 全局匹配 → guidance 调整采样权重 → 保留全局约束 + 聚焦关键区域

详见 `semantic_swd_theory.md` 第3节。

### S5: band-split + cross-attn-guided
- **配置**: `configs/musiq_s5_band_xattn.json`
- **机制**: DWT 4子带分解 (LL=0.25, LH=1.0, HL=1.0, HH=2.0) + cross-attn importance sampling
- **假设**: HF频段重点约束 + 编辑区域聚焦 → MUSIQ提升
- **状态**: 待训练

### S6: multi-patch + cross-attn-guided
- **配置**: `configs/musiq_s6_patch_xattn.json`
- **机制**: 多尺度patch (1×1=0.3, 3×3=0.4, 5×5=0.3) + cross-attn importance sampling
- **假设**: 多尺度纹理匹配 + 编辑区域聚焦 → MUSIQ提升
- **状态**: 待训练

### S7: DWT-energy guided
- **配置**: `configs/musiq_s7_dwt_energy.json`
- **机制**: content DWT高频能量作为 guidance signal (不依赖模型内部状态)
- **假设**: 纹理丰富区域重点约束 → MUSIQ提升
- **状态**: 待训练

### S8: combined (cross-attn × DWT)
- **配置**: `configs/musiq_s8_combined.json` (待创建)
- **机制**: cross-attn entropy × DWT energy 逐元素乘积
- **假设**: 两个signal互补 → 既关注转移区域又关注纹理区域
- **状态**: 待创建配置

---

## 时间线

- 2026-07-07: 阶段一 S1-S4 训练+评估完成
- 2026-07-08: 代码清理（删除k-means死代码），写理论文档，创建S5-S8配置
- 2026-07-08: 阶段二 S5-S8 训练（进行中）
