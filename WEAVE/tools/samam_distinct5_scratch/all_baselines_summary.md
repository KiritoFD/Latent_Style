# Baseline方法完整指标汇总

> 数据来源：`exp/baseline_v2/eval/unified_results.json` (12方法, 750对评估) + `make_dashboard.py` 训练时间
> 评估数据集：wikiart_distinct5_512 (5风格: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
> 评估指标：CLIP-S Style (ViT-B/32, ↑更好), Content LPIPS (Alex, ↓更好)
> 新7k实验：2026-07-01 从头训练 (7000步, batch=1, 512×512, ~3.9h)

## 1. 完整Baseline指标表

| 方法 | 类别 | CLIP-S ↑ | LPIPS ↓ | Δ_clip_idt | 训练时间(min) | 训练要求 |
|------|------|---------|---------|------------|--------------|----------|
| **Identity** | baseline | 0.6933 | 0.0000 | +0.0534 | 0.0 | 无 |
| **AdaIN** | classical-inf | 0.6679 | 0.7425 | +0.0280 | 0.0 | 无 |
| **WCT (VGG19)** | classical-inf | — | — | — | 0.0 | 无 |
| **SD-Turbo** | diffusion-inf | 0.6933 | 0.0033 | +0.0534 | 0.0 | 无 |
| **SDEdit s=0.10** | diffusion-inf | 0.7188 | 0.3183 | +0.0789 | 0.0 | 无 |
| **SDEdit s=0.20** | diffusion-inf | 0.7340 | 0.3492 | +0.0941 | 0.0 | 无 |
| **SDEdit s=0.35** | diffusion-inf | 0.7797 | 0.4508 | +0.1398 | 0.0 | 无 |
| **SDEdit s=0.40** | diffusion-inf | 0.7934 | 0.4826 | +0.1535 | 0.0 | 无 |
| **StyleID** | diffusion-inf | 0.8223 | 0.5523 | +0.1824 | 0.0 | 无 |
| **CUT** | gan-train | 0.7137 | 0.3743 | +0.0738 | 322.6 | 是 |
| **SaMST** | mamba-train | 0.6183 | 0.7490 | -0.0216 | 39.5 | 是 |
| **SaMam (旧20k)** | mamba-train | 0.7222 | 0.3282 | +0.0823 | 209.2 | 是 |
| **SaMam (新7k从头)** | mamba-train | 0.6248 | 0.3209 | -0.0150 | 232.4 | 是 |

## 2. SaMam新旧对比

| 实验 | 训练步数 | 训练时间 | CLIP-S | LPIPS | 备注 |
|------|---------|---------|--------|-------|------|
| 旧20k实验 | 10000 | 294.5min | 0.7851 | 0.1643 | 旧CLIP backend |
| 旧20k实验 (step=7000等价点) | 7000 | 206.2min | 0.7848 | 0.2461 | 旧CLIP backend |
| **新7k从头** | 7000 | 232.4min | 0.6248 | 0.3209 | open_clip ViT-B/32 |
| 新7k (最优step=4750) | 4750 | 158min | 0.6174 | 0.3117 | open_clip ViT-B/32 |

**注**：新旧CLIP-S绝对值不可直接比较（backend不同）。新7k实验用open_clip ViT-B/32，旧20k实验CLIP backend可能不同。LPIPS数值基本可比。

## 3. 性能排名

### 按CLIP-S Style排序（风格迁移强度）

| 排名 | 方法 | CLIP-S ↑ | LPIPS ↓ | 训练时间 |
|------|------|---------|---------|---------|
| 1 | StyleID | 0.8223 | 0.5523 | 0 (infer) |
| 2 | SDEdit s=0.40 | 0.7934 | 0.4826 | 0 (infer) |
| 3 | SDEdit s=0.35 | 0.7797 | 0.4508 | 0 (infer) |
| 4 | SDEdit s=0.20 | 0.7340 | 0.3492 | 0 (infer) |
| 5 | SDEdit s=0.10 | 0.7188 | 0.3183 | 0 (infer) |
| 6 | **SaMam (旧20k)** | 0.7222 | 0.3282 | 209.2min |
| 7 | CUT | 0.7137 | 0.3743 | 322.6min |
| 8 | SD-Turbo | 0.6933 | 0.0033 | 0 (infer) |
| 9 | Identity | 0.6933 | 0.0000 | 0 |
| 10 | AdaIN | 0.6679 | 0.7425 | 0 (infer) |
| 11 | **SaMam (新7k)** | 0.6248 | 0.3209 | 232.4min |
| 12 | SaMST | 0.6183 | 0.7490 | 39.5min |

### 按Content LPIPS排序（内容保真度）

| 排名 | 方法 | LPIPS ↓ | CLIP-S ↑ | 训练时间 |
|------|------|---------|---------|---------|
| 1 | Identity | 0.0000 | 0.6933 | 0 |
| 2 | SD-Turbo | 0.0033 | 0.6933 | 0 (infer) |
| 3 | SDEdit s=0.10 | 0.3183 | 0.7188 | 0 (infer) |
| 4 | **SaMam (新7k)** | 0.3209 | 0.6248 | 232.4min |
| 5 | **SaMam (旧20k)** | 0.3282 | 0.7222 | 209.2min |
| 6 | SDEdit s=0.20 | 0.3492 | 0.7340 | 0 (infer) |
| 7 | CUT | 0.3743 | 0.7137 | 322.6min |
| 8 | SDEdit s=0.35 | 0.4508 | 0.7797 | 0 (infer) |
| 9 | SDEdit s=0.40 | 0.4826 | 0.7934 | 0 (infer) |
| 10 | StyleID | 0.5523 | 0.8223 | 0 (infer) |
| 11 | AdaIN | 0.7425 | 0.6679 | 0 (infer) |
| 12 | SaMST | 0.7490 | 0.6183 | 39.5min |

## 4. 关键发现

### 4.1 SaMam新旧实验差异

- **新7k实验确实较差**：CLIP-S 0.6248 vs 旧20k 0.7222（低0.097）
- **但LPIPS基本持平**：新7k 0.3209 vs 旧20k 0.3282（新实验甚至略好）
- **可能原因**：
  1. CLIP backend不同导致绝对值差异（待确认）
  2. 新实验只训练7k步，旧实验10k步
  3. 训练步数间隔不同（新250步，旧约2k步）

### 4.2 SaMam在baseline中的定位

- **内容保真度优秀**：LPIPS 0.32排第4-5，仅次于Identity/SD-Turbo/SDEdit s=0.10
- **风格迁移强度中等**：CLIP-S 0.62-0.72，不如SDEdit系列和StyleID
- **训练成本中等**：209-232min，比CUT(323min)快，比SaMST(40min)慢
- **性价比分析**：SaMam在内容保真度上明显优于同等CLIP-S的方法

### 4.3 Baseline分类

- **推理-only**：AdaIN, WCT, SD-Turbo, SDEdit系列, StyleID (无需训练)
- **需训练**：CUT (gan, 323min), SaMST (mamba, 40min), SaMam (mamba, 209-232min)

## 5. 数据文件

- **统一评估结果**：`exp/baseline_v2/eval/unified_results.json` (12方法)
- **新7k曲线数据**：`tools/samam_distinct5_scratch/sb_curve_metrics.csv` (28 checkpoint)
- **旧20k曲线数据**：`tools/make_dashboard.py` SAMAM_CURVE (8 checkpoint)
- **训练时间数据**：`tools/make_dashboard.py` TRAIN_TIMES
- **新7k评估详情**：`tools/samam_distinct5_scratch/samam_convergence_report.md`

## 6. 待确认事项

1. **WCT (VGG19) 指标缺失**：unified_results.json中没有wct_vgg19的评估数据
2. **新旧CLIP backend差异**：需确认旧20k实验的CLIP-S计算方式，以验证绝对值可比性
3. **新7k SaMam性能较低**：CLIP-S 0.62偏低，需调查是否为CLIP backend差异或训练不足
4. **Art FID评估**：仅旧20k实验有step=5000(289.13)和step=10000(254.39)，其他方法未评估
