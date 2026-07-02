# FC-SB Baseline 数据完整性核查报告

**生成时间**: 2026-07-03
**核查范围**: 12 个 baseline (论文 AAAI 2027 对比用) 的评估协议一致性 + 训练类 baseline 收敛证据
**数据真相源**: `exp/exp_baselines/baseline_v2/eval/unified_results.json`

---

## 0. 核查结论速览

| # | Baseline | 类别 | CLIP-S | LPIPS | n_pairs | 收敛证据 |
|---|----------|------|--------|-------|---------|---------|
| 1 | Identity | baseline (推理) | 0.6933 | 0.0000 | 750 | N/A (单点) |
| 2 | AdaIN | classical-inf | 0.6679 | 0.7425 | 750 | N/A (推理) |
| 3 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | 750 | N/A (推理) |
| 4 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 750 | N/A (推理) |
| 5 | SDEdit s=0.35 | diffusion-sweep | 0.7797 | 0.4508 | 750 | N/A (推理) |
| 6 | SDEdit s=0.40 | diffusion-sweep | 0.7934 | 0.4826 | 750 | N/A (推理) |
| 7 | StyleID | diffusion-inf | 0.8223 | 0.5523 | 750 | N/A (推理) |
| 8 | **CUT** | gan-train | 0.7137 | 0.3743 | 745 | ⚠️ **缺失** (见 §3.1) |
| 9 | **SaMST** | mamba-train | 0.6183 | 0.7490 | 750 | ✅ e5/e15 plateau (见 §3.2) |
| 10 | **SaMam** | mamba-train | 0.5816 | 0.2434 | 750 | ✅ 81 ckpt 完整曲线 (见 §3.3) |
| 11 | Seedream 4.5 | commercial-API | 0.7198 | 0.4767 | 750 | N/A (API) |

> **注**: 第 12 项 FC-SB T11 为本项目自有方法，不在 baseline 核查范围内。

---

## 1. 评估协议一致性确认

**统一协议** (来自 `docs/72/07_related_works.md` §"CLIP Backend 对齐说明"):

| 项 | 配置 |
|---|---|
| CLIP backend | HF transformers (`openai/clip-vit-base-patch32`, ViT-B/32) |
| LPIPS backbone | Alex |
| n_pairs | 750 (150 src × 5 styles, 含 identity 对) — CUT 为 745 |
| 数据集 | `wikiart_distinct5_samam_512_classview/test` (distinct5_512, 5 styles: Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e) |
| Identity 基线 | CLIP-S=0.6933, LPIPS=0.0 |
| 评估脚本 | FC-SB 统一管线 `src/utils/run_evaluation.py` (SaMam 例外: `tools/samam_distinct5_scratch/eval_samam_metrics_phase2.py`，与统一管线协议一致) |

**协议一致性核对**: 11 个评估类 baseline (含 SaMam) 均使用 HF transformers CLIP + LPIPS Alex + 750 pairs + distinct5_512。SaMam 虽然用自有评估脚本，但 CLIP/LPIPS backend、n_pairs、数据集完全对齐，可横向比较（详见 `docs/exp/samam_data_integrity_audit.md` §2.2）。

---

## 2. unified_results.json 与评估文件值核对

**数据源**: `exp/exp_baselines/baseline_v2/eval/unified_results.json`

| Baseline | unified_results.json | 评估文件实际值 | 一致性 |
|----------|---------------------|----------------|--------|
| identity | 0.6933 / 0.0000 / 750 | `baseline_reeval/identity/summary.json`: 0.6933 / 0.0000 / 750 | ✅ |
| adain | 0.6679 / 0.7425 / 750 | `baseline_reeval/adain_v32k/summary.json` | ✅ |
| wct_vgg19 | 0.7063 / 0.6348 / 750 | `baseline_v2/eval/wct_vgg19_summary.json` (clip_backend=hf, lpips_net=alex) | ✅ |
| sdturbo | 0.6933 / 0.0033 / 750 | `baseline_reeval/sdturbo/summary.json` | ✅ |
| sdedit_str0.35 | 0.7797 / 0.4508 / 750 | `baseline_reeval/sdedit_035/summary.json` | ✅ |
| sdedit_str0.40 | 0.7934 / 0.4826 / 750 | `baseline_reeval/sdedit_040/summary.json` | ✅ |
| styleid | 0.8223 / 0.5523 / 750 | `exp_baselines/styleid/` (远程 `I:\Github\Latent_Style\exp_baselines\styleid\`, 33M images) | ✅ |
| cut | 0.7137 / 0.3743 / 745 | `exp_baselines/cut/` (远程, 7.0M images) | ✅ |
| samst | 0.6183 / 0.7490 / 750 | `baseline_reeval/samst_stepalign40/summary.json` (clip_backend=hf, lpips_net=alex, n=750) | ✅ |
| samam | 0.5816 / 0.2434 / 750 | `tools/samam_distinct5_scratch/curve_metrics_hf.csv` step 20000 行 (远程 I 盘) | ✅ |
| seedream | 0.7198 / 0.4767 / 750 | `exp_baselines/seedream45_api/` (远程, 1006M images) | ✅ |

**已知不一致项**: `baseline_v2/baseline_summary_table.csv` 中 SaMam 仍为旧错误值 0.7222 / 0.3282 (256/wikiart5 历史值, 已废弃)。**以 `unified_results.json` 为准** (已对齐 v5 真实值 0.5816 / 0.2434, 见 `docs/exp/samam_data_integrity_audit.md` §6)。

---

## 3. 训练类 baseline 收敛证据

### 3.1 CUT (gan-train) — ⚠️ 收敛证据缺失

| 项 | 值 | 来源 |
|---|---|---|
| 训练时长 | 322.6 min (5.38 hr) | 用户记忆, 记于 `baseline_v2/baseline_conclusions.md` |
| 训练配置 | 5 styles × 4 epochs (2+2 decay) | `docs/72/07_related_works.md` §三 |
| 训练日志 | **缺失** | `docs/exp/remote_experiments.md` §4 第 1 条: "CUT 训练时长未记录, 用户记忆 322.6min, 但远程目录无 train.log, 仅 summary.json(eval 指标)" |
| Checkpoint 曲线 | **缺失** | 仅最终评估点 (n=745, 5 张图缺失) |
| 最终采用 ckpt | 未明确记录 | — |
| 评估协议 | HF CLIP + LPIPS Alex, 745 pairs, distinct5 | 与统一协议一致 (n_pairs 偏差 5) |

**判定**: ⚠️ **CUT 复现跑到收敛的证据不完整**。仅有最终评估点和用户记忆的训练时长，无 train.log，无多 checkpoint 曲线。建议补充训练日志或在论文中明确标注 "CUT 训练沿用 `Related_Works/runs/cut_5x5/` 现有 checkpoint"。

### 3.2 SaMST (mamba-train) — ✅ 收敛证据完整

**收敛测试文档**: `docs/archive/experiments/samst_distinct5_converged_notice_20260604.md`

| 项 | 值 |
|---|---|
| 训练配置 | b1_e5 (1:55:58.5) vs b2_e15 (5:47:15.4) |
| e5 (中点) Full metrics | CLIP-S=0.7276, LPIPS=0.6271 |
| e15 (终点) Full metrics | CLIP-S=0.7247, LPIPS=0.6255 |
| e5 → e15 ΔCLIP-S | -0.0029 (平台) |
| e5 → e15 ΔLPIPS | -0.0016 (平台) |
| 最终采用 ckpt | e15 (targetwise ArtFID 更低, 更安全) |
| 训练总时长 | ~3.03 hr (含 7 个 SaMST 训练实验, `docs/exp/remote_experiments.md` §1.2) |
| 评估目录 | `baseline_reeval/samst_stepalign40/` (clip_backend=hf, lpips_net=alex, n=750) |

**判定**: ✅ **SaMST 复现跑到收敛**。e5/e15 两点 CLIP-S/LPIPS 变化 < 0.003，曲线已 plateau，e15 作为 manuscript endpoint。

### 3.3 SaMam (mamba-train) — ✅ 81 checkpoint 完整曲线

**详细审计文档**: [`docs/exp/samam_data_integrity_audit.md`](../exp/samam_data_integrity_audit.md) (本节仅引用, 不重复)

| 项 | 值 |
|---|---|
| 训练步数 | 20000 (batch=1, 512×512, 32-true, distinct5) |
| 训练时长 | 13948.61s = 3.87 hr (~436 min) |
| Checkpoints | 81 个 (step 250-20000 + last, 250 步间隔) |
| 评估协议 | HF transformers CLIP ViT-B/32 + LPIPS Alex, 750 pairs/ckpt, distinct5 |
| 评估脚本 | `tools/samam_distinct5_scratch/eval_samam_metrics_phase2.py` |
| 评估数据源 | `curve_eval_hf_750_batched/curve_metrics.csv` (81 行, 远程 I 盘) |
| CLIP-S 收敛点 | step 3000 后 delta < 0.01, **已收敛** |
| CLIP-S 峰值 | step 6500, CLIP-S=0.5925 |
| 最终采用值 | step 20000, CLIP-S=0.5816, LPIPS=0.2434 |
| 历史废弃值 | ~~0.7222~~ (256/wikiart5), ~~0.7175/0.2423~~ (v4 编造值, 不存在于任何评估文件) |

**判定**: ✅ **SaMam 复现跑到收敛**。81 个 checkpoint 完整曲线, step 3000 后已平台, step 20000 为最终采用。详见 `docs/exp/samam_data_integrity_audit.md`。

---

## 4. 推理类 baseline 评估路径

| Baseline | 评估文件路径 | 备注 |
|----------|-------------|------|
| Identity | `baseline_reeval/identity/summary.json` + `metrics.csv` | 单点 (copy) |
| AdaIN | `baseline_reeval/adain_v32k/summary.json` | 32k checkpoint variant |
| WCT (VGG19) | `baseline_v2/eval/wct_vgg19_summary.json` | VGG-19 ImageNet encoder + AdaIN-trained VGG-19 decoder, adain_post=True |
| SD-Turbo | `baseline_reeval/sdturbo/summary.json` | 1-step inference |
| SDEdit s=0.35 | `baseline_reeval/sdedit_035/summary.json` | strength=0.35 |
| SDEdit s=0.40 | `baseline_reeval/sdedit_040/summary.json` | strength=0.40 |
| StyleID | `exp_baselines/styleid/` (远程 I 盘, 33M images) | diffusion-inf, single-point |
| Seedream 4.5 | `exp_baselines/seedream45_api/` (远程 I 盘, 1006M images) | API 调用, `seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750` |

> 注: SDEdit s=0.10/0.20 已按用户要求不再用于论文，但仍保留在 `baseline_reeval/sdedit_010/`, `sdedit_020/`。

---

## 5. 数据完整性结论

### 5.1 整体判定

- **协议一致性**: ✅ 11 个评估类 baseline 全部使用 HF transformers CLIP ViT-B/32 + LPIPS Alex + 750 pairs (CUT=745) + distinct5_512，可横向比较
- **unified_results.json 真实性**: ✅ 11 个 baseline 的值与各评估文件实际值一致 (SaMam 已对齐 v5 真实值)
- **训练类收敛证据**:
  - SaMST ✅ e5/e15 plateau 测试完成
  - SaMam ✅ 81 checkpoint 完整曲线
  - CUT ⚠️ **缺失 train.log 与多 checkpoint 曲线**, 仅用户记忆训练时长

### 5.2 待补充项

1. **CUT 训练日志**: 若论文需严格 "复现跑到收敛" 证据, 建议补充:
   - 远程 `Related_Works/runs/cut_5x5/` 训练日志归档
   - 或明确在论文中标注 "CUT 沿用 `cut_5x5` 现有 checkpoint, 训练时长 322.6 min (用户记录)"
2. **baseline_summary_table.csv 中 SaMam 旧值**: 该 CSV 仍为 0.7222, 与 `unified_results.json` 0.5816 不一致。**以 `unified_results.json` 为准**, CSV 待后续清理 (本报告不修改现有文件)。

### 5.3 数据真相源优先级

1. `exp/exp_baselines/baseline_v2/eval/unified_results.json` (最高优先级, v5 已修正)
2. `docs/exp/samam_data_integrity_audit.md` (SaMam 81 ckpt 详细审计)
3. `docs/archive/experiments/samst_distinct5_converged_notice_20260604.md` (SaMST 收敛判定)
4. `docs/72/07_related_works.md` (12 baseline 完整指标表)
5. `exp/exp_baselines/baseline_v2/baseline_summary_table.csv` (⚠️ SaMam 旧值未更新, 历史保留)
6. `exp/exp_baselines/baseline_v2/baseline_conclusions.md` (⚠️ SaMam 旧值未更新, 历史保留)

---

## 6. 相关文档索引

| 文档 | 内容 |
|------|------|
| [`docs/exp/samam_data_integrity_audit.md`](../exp/samam_data_integrity_audit.md) | SaMam 81 checkpoint 完整曲线 + 编造值调查 |
| [`docs/archive/experiments/samst_distinct5_converged_notice_20260604.md`](../archive/experiments/samst_distinct5_converged_notice_20260604.md) | SaMST e5/e15 plateau 收敛判定 |
| [`docs/exp/remote_experiments.md`](../exp/remote_experiments.md) | 远程 I 盘 baseline 评估实验清单 (含 CUT 训练时长缺失说明 §4) |
| [`docs/72/07_related_works.md`](../72/07_related_works.md) | 12 baseline 完整指标表 + CLIP backend 对齐说明 |
| [`docs/archive/Related/baseline_methods_catalog.md`](../archive/Related/baseline_methods_catalog.md) | 各 baseline 论文/代码/训练协议目录 |
| [`exp/exp_baselines/baseline_v2/eval/unified_results.json`](../../exp/exp_baselines/baseline_v2/eval/unified_results.json) | 12 baseline 统一评估结果 (数据真相源) |
| [`exp/exp_baselines/baseline_v2/baseline_conclusions.md`](../../exp/exp_baselines/baseline_v2/baseline_conclusions.md) | Baseline 复现结论 (注: SaMam 旧值未更新) |
