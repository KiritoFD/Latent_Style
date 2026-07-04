# 统一基线复现与评估 Spec (v2 深度盘点版)

## Why
论文需要与所有相关工作在统一口径下公平对比。当前存在三个核心问题：
1. **评估口径分裂**：protocol_a_800（远程GPU评估）与run_evaluation.py（本地评估）产出的clip_style绝对值不同（同一SaMAM图片，远程0.70 vs 本地eval_external_baseline.py 0.44），导致所有现有指标无法直接与FC-SB对比
2. **数据集不一致**：部分基线跑在旧5x5风格集（photo/monet/vangogh/cezanne/Hayao）而非distinct5_512（Early_Renaissance/Impressionism/Minimalism/Rococo/Ukiyo_e），两个集不重叠
3. **图片缺失**：大部分基线的生成图片仅在远程，本地无法直接评估；部分方法完全没跑过

## 深度盘点结果

### A. 已有protocol_a_800评估数据（远程GPU跑的，可参考但不可直接用）

| 方法 | clip_style | content_lpips | 图片位置(远程) | 数据集 | 可信度 |
|------|-----------|---------------|--------------|--------|--------|
| Ours (probe4 ep1) | 0.6908 | 0.4184 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| Ours (ep7) | 0.7041 | 0.4587 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| CUT | 0.7588 | 0.4906 | `runs/cut_5x5/` | **旧5x5风格集** | ❌ 数据集不对 |
| SaMST (5style) | 0.7253 | 0.5390 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SaMST strict | 0.7194 | 0.4664 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| S2WAT | 0.7138 | 0.5263 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| StyleID | 0.7777 | 0.5928 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SD-Turbo | 0.7769 | 0.6265 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SDEdit str=0.10 | 0.7023 | 0.3236 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SDEdit str=0.20 | 0.7063 | 0.4087 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SDEdit str=0.35 | 0.6966 | 0.4904 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SDEdit str=0.40 | 0.6968 | 0.5155 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| AdaIN v32k | 0.7130 | 0.6298 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| AdaIN vgg19 | 0.6930 | 0.6870 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| AdaIN bad | 0.6308 | 0.8490 | 远程 | distinct5_512 | ⚠️ 远程评估 |
| SaMAM diag 2250 | 0.581* | 0.322* | 远程 | distinct5_512 | ⚠️ *手工算的，不一致 |

### B. 有图片可拉取的方法（远程已有生成图）

| 方法 | 远程路径 | 图片数 | 数据集确认 |
|------|---------|--------|-----------|
| SaMAM diag 2250/3000 | `results/samam_..._diag/eval_curve/step_*/images` | 750/step | ✅ distinct5_512 |
| SaMAM latent 300/600/1000 | `results/samam_latent_.../eval_bundle_fast_step*/` | 750/step | ✅ distinct5_512 |
| SaMST stepalign40 | `results/samst_.../eval_curve/step_000040/images` | 750 | ✅ distinct5_512 |
| SaMST latent convergence | `results/samst_latent_.../` | 750/step | ✅ distinct5_512 |
| CUT 5x5 | `runs/cut_5x5/` | 2427 | ❌ 旧5x5风格集 |
| SDEdit ×4 | `runs/sdedit_multi/str_*/` | 750/str | ✅ distinct5_512 |
| SD-Turbo | `runs/sdturbo_5x5/` | 750 | ✅ distinct5_512 |
| StyleID | `runs/styleid/` | 750 | ✅ distinct5_512 |
| S2WAT | `runs/s2wat_bs1_safe_e2000_full_eval/` | 750 | ✅ distinct5_512 |
| AdaIN ×3 | 远程训练已完成 | 750/variant | ✅ distinct5_512 |
| Seedream45 API | `baseline_pipeline/results/seedream45_api/` | 750 | ✅ distinct5_512 |

### C. 有代码/模型但需本地推理的方法

| 方法 | 代码位置 | 检查点 | VRAM | 推理难度 | 论文必需度 |
|------|---------|--------|------|---------|-----------|
| SDEdit | Diffusers即用 | HF SD1.5 | ~4GB | 极低 | Tier 1 |
| SD-Turbo | Diffusers即用 | HF SD-Turbo | ~4GB | 极低 | Tier 1 |
| AdaIN | 远程有训练结果 | 已训练 | ~2GB | 低 | Tier 1 |
| CUT | 远程有checkpoint | `runs/cut_5x5/checkpoints/` | ~4GB | 中（需重训到distinct5） | Tier 1 |
| SaMAM | `repos/SaMam/` | `final_model.ckpt` 451MB | ~6GB | 中 | Tier 1 |
| SaMST | `repos/SaMST-main/` | 5个`epoch_100.model` | ~6GB | 中 | Tier 1 |
| S2WAT | `repos/S2WAT-main/` | `checkpoint_bs1_256/` | ~4GB | 中 | Tier 2 |
| StyleID | HF diffusers | IP-Adapter+SD1.5 | ~6GB | 中 | Tier 2 |
| StyTR-2 | 远程有repo | ❌ 无官方权重 | ~4GB | 高 | Tier 3 |
| StarGAN | 远程有repo+ckpt | `epoch_100000` | ~4GB | 中 | Tier 3 |
| CycleGAN | smoke test通过 | 需训练 | ~4GB | 高 | Tier 3 |

### D. 不可用/放弃的方法

| 方法 | 原因 |
|------|------|
| CSGO | 空目录，无代码 |
| StyleShot | 空目录，无代码 |
| SCSA | 空目录，无代码 |
| AesFA | 缺`ckpt/main/main.pth` |
| AesPA-Net | 无权重 |
| ArtBank | 无权重 |
| Flux2 Klein | VRAM不足 |
| CycleGAN-Turbo | 无checkpoint |
| StyleGallery | 代码不完整 |
| CAST | 多次smoke未成功 |

## What Changes

### 核心变更
- 将远程已有基线图片（SaMAM/SaMST/SDEdit/SD-Turbo/StyleID/S2WAT/AdaIN）SCP拉回本地
- 在本地GPU(4070 8GB)上用`run_evaluation.py --reuse_generated`对全部基线图片跑统一评估
- CUT需在distinct5_512上重新训练+推理（旧5x5风格集不可用）
- 将结果写入exp_unified.csv并更新dashboard
- 建立扫描+评估pipeline脚本

### 关键决策
1. **不使用远程GPU**：所有评估必须在本地跑，确保与FC-SB完全相同的评估环境
2. **协议对齐策略**：拉图片→本地run_evaluation.py评估→产出summary.json→入库。不再信任远程评估的绝对值
3. **CUT特殊处理**：旧5x5风格集与distinct5_512不重叠，需重新训练。但这很耗时，优先级排后
4. **SaMAM特殊处理**：2250 step数据是手工从metrics.csv算的，需用原始图片重新评估

## Impact
- Affected code: `docs/exp_unified.csv`, `docs/exp_dashboard_v2.html`, `tools/batch_reeval_baselines.py`, `docs/scan_and_dashboard.py`
- Affected data: 本地需新增~10GB基线图片存储空间
- 关键风险：本地8GB VRAM可能不足以同时加载评估模型和某些推理模型

## ADDED Requirements

### Requirement: 统一本地评估协议
系统 SHALL 使用本地 `src/utils/run_evaluation.py --reuse_generated` 对所有基线方法图片进行评估，产出与FC-SB实验完全一致的summary.json。

#### Scenario: 评估已有图片
- **WHEN** 基线方法的图片已存在于本地磁盘（从远程拉取）
- **THEN** 将图片重命名为`{src_style}__{src_name}__to__{tgt_style}.png`格式，放入eval_dir/images/下，调用run_evaluation.py --reuse_generated

#### Scenario: 评估需重新推理的方法
- **WHEN** 基线方法没有图片或图片来自错误数据集
- **THEN** 先用该方法在distinct5_512上推理，再统一评估

### Requirement: 基线方法分层复现
- **Tier 1（必须，论文核心对比）**: SaMAM, SaMST, SDEdit, SD-Turbo, AdaIN — 都有图片可拉取
- **Tier 2（重要，补充对比）**: S2WAT, StyleID, CUT — CUT需重训
- **Tier 3（可选）**: StyTR-2, StarGAN, CycleGAN — 有代码但缺权重或需训练
- **放弃**: CSGO, StyleShot, AesFA, Flux2 Klein等

### Requirement: 数据集对齐
所有方法 SHALL 使用 `distinct5_512` 数据集（wikiarts 5种风格各1000张：Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e），生成5×5=25个风格对的750张图片（每对30张）。

### Requirement: 结果入库
所有评估结果 SHALL 写入 `docs/exp_unified.csv`，group=RW/{method}, dataset=distinct5_512, eval_type=unified_reeval，并更新dashboard。

### Requirement: Sanity Check
对已有远程评估结果的方法，本地重新评估后 SHALL 比较clip_style差异。如差异<5%，确认评估一致；如差异>5%，调查原因（参考特征集差异、图片命名等）。

## MODIFIED Requirements

### Requirement: 评估管线
从"在远程跑batch_reeval_baselines.py"改为"拉图片到本地→本地run_evaluation.py评估"。废弃eval_external_baseline.py（使用不同CLIP模型导致结果不一致）。

## REMOVED Requirements

### Requirement: 远程GPU评估
**Reason**: 用户明确禁止使用远程GPU，评估必须在本地进行确保口径一致
**Migration**: 所有数据拉到本地再评估

### Requirement: CSGO/StyleShot/StyleGallery复现
**Reason**: 远程仓库目录为空，无可用代码
**Migration**: 论文中注明"code unavailable"

### Requirement: Flux2 Klein复现
**Reason**: 本地8GB VRAM不足
**Migration**: 论文中注明"requires >8GB VRAM"
