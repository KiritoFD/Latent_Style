# 远程 Baseline 全量复现 Spec

## Why
论文需要所有 baseline 在 distinct5_512 数据集上用统一评估协议（LPIPS-VGG + CLIP-ViT-L/14）的公平对比数据。当前仅有 SaMAM/SaMST 的远程拉取图片评估可信，其余方法要么未跑、要么参数/训练过程不对。远程 3060 12GB VRAM 充足且有 WSL（mamba-ssm 可用），应全部在远程完成训练+推理+评估。

## What Changes
- 在远程服务器 `ssh -p 2222 administrator@100.115.18.62` 上完成所有 baseline 的训练+推理+评估
- 统一输出目录：远程 `I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\`
- 统一评估协议：`run_evaluation.py`（LPIPS-VGG, CLIP-ViT-L/14, distinct5_512 test, clip_style_idt_baseline=0.6399）
- 所有生成图命名格式：`{src_style}__{src_stem}__to__{tgt_style}.png`
- 本地同步结果并更新文档和 Dashboard

## Impact
- Affected code: `exp/baseline_v2/`（新目录）, `docs/Related/baseline_methods_catalog.md`
- Affected systems: 远程 GPU 3060 12GB, Windows + WSL

## 远程环境

| 项目 | 值 |
|------|------|
| GPU | NVIDIA 3060 12GB (12288 MiB) |
| OS | Windows + WSL2 |
| Python | 3.12.10 (Windows侧) |
| 数据集 test | `I:\wikiart_distinct5_samam_512_classview\test` |
| 数据集 train | `I:\wikiart_distinct5_samam_512_classview\train` (5风格, 每风格1000张) |
| Repos | `I:\GitHub\Latent_Style\Related_Works\repos\` |
| 评估脚本 | `I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py` |

## Baseline 方法清单

### Tier 1：零样本推理（无需训练，只需推理+评估）

| 方法 | VRAM | 模型来源 | 预计产出 |
|------|------|---------|---------|
| SDEdit str=0.10 | ~4GB | runwayml/stable-diffusion-v1-5 | 750张 |
| SDEdit str=0.20 | ~4GB | 同上 | 750张 |
| SDEdit str=0.35 | ~4GB | 同上 | 750张 |
| SDEdit str=0.40 | ~4GB | 同上 | 750张 |
| SD-Turbo | ~4GB | stabilityai/sd-turbo | 750张 |
| StyleID | ~4GB | 零样本 SD1.5 变体 | 750张 |
| AdaIN | ~1GB | pytorch-AdaIN 官方 decoder.pth | 750张 |

### Tier 2：需训练（在 distinct5_512 上训练后推理）

| 方法 | 预计训练时间 | VRAM需求 | 训练方式 |
|------|------------|---------|---------|
| SaMAM | ~6-12h | 8-10GB (bs=2, fp16) | WSL mamba-ssm, 全风格联合训练 |
| SaMST | ~4-8h | 4-6GB (bs=1-2) | train2 模式, 预计算 Gram |
| S2WAT | ~8-16h | 6-8GB (bs=1, bf16, gc) | 全风格联合训练 |
| CUT | ~40-80h (20对) | 4-6GB (bs=1) | 逐风格对训练, 5个风格=20个模型 |

### Tier 3：低优先级

| 方法 | 状态 |
|------|------|
| StyTR-2 | 需下载预训练权重，推理即可 |
| CycleGAN-Turbo | 代码有但未验证 |

## 评估协议

**统一命令**:
```bash
python I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py ^
  <eval_dir> --reuse_generated --save_generated_images ^
  --style_subdirs Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e ^
  --test_dir I:\wikiart_distinct5_samam_512_classview\test ^
  --eval_only_lpips_clip_style ^
  --clip_style_idt_baseline 0.6399
```

**注意**: 远程 `run_evaluation.py` 硬编码 LPIPS-VGG，与本地 IDT 标定已验证一致。

## 目录结构

```
I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\
├── images\              # 各方法生成图
│   ├── identity\        # Identity 基线 (已验证)
│   ├── sdedit_str010\
│   ├── sdedit_str020\
│   ├── sdedit_str035\
│   ├── sdedit_str040\
│   ├── sdturbo\
│   ├── styleid\
│   ├── adain_v32k\
│   ├── samam\
│   ├── samst\
│   ├── s2wat\
│   └── cut\
├── eval\                # 评估结果 summary.json
│   ├── identity\
│   ├── sdedit_str010\
│   ...
│   └── unified_results.json
└── train_logs\          # 训练日志
    ├── samam\
    ├── samst\
    ├── s2wat\
    └── cut\
```

## ADDED Requirements

### Requirement: 统一远程推理
所有零样本方法（SDEdit×4, SD-Turbo, StyleID, AdaIN）必须在远程服务器上推理，生成 750 张图，使用固定的 seed=42 和统一 prompt 模板。

#### Scenario: SDEdit 推理成功
- **WHEN** 对 distinct5_512 的 150 张测试图执行 SDEdit img2img（strength=0.10/0.20/0.35/0.40）
- **THEN** 每个_strength_产出 750 张 PNG，命名格式正确，文件完整

#### Scenario: SD-Turbo 推理成功
- **WHEN** 使用 stabilityai/sd-turbo 进行 1 步 img2img 推理
- **THEN** 产出 750 张 PNG，num_inference_steps=1，guidance_scale=1.0

### Requirement: 统一远程训练
SaMAM/SaMST/S2WAT/CUT 必须在 distinct5_512 数据集上从零训练，使用各自的默认超参（仅调整 batch_size/precision 适配显存）。

#### Scenario: SaMAM 训练成功
- **WHEN** 在 WSL 下用 mamba-ssm 后端训练 SaMAM
- **THEN** 模型在 distinct5_512 上训练完成，可在 test 集上推理产出 750 张图

### Requirement: 统一评估+汇总
所有方法的生成图必须用同一个 `run_evaluation.py`（LPIPS-VGG）评估，产出 `summary.json`，最终汇总到 `unified_results.json` 和 Dashboard。

#### Scenario: 全部方法评估完成
- **WHEN** 所有 11+ 个方法的 summary.json 生成完毕
- **THEN** unified_results.json 包含所有方法的 clip_style/content_lpips/clip_s_delta_idt/clip_t，Dashboard 可视化正确

### Requirement: 本地同步
远程评估完成后，结果必须同步到本地文档 `docs/Related/baseline_methods_catalog.md`。

## REMOVED Requirements
### Requirement: 本地复现
**Reason**: 本地 RTX 4070 8GB VRAM 不足，且缺少 mamba-ssm
**Migration**: 全部迁移到远程 3060 12GB
