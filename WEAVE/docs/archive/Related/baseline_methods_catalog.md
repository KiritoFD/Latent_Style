# Baseline Methods Catalog

本文档记录 AAAI 2027 论文实验涉及的所有 baseline 方法，包括代码位置、训练/推理方式、数据集要求和已有结果。

> 最后更新: 2026-06-30

---

## 目录

1. [数据集](#1-数据集)
2. [评估协议](#2-评估协议)
3. [已有评估结果汇总](#3-已有评估结果汇总)
4. [Baseline 方法详情](#4-baseline-方法详情)
   - [Identity (恒等映射)](#41-identity)
   - [AdaIN](#42-adain)
   - [StyTR-2](#43-stytr-2)
   - [SaMAM](#44-samam)
   - [SaMST](#45-samst)
   - [S2WAT](#46-s2wat)
   - [StyleID](#47-styleid)
   - [CUT](#48-cut)
   - [SDEdit](#49-sdedit)
   - [SD-Turbo](#410-sd-turbo)
   - [CycleGAN / CycleGAN-Turbo](#411-cyclegan--cyclegan-turbo)
   - [CAST](#412-cast)
5. [Baseline Pipeline 自动化](#5-baseline-pipeline-自动化)
6. [本地复现状态](#6-本地复现状态)

---

## 1. 数据集

### 1.1 distinct5_512 (当前主数据集)

- **路径**: `G:\GitHub\Latent_Style\Dataset\distinct5_512\`
- **5个风格**: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e
- **train/**: `G:\GitHub\Latent_Style\Dataset\distinct5_512\train\`，每风格子目录 **1000** 张训练图片
- **test/**: `G:\GitHub\Latent_Style\Dataset\distinct5_512\test\`，每风格子目录 **30** 张测试图片，命名格式 `{Style}__{artist}_{title}.jpg`
- **test (eval副本)**: `G:\GitHub\Latent_Style\Dataset\eval\distinct5_512\test\`（同一套图的副本）
- **test_manifest.json**: `G:\GitHub\Latent_Style\Dataset\distinct5_512\test_manifest.json`
- **远程test目录**: `I:\wikiart_distinct5_samam_512_classview\test`（远程训练评估专用，与本地 distinct5_512/test 是同一套测试图）
- **总训练图**: 5 × 1000 = 5000 张
- **总测试图**: 5 × 30 = 150 张
- **评估图像对**: 5 src × 5 tgt × 30 img = **750** 对（含 identity 对，即 src_style == tgt_style）
- **远程训练**: 远程 SaMAM/SaMST 实验均在此数据集上训练和评估
- **本地状态**: ✅ train 和 test 目录均已有数据

### 1.2 wikiart512_5style (不同风格集)

- **路径**: `G:\GitHub\Latent_Style\Dataset\wikiart512_5style\`
- **5个风格**: **Realism, Impressionism, Post_Impressionism, Expressionism, Symbolism**（与 distinct5_512 不同！）
- **每风格 3600 张**训练图片
- **test_manifest.json**: `G:\GitHub\Latent_Style\Dataset\wikiart512_5style\test_manifest.json`
- **注意**: 此数据集与 distinct5_512 的风格集不同，不能混用

### 1.3 protocol_a_800 (旧评估协议)

- **风格**: photo, monet, vangogh, cezanne, Hayao
- **与 distinct5_512 完全不同**: 5个风格不重叠
- **来源**: 早期 baseline 复现（2026-05-11），来自 `baseline_metrics_unified.csv`
- **注意**: 此协议下所有指标与 distinct5_512 不可直接比较
- **区别**:
  - protocol_a_800 用的 5 风格是 photo/monet/vangogh/cezanne/Hayao（无 Early_Renaissance 等）
  - distinct5_512 用的 5 风格是 Early_Renaissance/Impressionism/Minimalism/Rococo/Ukiyo_e
  - 两者是**完全不同的数据集划分**，风格名不重叠

---

## 2. 评估协议

### 2.1 评估引擎

所有评估统一使用 `src/utils/run_evaluation.py`，通过 `tools/batch_reeval_baselines.py` 桥接调用。

### 2.2 统一评估命令

```bash
python tools/batch_reeval_baselines.py \
  --method <method_key> \
  --test_dir "G:\GitHub\Latent_Style\Dataset\distinct5_512\test" \
  --output_root "G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_reeval"
```

底层调用等价于：

```bash
python src/utils/run_evaluation.py <eval_dir> \
  --reuse_generated \
  --save_generated_images \
  --style_subdirs=Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
  --test_dir=<test_dir> \
  --eval_only_lpips_clip_style \
  --eval_lpips_net alex \
  --clip_style_idt_baseline 0.6399
```

### 2.3 核心指标定义

#### 2.3.1 CLIP-Style (clip_style)

```
clip_style = cos( CLIP_image(gen), CLIP_image(target_style_proto) )
```

- **target_style_proto**: 目标风格所有测试图的 CLIP 图像特征取均值后 L2 归一化
  - 具体地：对目标风格下 30 张测试图分别提取 CLIP 特征 → stack 为 [30, D] → 逐向量归一化 → 取均值 → 再归一化
  - 缓存于 `exp/*/ref_feats_*.pt`
- **CLIP 模型**: HuggingFace `openai/clip-vit-large-patch14` (CLIP-ViT-L/14)
- **方向**: 越高越好

#### 2.3.2 Content LPIPS (content_lpips)

```
content_lpips = LPIPS(gen, src)
```

- **LPIPS 网络选择**（⚠️ 重要差异）:
  - **远程评估 (run_evaluation.py)**: 硬编码使用 **LPIPS-VGG** (`lpips.LPIPS(net='vgg')`)
  - **本地 baseline_reeval**: 之前改为 LPIPS-Alex (`--eval_lpips_net alex`)
  - **ArtFID 计算**: 使用 LPIPS-Alex（论文标准）
- **数值差异极大**: 同一对图片 VGG LPIPS 通常为 Alex 的 2~3 倍
  - 例如：SaMAM-diag-3000 在 distinct5_512 上：Alex LPIPS=0.2423 vs VGG LPIPS≈0.59
  - 远程 FC-SB clean_base epoch10: VGG LPIPS=0.585
- **方向**: 越低越好（0 = 完全一致）
- **必须注意**: 对比不同来源的 LPIPS 值时，必须确认使用的是同一个网络！

#### 2.3.3 CLIP-Style Delta IDT (clip_s_delta_idt)

```
clip_s_delta_idt = clip_style - clip_style_idt_baseline
```

- **clip_style_idt_baseline = 0.6399**
  - 这是 distinct5_512 数据集上 transfer-only（排除 identity 对）的 IDT 基线
  - IDT 基线含义：使用 FC-SB 模型自身在同风格对（src_style == tgt_style）上的 clip_style 均值
  - 在 `comparison_20260602/comparison_report.md` 中确认: "transfer-only `clip_style=0.6399`"
- **方向**: 越高越好，衡量超越"不做迁移"的增量

#### 2.3.4 CLIP-Content (clip_content)

```
clip_content = cos( CLIP_image(gen), CLIP_image(src) )
```

- 仅在 `--no-eval_only_lpips_clip_style` 模式下计算
- 衡量生成图与源图在 CLIP 语义空间中的相似度
- **方向**: 越高越好

#### 2.3.5 CLIP-T (clip_t)

```
clip_t = cos( CLIP_image(gen), CLIP_text(style_name) )
```

- 将风格名（如 "Early Renaissance"）编码为文本 CLIP 特征
- 衡量生成图与风格名文本描述的亲和度
- **方向**: 越高越好

#### 2.3.6 CLIP-Dir (clip_dir)

```
clip_dir = cos( CLIP(gen) - CLIP(src), CLIP(target_style_proto) - CLIP(src) )
```

- 衡量编辑方向是否与目标风格方向一致
- 仅在 `--no-eval_only_lpips_clip_style` 模式下计算

### 2.4 评估参数详情

| 参数 | 值 | 说明 |
|------|------|------|
| `--style_subdirs` | Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e | 5个风格目录名 |
| `--test_dir` | `Dataset/distinct5_512/test` | 参考图根目录 |
| `--eval_only_lpips_clip_style` | True | 快速模式，仅计算 LPIPS + CLIP-Style + CLIP-T |
| `--eval_lpips_net` | `alex` | LPIPS 使用 AlexNet（不是 VGG） |
| `--clip_style_idt_baseline` | `0.6399` | IDT 基线值（distinct5_512 transfer-only） |
| `--clip_backend` | `hf` | 使用 HuggingFace CLIP |
| `--clip_model_name` | `openai/clip-vit-large-patch14` | CLIP-ViT-L/14 |
| `--reuse_generated` | True | 使用已有生成图（不重新推理） |
| `--save_generated_images` | True | 保存生成图副本 |

### 2.5 图片命名规范

baseline 生成图必须遵循以下命名格式，才能被 `run_evaluation.py` 的 `_parse_generated_name` 正确解析：

```
{src_style}__{src_stem}__to__{tgt_style}.png
```

其中 `src_stem` 必须与 test 目录下源图的 stem 一致（含风格前缀）：
- 源图: `test/Early_Renaissance/Early_Renaissance__andrea-mantegna_something.jpg`
- 生成图: `Early_Renaissance__Early_Renaissance__andrea-mantegna_something__to__Impressionism.png`

### 2.6 评估流程

1. **准备阶段** (`prepare_images`): 将 baseline 原始图片复制到 `eval_dir/images/`，重命名为标准格式
2. **参考特征缓存**: 对每个风格的测试图提取 CLIP 特征，缓存为 `ref_feats_*.pt`
3. **计算指标**: 遍历所有 750 张生成图，计算 LPIPS、CLIP-Style、CLIP-T
4. **汇总**: 写入 `summary.json`，包含 per-pair 指标和 all_pairs_overview

### 2.7 Identity 基线参考值（IDT 标定）

**已验证: 远程与本地评估完全一致。**

| 指标 | 远程结果 | 本地结果 | 匹配 |
|------|---------|---------|------|
| clip_style (all) | 0.6933 | 0.6933 | ✅ |
| content_lpips | 0.0000 | 0.0000 | ✅ |
| Δ_idt | +0.0534 | +0.0534 | ✅ |
| clip_t | 0.2135 | 0.2135 | ✅ |
| identity CLIP-S | 0.8534 | 0.8533 | ✅ |
| transfer CLIP-S | 0.6533 | 0.6532 | ✅ |

**评估命令（远程）**:
```bash
python src/utils/run_evaluation.py exp/baseline_reeval/identity_baseline \
  --reuse_generated --save_generated_images \
  --style_subdirs Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
  --test_dir I:/wikiart_distinct5_samam_512_classview/test \
  --eval_only_lpips_clip_style \
  --clip_style_idt_baseline 0.6399
```

**评估命令（本地，LPIPS-VGG对齐远程）**:
```bash
python src/utils/run_evaluation.py exp/baseline_reeval/identity_vgg \
  --reuse_generated --save_generated_images \
  --style_subdirs Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
  --test_dir "G:\GitHub\Latent_Style\Dataset\distinct5_512\test" \
  --eval_only_lpips_clip_style \
  --eval_lpips_net vgg \
  --clip_style_idt_baseline 0.6399
```

**评估结果文件**:
- 远程: `I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_reeval\identity_baseline\summary.json`
- 本地: `G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_reeval\identity_vgg\summary.json`

Identity 基线（源图直接复制为目标图，不做任何风格迁移）在 distinct5_512 上的完整评估结果：

#### 全局指标

| 指标 | all_pairs (750) | identity_reconstruction (150) | style_transfer (600) |
|------|:---:|:---:|:---:|
| CLIP-Style | **0.6933** | 0.8533 | 0.6532 |
| Content LPIPS | 0.0000 | 0.0000 | 0.0000 |
| Δ_idt | +0.0534 | +0.2134 | +0.0133 |
| CLIP-T | 0.2135 | 0.2695 | 0.1995 |

- **all_pairs**: 5×5×30 = 750 对（含 identity 对 src_style == tgt_style）
- **identity_reconstruction**: 5×30 = 150 对（src_style == tgt_style，即同风格对）
- **style_transfer**: 5×4×30 = 600 对（src_style ≠ tgt_style，即跨风格对）

#### 逐对 CLIP-Style 矩阵

| src → tgt | Early_Ren. | Impressionism | Minimalism | Rococo | Ukiyo_e |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Early_Ren.** | 0.845 | 0.659 | 0.614 | 0.711 | 0.649 |
| **Impressionism** | 0.652 | 0.836 | 0.640 | 0.677 | 0.646 |
| **Minimalism** | 0.635 | 0.669 | 0.874 | 0.613 | 0.653 |
| **Rococo** | 0.684 | 0.659 | 0.571 | 0.814 | 0.607 |
| **Ukiyo_e** | 0.690 | 0.694 | 0.672 | 0.671 | 0.898 |

#### IDT Baseline 标定值

```
clip_style_idt_baseline = 0.6399
```

- 这是 `--clip_style_idt_baseline` 参数的固定值
- 来源: distinct5_512 数据集上 transfer-only（排除 identity 对）的 IDT 基线
- 当前 identity 评估测得的 style_transfer CLIP-Style = 0.6532，略高于此固定值 0.6399

#### 关键解读

- **不做迁移时源图本身 CLIP-Style = 0.6933**（all_pairs），任何有效风格迁移方法应显著高于此
- **跨风格对的"自然"CLIP-Style = 0.6532**，Δ_idt 仅 +0.0133，说明跨风格源图与目标风格也有一定基线相似度
- **同风格对的 CLIP-Style = 0.8533**，这是上限——同一风格内图像的 CLIP 风格一致性

---

## 3. 已有评估结果汇总

### 3.1 distinct5_512 数据集（当前主数据集，5×1000 train / 5×30 test）

来源: `exp/baseline_reeval/unified_eval_results.json`

| 方法 | CLIP-Style | Content LPIPS | 1-LPIPS | Δ_idt | 图片来源 | 可信度 |
|------|-----------|---------------|---------|-------|---------|--------|
| **Identity** | 0.6933 | 0.0000 | 1.0000 | +0.0534 | 原图复制 | ✅ |
| SaMAM-diag-3000 | 0.7175 | 0.2423 | 0.7577 | +0.0776 | 远程SCP拉取 | ✅ |
| SaMAM-diag-2250 | 0.7074 | 0.2374 | 0.7626 | +0.0675 | 远程SCP拉取 | ✅ |
| SaMST-40 | 0.6795 | 0.7212 | 0.2788 | +0.0396 | 远程SCP拉取 | ✅ |

### 3.2 远程 FC-SB 最新实验 (clean_base, LPIPS-VGG)

来源: 远程 `I:\GitHub\Latent_Style\SchrodingerBridge\exp\clean_base\full_eval\clip_lpips_curve.csv`

| 实验 | Epoch | CLIP-Style | Δ_idt | Content LPIPS (VGG) | CLIP-T | 备注 |
|------|-------|-----------|-------|---------------------|--------|------|
| clean_base | 8 | 0.7148 | +0.0748 | 0.5566 | 0.238 | all_pairs |
| clean_base | 9 | 0.7075 | +0.0676 | 0.5817 | 0.238 | all_pairs |
| clean_base | 10 | 0.7073 | +0.0674 | 0.5852 | 0.237 | all_pairs |
| clean_base | 8 (transfer) | 0.7046 | +0.0647 | 0.5612 | 0.234 | transfer-only |
| clean_base | 9 (transfer) | 0.6992 | +0.0592 | 0.5857 | 0.235 | transfer-only |
| clean_base | 10 (transfer) | 0.6999 | +0.0600 | 0.5896 | 0.234 | transfer-only |

> ⚠️ **LPIPS-VGG**: 远程评估使用 LPIPS-VGG，与本地 baseline_reeval 的 LPIPS-Alex 不可直接比较。
> VGG LPIPS 通常为 Alex 的 2~3 倍。

### 3.3 protocol_a_800 (旧协议，photo/monet/vangogh/cezanne/Hayao)

来源: `baseline_metrics_unified.csv`

| 方法 | CLIP-Style | Content LPIPS | clip_content | 备注 |
|------|-----------|---------------|-------------|------|
| StyleID | 0.7777 | 0.5928 | 0.6402 | **旧协议，5风格不同，不与 distinct5_512 可比** |
| SD-Turbo | 0.7769 | 0.6265 | 0.6505 | 同上 |
| CUT | 0.7588 | 0.4906 | 0.7794 | 同上 |
| SaMST | 0.7253 | 0.5390 | 0.7752 | 同上 |
| S2WAT | 0.7138 | 0.5263 | 0.7464 | 同上 |
| AdaIN v32k | 0.7130 | 0.6298 | 0.6990 | 同上 |
| SDEdit str=0.20 | 0.7063 | 0.4087 | 0.7772 | 同上 |
| SDEdit str=0.10 | 0.7023 | 0.3236 | 0.8759 | 同上 |
| SDEdit str=0.40 | 0.6968 | 0.5155 | 0.6727 | 同上 |
| SDEdit str=0.35 | 0.6966 | 0.4904 | 0.6899 | 同上 |
| AdaIN vgg19 | 0.6930 | 0.6870 | 0.5991 | 同上 |
| AdaIN bad | 0.6308 | 0.8490 | 0.5297 | 同上 |

> **重要**: protocol_a_800 的 5 风格 (photo/monet/vangogh/cezanne/Hayao) 与 distinct5_512 的 5 风格 (Early_Renaissance/Impressionism/Minimalism/Rococo/Ukiyo_e) 完全不同。两组指标不可直接比较。

### 3.3 之前的本地"复现"结果（已标记无效）

以下结果因推理参数/训练过程与原始实验不一致，标记为 **不可信**：

| 方法 | 问题 |
|------|------|
| SDEdit ×4 | prompt/seed 可能与远程实验不一致 |
| SD-Turbo | 错误使用 2 步推理而非 1 步，pipeline 不对 |
| AdaIN ×3 | 官方 decoder.pth 下载失败，临时训练了劣质 decoder |

---

## 4. Baseline 方法详情

### 4.1 Identity

| 项目 | 详情 |
|------|------|
| **方法** | 恒等映射，不做任何风格迁移，输出 = 输入 |
| **用途** | 作为参考基线，衡量"不做任何事"时的指标下限 |
| **代码** | `exp/baseline_images/identity/`（直接复制源图） |
| **训练** | 不需要 |
| **推理** | 直接复制源图，按 `{src_style}__{src_stem}__to__{tgt_style}.png` 重命名 |
| **distinct5_512 结果** | CLIP-S=0.6933, LPIPS=0.0000, Δ_idt=+0.0534 |

---

### 4.2 AdaIN

| 项目 | 详情 |
|------|------|
| **论文** | Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (Huang & Belongie, ICCV 2017) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\adain\` (仅 `adain_net.py`) |
| **类型** | 需要训练 (VGG Encoder 固定 + Decoder 训练) |
| **训练数据** | MS-COCO (内容) + WikiArt (风格) |
| **训练** | `python train.py --content_dir /path/to/mscoco --style_dir /path/to/wikiart --vgg vgg_normalised.pth --save_dir models/ --batch_size 8` |
| **推理** | `python test.py --content input/content/ --style input/style/ --output output/ --model decoder.pth --vgg vgg_normalised.pth` |
| **预训练权重** | `https://github.com/naoto0804/pytorch-AdaIN` → `decoder.pth` + `vgg_normalised.pth` |
| **变体** | v32k (训练32k iter), vgg19 (VGG-19特征), bad (仅scale无shift) |
| **protocol_a_800 结果** | v32k: CLIP-S=0.713, LPIPS=0.630; vgg19: 0.693/0.687; bad: 0.631/0.849 |
| **复现状态** | ❌ 需下载官方预训练权重，在 distinct5_512 上推理评估 |

---

### 4.3 StyTR-2

| 项目 | 详情 |
|------|------|
| **论文** | StyTr^2: Image Style Transfer with Transformers (Deng et al., CVPR 2022) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\StyTR-2\` |
| **类型** | 需要训练 (Transformer encoder + decoder) |
| **训练数据** | WikiArt (风格) + COCO2014 (内容) |
| **依赖** | python 3.6, pytorch 1.4.0 |
| **预训练权重** | [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M), [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy) |
| **训练** | `python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8` |
| **推理** | `python test.py --content_dir input/content/ --style_dir input/style/ --output out` |
| **复现状态** | ❌ 需下载预训练权重，在 distinct5_512 上推理评估 |

---

### 4.4 SaMAM

| 项目 | 详情 |
|------|------|
| **论文** | SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer (Liu et al., CVPR 2025 Highlight) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\SaMam\` |
| **类型** | 需要训练 (Mamba backbone) |
| **训练数据** | WikiArt (风格) + MS_COCO (内容)。远程实验在 distinct5_512 上训练 |
| **依赖** | cuda≥12.0, python=3.10.4, torch=2.3.0, mamba-ssm=2.2.2, causal-conv1d=1.4.0 |
| **预训练权重** | [patch8](https://drive.google.com/file/d/1HH_cSdtUzdgUMspwJ30Osax-LpdF9zag), [patch4](https://drive.google.com/file/d/1fgs0JR06WBh2ACuI5OboDYrutI2G9tC7) |
| **VGG** | `vgg_normalised.pth` → `./LOSS/vgg_ckp/` |
| **训练命令** | `python train_SaMam.py --content ./Dataset/MS_COCO/ --style ./Dataset/wikiart/ --gpus 0 1 --patch-size 8` |
| **推理命令** | `python test_image.py --content-dir ./test_images/content/ --style-dir ./test_images/style/ --output-dir ./output --model_ckpt ./checkpoint/SaMam_patchsize_x8.ckpt` |
| **已有checkpoint** | `TRAIN/final_model.ckpt` (本地)，远程训练在 distinct5_512 上 |
| **已有结果** | distinct5_512: CLIP-S=0.7175(diag-3000), 0.7074(diag-2250); LPIPS=0.2423/0.2374 |
| **远程路径** | `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag` |
| **复现状态** | ✅ 图片已从远程拉取，本地评估完成。Latent 系列需从远程拉取 |

**评估曲线（distinct5_512, diag mamba b6 seg250）**:

| step | clip_style | content_lpips | clip_content |
|------|-----------|---------------|-------------|
| 250 | 0.5480 | 0.6006 | 0.6562 |
| 500 | 0.5630 | 0.5424 | 0.7046 |
| 1000 | 0.5659 | 0.4605 | 0.7571 |
| 1500 | 0.5805 | 0.4128 | 0.8410 |
| 2000 | 0.5833 | 0.3622 | 0.8670 |
| 2250 | 0.7074 | 0.2374 | — |
| 3000 | 0.7175 | 0.2423 | — |

> 注: step 250-2000 使用 LPIPS-VGG 评估，2250/3000 使用 LPIPS-Alex。两组不完全可比。

---

### 4.5 SaMST

| 项目 | 详情 |
|------|------|
| **论文** | Pluggable Style Representation Learning for Multi-Style Transfer (Liu et al., ACCV 2024) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\SaMST-main\` |
| **代码路径 (external)** | `G:\GitHub\Latent_Style\Related_Works\repos\external\SaMST\` (含已训练 checkpoint) |
| **类型** | 需要训练 (每风格训练 style representation + transfer net) |
| **训练数据** | WikiArt 子集 (风格) + MS_COCO (内容) |
| **依赖** | python=3.8.0, torch=2.0.0, CUDA=11.7 |
| **训练 (多风格)** | `cd train_model/train1/ && python train.py` |
| **训练 (少风格，快)** | `cd train_model/train2/ && python train.py` |
| **推理** | `cd test_model/test/ && python test.py` (需设置 test.yml 中 style_num) |
| **已有checkpoint** | `external/SaMST/checkpoint/repro_5style_train2/` (5风格 epoch 5~95), `wikiart5_3600_*` 系列 |
| **已有结果** | distinct5_512: CLIP-S=0.6795, LPIPS=0.7212; protocol_a_800: CLIP-S=0.7253, LPIPS=0.5390 |
| **复现状态** | ✅ 图片已从远程拉取，本地评估完成。distinct5_512 上 LPIPS=0.72 内容保持很差 |

---

### 4.6 S2WAT

| 项目 | 详情 |
|------|------|
| **论文** | S2WAT: Image Style Transfer via Hierarchical Vision Transformer using Strips Window Attention (Zhang et al., AAAI 2024) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\S2WAT-main\` |
| **类型** | 需要训练 (Swin Transformer + VGG decoder) |
| **训练数据** | MS_COCO 80000张 (内容) + WikiArt 80000张 (风格)，预处理为 224×224 |
| **依赖** | python=3.8.13, PyTorch=1.10.0, CUDA=11.3 |
| **预训练权重** | [VGG19](https://drive.google.com/file/d/1nJt6nnEIjBfQMzbH9__TrLJfmHqkaHjy), [Pre-trained model](https://drive.google.com/file/d/16Ihs_J9ULYSze2lL5cmptvMyy-ZYJ9kN) |
| **数据预处理** | `python3 data_preprocess.py --source_dir ./source --target_dir ./target` |
| **训练** | `python3 train.py --content_dir ./input/Train/Content --style_dir ./input/Train/Style --vgg_dir ./pre_trained_models/vgg_normalised.pth --epoch 40000` |
| **推理** | `python3 test.py --input_dir ./input/Test --output_dir ./output --checkpoint_import_path ./pre_trained_models/checkpoint/checkpoint_40000_epoch.pkl` |
| **protocol_a_800 结果** | CLIP-S=0.7138, LPIPS=0.5263 |
| **复现状态** | ❌ 需在 distinct5_512 上训练和推理 |

---

### 4.7 StyleID

| 项目 | 详情 |
|------|------|
| **论文** | StyleID: Zero-shot Style Transfer via Diffusion Models |
| **代码路径** | 需确认（可能在 `Related_Works/repos/StyleID/` 或 `baseline_pipeline/scripts/run_styleid.py`） |
| **类型** | Zero-shot 推理 (无需训练) |
| **推理** | 基于 diffusers img2img pipeline，使用 SD1.5 |
| **protocol_a_800 结果** | CLIP-S=0.7777, LPIPS=0.5928 |
| **复现状态** | ❌ 需确认代码位置，在 distinct5_512 上推理 |

---

### 4.8 CUT

| 项目 | 详情 |
|------|------|
| **论文** | Contrastive Unpaired Translation (Park et al., ECCV 2020) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\external\CUT\` |
| **类型** | 需要训练 (每对风格单独训练一个模型) |
| **训练** | `python train.py --dataroot /path/to/dataset --name experiment_name --CUT_mode CUT` |
| **推理** | `python test.py --dataroot /path/to/dataset --name experiment_name --CUT_mode CUT` |
| **protocol_a_800 结果** | CLIP-S=0.7588, LPIPS=0.4906 |
| **数据集问题** | CUT 需要逐风格对训练。在 distinct5_512 上需训练 20 个模型（5×4 对，不含 identity） |
| **复现状态** | ❌ 需在 distinct5_512 上逐对训练 |

---

### 4.9 SDEdit

| 项目 | 详情 |
|------|------|
| **论文** | SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations (Meng et al., ICLR 2022) |
| **代码路径** | `G:\GitHub\Latent_Style\SchrodingerBridge\tools\infer_sd_variants.py` |
| **类型** | Zero-shot 推理 (无需训练) |
| **原理** | 给源图加噪到 step T，然后用 SD1.5 去噪回完整图像。strength 越大风格迁移越强但内容保持越差 |
| **推理** | `python tools/infer_sd_variants.py --method sdedit --strength <float>` |
| **关键参数** | `strength`: {0.10, 0.20, 0.35, 0.40}，`prompt`: 风格名称文本提示 |
| **protocol_a_800 结果** | str=0.10: 0.7023/0.3236; str=0.20: 0.7063/0.4087; str=0.35: 0.6966/0.4904; str=0.40: 0.6968/0.5155 |
| **复现状态** | ⚠️ 需确认原始实验参数(prompt/seed)后重新推理 |

---

### 4.10 SD-Turbo

| 项目 | 详情 |
|------|------|
| **论文** | Adversarial Diffusion Distillation (Sauer et al., 2023) |
| **代码路径** | `G:\GitHub\Latent_Style\SchrodingerBridge\tools\infer_sd_variants.py` |
| **类型** | Zero-shot 推理 (无需训练) |
| **原理** | 使用 `stabilityai/sd-turbo` 模型进行 1 步 img2img 推理 |
| **关键参数** | `StableDiffusionImg2ImgPipeline`, `num_inference_steps=1`, `guidance_scale=1.0` |
| **protocol_a_800 结果** | CLIP-S=0.7769, LPIPS=0.6265 |
| **复现问题** | 之前错误使用 AutoPipeline + 2步推理，与原始 1 步实验不一致 |
| **复现状态** | ❌ 需修正为 1 步推理后重新评估 |

---

### 4.11 CycleGAN / CycleGAN-Turbo

| 项目 | 详情 |
|------|------|
| **论文** | CycleGAN (Zhu et al., ICCV 2017) / CycleGAN-Turbo (Parmar et al., 2024) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\cyclegan_turbo\`, `G:\GitHub\Latent_Style\Related_Works\repos\CycleGAN\` |
| **类型** | 需要训练 |
| **复现状态** | ❌ 低优先级，未开始 |

---

### 4.12 CAST

| 项目 | 详情 |
|------|------|
| **论文** | CAST (风格迁移方法) |
| **代码路径** | `G:\GitHub\Latent_Style\Related_Works\repos\cast\` |
| **类型** | 需确认 |
| **复现状态** | ❌ 低优先级，未开始 |

---

## 5. Baseline Pipeline 自动化

### 5.1 主启动脚本

- **路径**: `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\launch_all.py`
- **功能**: 统一调度各 baseline 的训练/推理/评估

```bash
# Smoke test (1 epoch, 5 images)
python launch_all.py --smoke

# Full training
python launch_all.py --full

# 特定方法
python launch_all.py --baselines s2wat styleid --styles monet vangogh

# 仅零样本方法
python launch_all.py --zero-shot

# 复制已有 CUT 结果
python launch_all.py --baselines cut
```

### 5.2 支持的方法

| 方法 | 类型 | launch_all 中的 key |
|------|------|---------------------|
| S2WAT | 需训练 | `s2wat` |
| SaMST | 需训练 | `samst` |
| StyleID | 零样本 | `styleid` |
| StyleAligned | 零样本 | `style_aligned` |
| CUT | 复制已有 | `cut` |

### 5.3 旧数据集风格列表 (protocol_a_800)

`launch_all.py` 中的 `ALL_STYLES = ["monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]`

**注意**: 这与 distinct5_512 的风格列表不同。要在 distinct5_512 上运行，需要更新风格列表为 `["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]`。

### 5.4 子脚本位置

- `baseline_pipeline/scripts/run_s2wat.py`
- `baseline_pipeline/scripts/run_styleid.py`
- `baseline_pipeline/scripts/run_style_aligned.py`
- `baseline_pipeline/scripts/run_samst.py`
- `baseline_pipeline/scripts/copy_cut_results.py`
- `baseline_pipeline/evaluation/eval_all_baselines.py`

---

## 6. 本地复现状态

### 6.1 状态总览

| 方法 | 图片 | 本地评估 | 可信度 | 待办 |
|------|------|---------|--------|------|
| Identity | ✅ 750张 | ✅ 完成 | ✅ | — |
| SaMAM-diag-2250 | ✅ 远程拉取 | ✅ 完成 | ✅ | 拉取 latent 系列 |
| SaMAM-diag-3000 | ✅ 远程拉取 | ✅ 完成 | ✅ | 同上 |
| SaMST-40 | ✅ 远程拉取 | ✅ 完成 | ✅ | — |
| SDEdit ×4 | ⚠️ 本地生成 | ⚠️ 参数存疑 | ❌ | 确认原始 prompt/seed |
| SD-Turbo | ❌ 参数错误 | ❌ | ❌ | 修正为1步推理 |
| AdaIN ×3 | ❌ 临时训练 | ❌ | ❌ | 用官方预训练权重 |
| S2WAT | ❌ 未做 | — | — | 需训练+推理 |
| StyleID | ❌ 未做 | — | — | 需推理 |
| CUT | ❌ 旧数据集 | — | — | 需 distinct5_512 上训练 |
| StyTR-2 | ❌ 未做 | — | — | 下载预训练权重+推理 |
| CycleGAN | ❌ 未做 | — | — | 低优先级 |

### 6.2 关键教训

1. **数据集必须对齐**: protocol_a_800 (photo/monet/vangogh/cezanne/Hayao) ≠ distinct5_512 (Early_Renaissance/Impressionism/Minimalism/Rococo/Ukiyo_e)，指标不可直接比较
2. **LPIPS 用 Alex 不是 VGG**: `run_evaluation.py` 原来硬编码了 LPIPS-VGG，给出异常高的值。已修复为默认使用 LPIPS-Alex
3. **推理参数必须严格对齐**: SDEdit 的 prompt、SD-Turbo 的推理步数、AdaIN 的预训练权重——任何偏差都会导致结果失真
4. **本地训练不等于复现**: 临时训练的 AdaIN decoder 完全不能代表原论文结果
5. **Identity baseline 至关重要**: CLIP-S=0.6933 说明不做迁移时源图本身就有较高的风格得分，所有方法的 Δ_idt 才是真正衡量风格迁移能力的指标
6. **远程实验确实在 distinct5_512 上**: SaMAM/SaMST 的 distinct5_512 实验是在远程 WSL 上使用 distinct5_512 数据集训练的（5×1000 train / 5×30 test）

---

## 附录: 文件路径速查

```
项目根目录
├── G:\GitHub\Latent_Style\
│   ├── Dataset/
│   │   ├── distinct5_512/          # 当前实验数据集 (5×1000 train, 5×30 test)
│   │   │   ├── train/              # Early_Renaissance/ (1000), Impressionism/ (1000), ...
│   │   │   └── test/               # Early_Renaissance/ (30), Impressionism/ (30), ...
│   │   ├── wikiart512_5style/      # WikiArt 512 5风格 (5×3600 train)
│   │   └── wikiart_stress_splits_512/
│   ├── Related_Works/
│   │   ├── repos/
│   │   │   ├── SaMam/              # SaMAM 代码 (含本地 final_model.ckpt)
│   │   │   ├── SaMST-main/         # SaMST 官方代码
│   │   │   ├── S2WAT-main/         # S2WAT 代码
│   │   │   ├── StyTR-2/            # StyTR-2 代码
│   │   │   ├── external/
│   │   │   │   ├── CUT/            # CUT 代码 + overfit50 结果
│   │   │   │   └── SaMST/          # SaMST (含已训练 checkpoints)
│   │   │   ├── cyclegan_turbo/     # CycleGAN-Turbo
│   │   │   ├── cast/               # CAST
│   │   │   └── ...                 # AesFA, AesPA-Net, ArtBank, blora 等
│   │   └── baseline_pipeline/
│   │       ├── launch_all.py       # 自动化启动脚本 (protocol_a_800 风格)
│   │       ├── scripts/            # 各方法子脚本
│   │       ├── evaluation/         # 评估脚本
│   │       └── results/            # 远程评估结果 (protocol_a_800)
│   └── SchrodingerBridge/          # FC-SB 主项目
│       ├── tools/
│       │   ├── batch_reeval_baselines.py  # 统一评估脚本
│       │   ├── infer_sd_variants.py       # SDEdit/SD-Turbo 推理
│       │   └── infer_adain.py             # AdaIN 推理 (⚠️ 临时训练版，不可信)
│       ├── src/utils/run_evaluation.py    # 核心评估引擎
│       ├── exp/
│       │   ├── baseline_images/           # 各方法输出图像
│       │   │   ├── identity/              # Identity 基线 (750张, ✅)
│       │   │   ├── samam_diag_2250/       # SaMAM 图片 (远程拉取, ✅)
│       │   │   ├── samam_diag_3000/       # SaMAM 图片 (远程拉取, ✅)
│       │   │   ├── samst_40/              # SaMST 图片 (远程拉取, ✅)
│       │   │   ├── sdedit_str0.* /        # SDEdit 图片 (本地生成, ⚠️ 参数存疑)
│       │   │   ├── sdturbo/               # SD-Turbo (参数错误, ❌)
│       │   │   └── adain_*/               # AdaIN (临时训练, ❌)
│       │   └── baseline_reeval/           # 本地评估结果
│       │       ├── unified_eval_results.json
│       │       ├── identity/summary.json
│       │       ├── samam_diag_step2250/summary.json
│       │       ├── samam_diag_step3000/summary.json
│       │       └── samst_stepalign40/summary.json
│       ├── docs/
│       │   ├── exp_unified.csv            # 统一实验结果表
│       │   ├── exp_dashboard_v2.html      # 可视化 Dashboard
│       │   └── Related/                   # 本文档所在目录
│       └── baseline_metrics_unified.csv   # 所有 baseline 指标汇总 (含 protocol_a_800)
```
