# Baseline 256 photo2art 重做实验（legacy256_overfit50）

**整理周期**: 2026-07-04
**实验目的**: 在正确的 256 数据集（legacy256_overfit50, photo2art 5 风格）上重跑所有 baseline，与 512 (distinct5) 对比，验证结论普适性
**与之前 distinct5 256 实验的区别**: 数据集从 distinct5 切换为 photo2art 5 风格（cezanne/Hayao/monet/photo/vangogh），每风格 30 张 test

---

## 1. 实验设计

### 1.1 测试集

`I:/legacy256_overfit50/test`（photo2art 5 风格 × 30 张 = 150 src × 5 target = 750 生成）

**风格列表**: cezanne, Hayao, monet, photo, vangogh

**文件名格式**: `{src_style}_{src_id}_to_{tgt_style}.jpg`

### 1.2 评估指标（5 个）

| 指标 | 计算方式 | 后端 |
|---|---|---|
| CLIP-S | cos(CLIP_image(gen), CLIP_image(ref_style_prototype))，ref_prototype 为目标风格测试集 30 张图的 CLIP 特征均值 | HF `openai/clip-vit-base-patch32` |
| CLIP-T | cos(CLIP_image(gen), CLIP_text(style_prompt)) | 同上 |
| LPIPS | content_distance = LPIPS(gen, src_content)，AlexNet | pyiqa |
| MUSIQ | Resize(256) → pyiqa musiq | pyiqa musiq_koniq |
| ART-FID | (1 + FID) * (1 + LPIPS_content)，max_gen=200 | `src/utils/artfid_metric.py` |

### 1.3 执行环境

| 项 | 值 |
|---|---|
| 远程 | `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62` |
| WSL venv | `/home/xy/venvs/samam312/bin/python`（torch 2.4.0+cu121, mamba_ssm 2.2.4） |
| GPU | RTX 3060 12GB |
| 评估脚本 | `scripts/batch_compute_photo2art.py` |

---

## 2. 完整结果（legacy256_overfit50, photo2art 5 风格）

### 2.1 主表：所有方法 5 指标

| 方法 | 空间 | CLIP-S↑ | CLIP-T↑ | LPIPS↓ | MUSIQ↑ | ART-FID↓ | 备注 |
|---|---|---|---|---|---|---|---|
| Identity (上界参考) | pixel 256 | 0.6632 | 0.2302 | 0.0000 | 56.83 | 140.80 | 恒等复制 |
| Seedream (生成上界) | pixel 256 | 0.7515 | 0.2731 | 0.2270 | 64.00 | 174.45 | 文生图参考 |
| AdaIN | pixel 256 | 0.6659 | 0.2362 | 0.6057 | 41.23 | 334.58 | train-free, VGG19+decoder |
| WCT | pixel 256 | 0.6880 | 0.2386 | 0.6142 | 40.33 | 342.66 | train-free, SVD whitening+coloring |
| SAMST | pixel 256 | 0.7094 | 0.2439 | 0.2785 | 40.73 | 184.06 | trained, per-style TransformerNet |
| SaMam | pixel 256 | 0.6769 | 0.2309 | 0.1172 | 50.03 | 186.25 | trained, mamba SSM + SDXL VAE |
| **Our latent256 e10** | latent 256 | ⏳ | | | | | 需重训（无 photo2art train 集） |
| **Our pixel256 e3** | pixel 256 | ⏳ | | | | | 需重训（无 photo2art train 集） |

### 2.2 Baseline 排名（CLIP-S 降序）

| 排名 | 方法 | CLIP-S | LPIPS | MUSIQ | ART-FID |
|---|---|---|---|---|---|
| 1 | **SAMST** | 0.7094 | 0.2785 | 40.73 | 184.06 |
| 2 | WCT | 0.6880 | 0.6142 | 40.33 | 342.66 |
| 3 | SaMam | 0.6769 | 0.1172 | 50.03 | 186.25 |
| 4 | AdaIN | 0.6659 | 0.6057 | 41.23 | 334.58 |

### 2.3 关键观察

1. **SAMST 风格转移最强（CLIP-S=0.7094）**：per-style 训练的 TransformerNet 在 photo2art 5 风格上表现最佳，因为训练集与测试集风格完全匹配。
2. **SaMam 内容保留最佳（LPIPS=0.1172）**：远低于其他 baseline（0.27-0.61），mamba SSM + latent 空间使其几乎不改变内容结构。
3. **WCT/AdaIN 风格转移弱但内容破坏大**：LPIPS 0.61 意味着内容严重失真，CLIP-S 仅 0.67-0.69。
4. **MUSIQ 两极分化**：SaMam (50.03) 和 Identity (56.83) 较高，其他 baseline 在 40-41 区间（生成质量较差）。
5. **ART-FID SAMST/SaMam 最优（~185）**：WCT/AdaIN 高达 335-342（质量差 + 内容破坏双重惩罚）。

---

## 3. 与 distinct5 256 结果对比（验证数据集普适性）

### 3.1 distinct5 256 结果（之前，来自 compare_256_vs_512.md）

| 方法 | CLIP-S | LPIPS |
|---|---|---|
| SaMam | 0.5837 | 0.3584 |
| AdaIN | 0.5547 | 0.7142 |
| WCT | 0.5599 | 0.7177 |
| SAMST | 0.5584 | 0.5824 |

### 3.2 photo2art 256 结果（本次）

| 方法 | CLIP-S | LPIPS |
|---|---|---|
| SaMam | 0.6769 | 0.1172 |
| AdaIN | 0.6659 | 0.6057 |
| WCT | 0.6880 | 0.6142 |
| SAMST | 0.7094 | 0.2785 |

### 3.3 跨数据集一致性分析

**⚠️ 排名变化**：
- distinct5: SaMam > WCT > SAMST > AdaIN（CLIP-S 0.55-0.58，差距小）
- photo2art: **SAMST > WCT > SaMam > AdaIN**（CLIP-S 0.67-0.71，差距大）

**关键差异**：
1. **photo2art 上所有 baseline CLIP-S 显著提升（+0.09 ~ +0.15）**：photo2art 5 风格（cezanne/monet/vangogh 等）风格特征更鲜明，CLIP 更易识别。
2. **SAMST 在 photo2art 上跃升至首位**：因为 SAMST 是 per-style 训练，photo2art 训练集与测试集风格完全匹配，优势充分发挥。
3. **SaMam 在 photo2art 上 LPIPS 极低（0.1172）**：内容保留能力在鲜明风格下更突出。
4. **WCT/AdaIN 在 photo2art 上 CLIP-S 提升但 LPIPS 仍高**：风格转移力度增加，但内容破坏未改善。

**核心一致性**：
- ✅ AdaIN 始终最弱（CLIP-S 最低 + LPIPS 高）
- ✅ SaMam 始终内容保留最佳（LPIPS 最低）
- ⚠️ SAMST 与 SaMam 的排名取决于数据集：per-style 训练数据匹配时 SAMST 更强

---

## 4. 各方法详细数据

### 4.1 SaMam 256 (photo2art)

| 项 | 值 |
|---|---|
| Checkpoint | `/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/final_model.ckpt` (451MB) |
| VAE | 本地 modelscope 缓存 `stabilityai/sd-vae-ft-ema` |
| 推理脚本 | `scripts/gen_samam_256_photo2art.py` |
| 评估脚本 | `scripts/batch_compute_photo2art.py` |
| 输出目录 | `/mnt/i/exp_256_photo2art/samam_256/images/` |
| CLIP-S | 0.6769 |
| CLIP-T | 0.2309 |
| LPIPS | 0.1172 |
| MUSIQ | 50.03 |
| ART-FID | 186.25 (FID=165.70, content_dist=0.1172) |
| 推理耗时 | 99.7s（750 张，0.13s/img） |
| 评估耗时 | 27s |

### 4.2 SAMST 256 (photo2art)

| 项 | 值 |
|---|---|
| Checkpoint | per-style `epoch_100.model`（4 art styles）+ photo identity |
| 推理脚本 | `scripts/gen_samst_256_photo2art.py` |
| 输出目录 | `/mnt/i/exp_256_photo2art/samst_256/images/` |
| CLIP-S | 0.7094 |
| CLIP-T | 0.2439 |
| LPIPS | 0.2785 |
| MUSIQ | 40.73 |
| ART-FID | 184.06 |
| 推理耗时 | 17.4s（750 张） |

### 4.3 AdaIN 256 (photo2art)

| 项 | 值 |
|---|---|
| 推理脚本 | `scripts/infer_adain_wct_256.py --method adain` |
| 输出目录 | `/mnt/i/exp_256_photo2art/adain_256/images/` |
| CLIP-S | 0.6659 |
| CLIP-T | 0.2362 |
| LPIPS | 0.6057 |
| MUSIQ | 41.23 |
| ART-FID | 334.58 |
| 推理耗时 | 16.1s（750 张） |
| 关键修复 | vgg_normalised.pth (Conv2d(3,3,1) 首层) + 无 ImageNet normalize |

### 4.4 WCT 256 (photo2art)

| 项 | 值 |
|---|---|
| 推理脚本 | `scripts/infer_adain_wct_256.py --method wct` |
| 输出目录 | `/mnt/i/exp_256_photo2art/wct_256/images/` |
| CLIP-S | 0.6880 |
| CLIP-T | 0.2386 |
| LPIPS | 0.6142 |
| MUSIQ | 40.33 |
| ART-FID | 342.66 |
| 推理耗时 | 54.9s（750 张，含 SVD 分解） |
| alpha | 0.6（内容保留混合） |

### 4.5 Identity 256 (上界参考)

| 项 | 值 |
|---|---|
| CLIP-S | 0.6632 |
| CLIP-T | 0.2302 |
| LPIPS | 0.0000（恒等） |
| MUSIQ | 56.83 |
| ART-FID | 140.80 |

### 4.6 Seedream 256 (生成上界参考)

| 项 | 值 |
|---|---|
| CLIP-S | 0.7515 |
| CLIP-T | 0.2731 |
| LPIPS | 0.2270 |
| MUSIQ | 64.00 |
| ART-FID | 174.45 |

---

## 5. Ours (latent256/pixel256) 处理状态

### 5.1 当前障碍

**legacy256_overfit50 数据集只有 test 集（每风格 30 张），没有 train 集**：
- `I:/legacy256_overfit50/` 仅含 `test/` 子目录
- 无 `train/` 目录
- 无 latent/pixel 预编码 cache（`legacy256_overfit50_latent256` / `legacy256_overfit50_pixel256` 不存在）

### 5.2 Ours 训练需求

Ours 模型（latent256/pixel256）需要 train 集进行训练：
- `630_latent_256.json` 原配置用 distinct5 数据集（`I:/wikiart_distinct5_samam_512_latent256/train`）
- `630_pixel_256.json` 原配置用 distinct5 数据集（`I:/wikiart_distinct5_samam_512_pixel256/train`）
- 要在 photo2art 5 风格上训练，需要 photo2art 的 train 集 + latent/pixel 预编码

### 5.3 可选方案（待用户决定）

1. **方案 A**：仅用 baseline 对比（Ours 在 256 photo2art 上缺席）
2. **方案 B**：用 distinct5 训练的 Ours checkpoint 直接在 legacy256 test 上评估（跨数据集泛化测试，但风格不匹配）
3. **方案 C**：寻找/构造 photo2art 5 风格的 train 集，重训 Ours（工程量大）
4. **方案 D**：Ours 256 沿用 distinct5 结果（与 baseline photo2art 结果分开呈现）

---

## 6. 复现命令

### 6.1 SaMam 256 推理 + 评估

```bash
# 推理（远程 WSL）
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_samam_256_photo2art.sh

# 评估
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_eval_samam_only_256.sh
```

### 6.2 AdaIN + WCT 256 推理 + 评估

```bash
# 推理
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_adain_wct_infer.sh

# 评估
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_reinfer_eval_adain_wct.sh
```

### 6.3 SAMST 256 推理 + 评估

```bash
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_eval_samst_only_256.sh
```

---

## 7. 实验文件清单

### 7.1 推理脚本

| 脚本 | 用途 |
|---|---|
| `scripts/gen_samam_256_photo2art.py` | SaMam 256 推理（latent 空间，SDXL VAE） |
| `scripts/gen_samst_256_photo2art.py` | SAMST 256 推理（per-style TransformerNet） |
| `scripts/infer_adain_wct_256.py` | AdaIN + WCT 256 推理（vgg_normalised encoder） |
| `scripts/gen_identity_256.py` | Identity 256 生成（恒等复制） |

### 7.2 评估脚本

| 脚本 | 用途 |
|---|---|
| `scripts/batch_compute_photo2art.py` | 统一 5 指标评估（CLIP-S/CLIP-T/MUSIQ/ART-FID） |

### 7.3 远程输出

| 方法 | 远程路径 |
|---|---|
| AdaIN | `/mnt/i/exp_256_photo2art/adain_256/images/` |
| WCT | `/mnt/i/exp_256_photo2art/wct_256/images/` |
| SAMST | `/mnt/i/exp_256_photo2art/samst_256/images/` |
| SaMam | `/mnt/i/exp_256_photo2art/samam_256/images/` |
| Identity | `/mnt/i/exp_256_photo2art/identity_256/images/` |
| Seedream | `/mnt/i/exp_256_photo2art/seedream_256/images/` |
| 评估结果 | `/mnt/i/exp_256_photo2art/eval_*.json` |

---

**最后更新**: 2026-07-04 20:30 (Asia/Shanghai)
**实验执行**: 远程 RTX 3060 12GB (ssh -p 2222 administrator@100.115.18.62) + WSL samam312 venv
**文档作者**: TRAE agent
