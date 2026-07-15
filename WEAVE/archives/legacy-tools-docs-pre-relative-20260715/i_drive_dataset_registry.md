# I 盘数据集全量注册表 (Dataset Registry)

> 全量盘点远程 I 盘 (`/mnt/i/`) 所有数据集目录, 整理为统一参考。
> 扫描时间: 2026-07-06 | 总数据集目录数: 26 | 总占用: 60.26 GB | I 盘可用: 151 GB
> 配套文档: [docs/tools/README.md](README.md) §1 数据库 (旧的简化版, 本文为全量替代)

---

## 0. 扫描方法

执行 `scripts/_scan_all_datasets.py` (远程 `I:/_scan_all_datasets.py`), 遍历 `/mnt/i/` 根目录所有候选数据集子目录, 计算:
- `size_mb` / `size_gb`: 递归累计大小
- `n_files`: 文件总数
- `kind`: 按目录名启发式分类 (latent / pixel / classview_test / flat_pixel / overfit / fewshot / splits / scitexture / exp_artifacts / container)
- `subdirs` / `sample_files`: 顶层结构样本
- `has_train` / `has_test` / `has_latent_cache` / `has_manifest`: 结构标记

输出 JSON: `/mnt/i/_dataset_registry.json`

---

## 1. 核心训练数据 (FC-SB 主线, 512 分辨率)

### 1.1 **`wikiart_distinct5_samam_512_latents_ema`** ⭐ 主线训练数据

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_latents_ema` |
| 大小 | 1588.2 MB |
| 文件数 | 5178 |
| 用途 | **FC-SB 630 系列 SOTA 训练数据 (5ep Heun)** |
| 子结构 | `train/` + `test/` |
| 5 风格 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e |
| 训练量 | 5 × 1000 = 5000 latent (SDXL VAE 编码, 4×32×32, EMA) |
| 缓存 | `train/.latent_cache/packed/` (5 个 .pt + manifest) |
| Pairing | `train/.latent_cache/prototype_pairing_top8.pt` (1.27 MB) |
| DINO pairing | `train/.latent_cache/dino_pairing_top8.pt` |
| Style cache | `train/_style_cache_620/` (model style_memory 缓存) |
| 本地等价 | `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train` |

**配置字段**:
```json
"data": {
  "data_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
  "latent_cache_dir": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
  "pairing_cache_path": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
}
```

**注意**: train 目录下有一个 stray 文件 `F:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed` (Windows 路径被误写为文件名), 不影响训练但建议清理。

### 1.2 **`wikiart_distinct5_samam_512_classview`** ⭐ 主线测试集

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_classview` |
| 大小 | 1814.4 MB |
| 文件数 | 5151 |
| 用途 | **FC-SB + 12 baseline 统一评估测试集** |
| 子结构 | `train/` + `test/` |
| 测试量 | 5 风格 × 30 = 150 source images (pixel, 512×512) |
| Pairs | 750 (150 src × 5 tgt styles, 含 identity) |

**配置字段**:
```json
"training": {
  "test_image_dir": "/mnt/i/wikiart_distinct5_samam_512_classview/test"
}
```

### 1.3 `wikiart_distinct5_latents_512_ema` (备用, 无 train/test 切分)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_latents_512_ema` |
| 大小 | 948.6 MB |
| 文件数 | 5010 |
| 用途 | 早期 latent cache (无 train/test split, 5 风格直接在根) |
| 子结构 | 5 风格目录 + `.latent_cache/` |
| 缓存 | `.latent_cache/packed/` |

**注**: 与 1.1 内容近似但结构不同 (无 train/test split)。建议优先用 1.1。

### 1.4 `wikiart_distinct5_latents_512_ema_test` (小测试 latent)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_latents_512_ema_test` |
| 大小 | 9.7 MB |
| 文件数 | 151 |
| 用途 | latent 形式的测试集 (5 风格 × 30) |

---

## 2. 全量像素 / Latent 数据集 (其他分辨率或全量版)

### 2.1 `wikiart_distinct5_samam_512_pixel256` (15 GB 像素)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_pixel256` |
| 大小 | 7510.3 MB |
| 文件数 | 5006 |
| 用途 | 256 像素版 (5 × 1000), 用于 SaMam 256 实验 |
| 子结构 | `train/` |

### 2.2 `wikiart_distinct5_samam_512_pixel128` (1.8 GB 像素)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_pixel128` |
| 大小 | 1885.2 MB |
| 文件数 | 5006 |
| 用途 | 128 像素版 (5 × 1000) |
| 子结构 | `train/` |

### 2.3 `wikiart_distinct5_samam_512_latent256` (166 MB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_latent256` |
| 大小 | 166.4 MB |
| 文件数 | 5006 |
| 用途 | 256 分辨率 latent (downsampled), 用于 256 res 实验 |
| 子结构 | `train/` |

### 2.4 `wikiart_distinct5_samam_512_flat` (3.6 GB flat pixel)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_flat` |
| 大小 | 3627.8 MB |
| 文件数 | 10300 |
| 用途 | flat 结构 pixel (train_flat + test_flat) |

### 2.5 `wikiart_distinct5_samam_512_classview_real` (2.4 MB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_distinct5_samam_512_classview_real` |
| 大小 | 2.4 MB |
| 文件数 | 1 |
| 用途 | 仅 `train_style_captions.jsonl`, 风格描述文本 |

### 2.6 `datasets/` (容器目录, 7.2 GB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/datasets` |
| 大小 | 7239.0 MB |
| 文件数 | 15453 |
| 用途 | 容器目录, 内含 `wikiart_distinct5_512_images/` + `wikiart_distinct5_samam_512/` + zip |

### 2.7 `wikiarts_5_full_notest` + `_latents_ema` (无测试切分全量)

| 路径 | 大小 | 文件数 | 用途 |
|---|---|---|---|
| `/mnt/i/wikiarts_5_full_notest` | 6596.3 MB | 18896 | 全量像素 (5 × 3600), 无 test |
| `/mnt/i/wikiarts_5_full_notest_latents_ema` | 2402.8 MB | 18904 | 对应 latent 版本 |

**注**: 早期 628 之前的训练数据, 现已被 1.1 取代。

### 2.8 `wikiart_latents_512_ema` + `_test` (旧 5 风格, 不同于 distinct5)

| 路径 | 大小 | 文件数 | 风格 |
|---|---|---|---|
| `/mnt/i/wikiart_latents_512_ema` | 1160.5 MB | 17996 | Expressionism, Impressionism, Post_Impressionism, Realism, Symbolism |
| `/mnt/i/wikiart_latents_512_ema_test` | 9.7 MB | 150 | 同上 |

**注**: 与 distinct5 的 5 风格不同 (这里是 Expressionism 系列), 历史数据。

### 2.9 `wikiart_images_512_ema_test` (66 MB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_images_512_ema_test` |
| 大小 | 66.8 MB |
| 文件数 | 150 |
| 用途 | pixel 测试图 (旧 5 风格 Expressionism 系列) |

---

## 3. Stress Test 数据集 (Faraday Splits, 15 风格)

### 3.1 `wikiart_faraday_splits` (4.5 GB, 3 splits × 5 风格)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/wikiart_faraday_splits` |
| 大小 | 4455.9 MB |
| 文件数 | 21000 |
| 用途 | Random-20 stress test (3 个 5 风格 split, 共 15 风格) |
| 元数据 | `selected_splits.json` |

**3 个 splits**:
- **stress1** (5 风格): Color_Field_Painting, High_Renaissance, Mannerism_Late_Renaissance, Pop_Art, Realism
- **stress2** (5 风格): Abstract_Expressionism, Baroque, Cubism, Northern_Renaissance, Post_Impressionism
- **stress3** (5 风格): Art_Nouveau_Modern, Expressionism, Naive_Art_Primitivism, Romanticism, Symbolism

---

## 4. Few-shot 数据集

### 4.1 `fewshot_data` (5.6 GB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/fewshot_data` |
| 大小 | 5660.2 MB |
| 文件数 | 3270 |
| 子目录 | `5p1_shot01`, `5p1_shot06`, `5p1_shot10`, `5p1_shot30`, `5p1_shot50`, `5p2_shot01`, `5p2_shot06`, `5p2_shot10`, `5p2_shot30`, `5p2_shot50` |
| 用途 | Few-shot 风格注入实验 (5p1, 5p2 两套, 各 5 个 shot 档位: 1/6/10/30/50) |

---

## 5. Legacy / Overfit 数据集 (256 历史)

### 5.1 `legacy256_overfit50` 系列

| 路径 | 大小 | 文件数 | 用途 |
|---|---|---|---|
| `/mnt/i/legacy256_overfit50` | 342.1 MB | 10511 | 256 overfit50 (train + test) |
| `/mnt/i/legacy256_overfit50_latent256` | 336.9 MB | 10368 | 256 latent 版 |
| `/mnt/i/legacy256_overfit50_pixel256` | 15554.6 MB | 10368 | 256 像素版 (15 GB, 最大单数据集) |

**注**: 早期 overfit50 调试数据, 现已废弃但保留作历史参考。

---

## 6. Scitexture 数据集

### 6.1 `Scitexture_latent_512_smoke_ema` (空目录)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/Scitexture_latent_512_smoke_ema` |
| 大小 | 0.0 MB |
| 文件数 | 0 |
| 子目录 | `Abelian_Sandpile_Critical_State_Mosaic`, `Abrasive_Grit_Random_Packing_Texture` (空) |
| 用途 | smoke test 占位 (未实际填充) |

### 6.2 `scitexture_latent_512_smoke_ema.zip` (0.6 MB)

| 属性 | 值 |
|---|---|
| 路径 | `/mnt/i/scitexture_latent_512_smoke_ema.zip` |
| 大小 | 0.59 MB |
| 用途 | 压缩包 (smoke test) |

---

## 7. Exp Artifacts (实验产物, 非数据集)

| 路径 | 大小 | 文件数 | 内容 |
|---|---|---|---|
| `/mnt/i/exp_256_photo2art` | 94.1 MB | 3772 | 5 子目录: adain_256, identity_256, samam_256, samst_256, wct_256 |
| `/mnt/i/exp_our_models_eval` | 125.3 MB | 1212 | 4 子目录: latent256_e10, latent512_e7, logs, pixel256_e3 |
| `/mnt/i/exp_samam_latent` | 0.0 MB | 2 | 仅 logs/ |
| `/mnt/i/exp_samst_latent` | 32.7 MB | 15 | SaMST epoch_01..05.model |
| `/mnt/i/exp_samst_latent_eval` | 74.2 MB | 753 | step_000001 + curve_metrics |

---

## 8. 数据集使用矩阵 (按实验类型)

| 实验类型 | 训练数据 | 测试集 | 备注 |
|---|---|---|---|
| **FC-SB 主线 (512, 630 系列)** | §1.1 `wikiart_distinct5_samam_512_latents_ema/train` | §1.2 `wikiart_distinct5_samam_512_classview/test` | 当前 SOTA 基准 |
| **512 消融实验 (本次)** | §1.1 同上 | §1.2 同上 | 40+ 极端配置 |
| **256 历史实验** | §5.1 `legacy256_overfit50_latent256` | (内置 test) | 已废弃 |
| **Stress test (Random-20)** | §3.1 `wikiart_faraday_splits` | §1.2 同上 | 3 splits × 5 风格 |
| **Few-shot 注入** | §4.1 `fewshot_data` | §1.2 同上 | 5p1/5p2 × 5 shot 档位 |
| **SaMam 256 baseline** | §2.1 `wikiart_distinct5_samam_512_pixel256` | §1.2 同上 | SaMam 自有评估管线 |
| **早期 wikiarts5 full** | §2.7 `wikiarts_5_full_notest_latents_ema` | §1.2 同上 | 628 之前, 已弃 |

---

## 9. 关键工程约束 (复述)

- **训练数据路径必须配置为 I 盘** (`/mnt/i/...`), 非 F 盘
- **测试集统一**为 §1.2 `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- **DataLoader**: `num_workers=0, pin_memory=False, persistent_workers=false` 防 CUDA OOM
- **训练必须从零开始独立目录**, 禁止 `--skip-train` resume
- **每个实验一定要分清楚用的是哪个数据集**, 在 `exp/{type}/` 下分别存放, 避免污染主线结论
- **DINO cache 必须先配置** 再跑 ablation

---

## 10. 待清理建议

| 项目 | 路径 | 大小 | 建议 |
|---|---|---|---|
| Stray 文件 | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/F:\...` | <1 KB | 删除 (Windows 路径误写为文件名) |
| Scitexture 空目录 | §6.1 | 0 MB | 可删除 (未填充) |
| Scitexture zip | §6.2 | 0.59 MB | 可保留 |
| Legacy256 像素版 | §5.1 pixel256 | 15 GB | 已废弃可考虑删除 (释放空间) |
| 旧 wikiarts5 full | §2.7 | 9 GB | 已被 1.1 取代, 可考虑删除 |
| 旧 wikiart_latents_512_ema | §2.8 | 1.2 GB | 历史数据, 可考虑归档 |

**保守清理可释放**: ~25 GB (Legacy256 pixel + 旧 wikiarts5 full)
**激进清理可释放**: ~30 GB (上述 + 旧 latent 备份)

---

**最后更新**: 2026-07-06 (基于 `scripts/_scan_all_datasets.py` 全量扫描)
**维护**: 每次新增数据集时同步更新本文档 + `_dataset_registry.json`
