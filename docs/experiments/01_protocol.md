# 01 评测协议与指标

## 评测板（boards）

| 名称 | 分辨率 | 风格/域 | 请求数 | 说明 |
|---|---:|---|---:|---|
| **D5-512**（Distinct5-WikiArt，主评测） | 512 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | 750 | 每风格 3600 训练 / 150 测试；每源风格取 30 张内容图 × 5 个目标风格 = 750 有序请求；150 个同风格请求保留作校准行 |
| **P2A-256** | 256 | photo, monet, vangogh, cezanne, Hayao | 750（多数外部 baseline 只有 600 个跨域请求） | 即旧 `legacy256_overfit50` / `style_data/` 数据；"photo2art" 名称有误导：photo 也是网格中的一个域 |
| **R5-WikiArt**（Random5） | 512 | Cubism, Expressionism, Pop_Art, Romanticism, Symbolism | 750 | 从 wikiarts20（20 个家族）随机抽 5 个；**不是** "20-family benchmark"（AAAI 正文写错，ICME 版已改） |
| **Other5**（零样本） | 512 | 5 个与 D5 不相交的 WikiArt 家族 | 750 | 用 D5 checkpoint 直接评测；家族列表与原始结果**未入库**（工具：`WEAVE/tools/*other5*`） |

权威定义：`WEAVE/docs/79/README.md`（其中的数字表已过期，只看定义部分）。原始图像根目录在远程机器（`I:\datasets\...`），不在仓库内。

**命名冲突**：`D5` 既指数据集，也指 rebuttal batch-2 的"learned HH head"消融（`rebuttal_D5_hh_head.json`）。本目录中消融一律写全名。
远程机器上 `SchrodingerBridge\results\R5-WikiArt` 一类旧目录里有文件名为 D5 风格的包，见 `WEAVE/docs/79/README.md` §3 的清理规则。

## 指标（论文口径，`protocol: paper_canonical_dinov2_small`）

| 指标 | 方向 | 定义 | 实现 |
|---|---|---|---|
| DINO-S（主风格指标） | ↑ | 输出 DINOv2-small CLS 与目标风格 ≤30 张 held-out 参考 CLS 的**最大**余弦，按请求平均；源图从参考池剔除 | `WEAVE/utils/compute_dino_metrics.py`（`--dino_model_name facebook/dinov2-small --max_refs_per_style 30`），批量：`WEAVE/scripts/batch_eval_all.py` |
| CLIP-S | ↑ | CLIP ViT-B/32（`openai/clip-vit-base-patch32`）图像特征与目标风格原型的余弦 | `WEAVE/utils/run_evaluation.py` |
| DINO-C | ↑ | 输出与源图 DINOv2-small CLS 余弦 | 同 DINO-S |
| LPIPS | ↓ | AlexNet backbone，输出 vs 源图 | `run_evaluation.py --eval_lpips_net alex` |
| ArtFID | ↓ | 每目标风格 `(1+FID)(1+LPIPS)` 再平均（target-pooled） | `WEAVE/utils/artfid_metric.py`、`targetwise_artfid_summary.py` |

预处理：DINO 用 Resize224 + CenterCrop224 + ImageNet 归一化；参考缓存每风格上限 16（CLIP full-eval 记录）；后处理关闭。

> 旧协议（2026-06-15 之前）只有 `clip_style / clip_content / content_lpips`，数值尺度不同（例如 clip_style≈0.70、LPIPS≈0.34–0.46），
> 全部 root 级 CSV、`docs/results/*.csv`、`WEAVE/results/*.csv`、`Related_Works/` 的表都属于旧协议，**不能**与上表混用。

## IDT / TGT 夹逼（sandwich）

- `y_IDT = x`（原样输出）；`y_TGT = r_s`（目标风格 held-out 参考池中确定性顺序的第一张图，所有发往该风格的请求共用）。
- 判读：风格指标 > IDT，且 LPIPS < TGT、DINO-C > TGT；**逐条检查，不合成加权分**。
- D5：IDT = (0.419, 0.693, 0, 1)，TGT = (1.000, 0.863, 0.776, 0.215)（DINO-S, CLIP-S, LPIPS, DINO-C）。TGT 的 DINO-S=1 是构造所致（参考在池中）。
- TGT 敏感性：前 5 个确定性参考 → LPIPS 0.752–0.783、DINO-C 0.189–0.246（`WEAVE/docs/reproduction/tgt_reference_sensitivity.*`）。
- ICME 表中的 † / ‡ 由 `make_main_table.py` 按严格不等式自动计算（AAAI 版有两处手工标记与定义不符：StyleShot D5 LPIPS 0.765、SaMST D5 LPIPS 0.749 均小于 TGT 0.776）。

## 分母约定

| 板 | 方法 | n |
|---|---|---|
| D5 | 大多数 | 750；StyleShot 745，Seedream n_content=720 |
| P2A | 多数外部 baseline | 600（跨域；`n_skipped=150`）；CUT 588、Seedream 574；AesPA/StyTR-2 750 |
| R5 | 大多数 | 750；SD-Turbo 1123（异常，多于 750）、Seedream 724、StyleShot 740 |

每个 sidecar JSON 都记录 `n_images / n_skipped`；引用时保留。
