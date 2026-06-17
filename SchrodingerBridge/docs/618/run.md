# 618 外部方法复现计划

> 目标机: RTX 4070 Laptop (8GB VRAM)
> 测试集: WikiArt distinct5, 5×5×30 = 750 images
> 仓库根: `G:\GitHub\Latent_Style\Related_Works\`

## 论文与代码索引

| 方法 | arXiv | 会议 | GitHub | 克隆 |
|------|-------|------|--------|:---:|
| StyleGallery | [2603.10354](https://arxiv.org/abs/2603.10354) | CVPR 2026 | [iiiiiiiword/StyleGallery](https://github.com/iiiiiiiword/StyleGallery) | ✅ |
| CSGO | [2408.16766](https://arxiv.org/abs/2408.16766) | NeurIPS 2025 | [instantX-research/CSGO](https://github.com/instantX-research/CSGO) | ✅ |
| SCSA | [2503.04119](https://arxiv.org/abs/2503.04119) | CVPR 2025 Highlight | [HZAI-ZJNU/SCSA](https://github.com/HZAI-ZJNU/SCSA) | ✅ |
| StyleShot | [2407.01414](https://arxiv.org/abs/2407.01414) | ICLR 2025 | [open-mmlab/StyleShot](https://github.com/open-mmlab/StyleShot) | ✅ |
| SaMST | — | arXiv 2025 | ✅ 已有 | ✅ |

> HAM (CVPR 2026) — 已移除: 核心机制 (attention 调制) 与我们的 TopoGate 重叠度太高, 单独对比价值有限.

## 克隆命令 (全部)

```bash
cd G:\GitHub\Latent_Style\Related_Works
git clone https://github.com/iiiiiiiword/StyleGallery.git   # CVPR 2026
git clone https://github.com/instantX-research/CSGO.git     # NeurIPS 2025
git clone https://github.com/HZAI-ZJNU/SCSA.git              # CVPR 2025 Highlight
git clone https://github.com/open-mmlab/StyleShot.git        # ICLR 2025
```

## 4070 可行性评估

| 方法 | 推理 | 训练 | 说明 |
|------|:---:|:---:|------|
| StyleGallery | ✅ ~8GB | 不要 | Training-free, SDXL inference |
| CSGO | ✅ ~8GB | 不要 | HuggingFace 有预训练权重 |
| SCSA | ⚠️ 需评估 | 不要 | 即插即用模块, 需加载到现有模型 |
| StyleShot | ⚠️ 需评估 | 不要 | Style-aware encoder, 需预训练权重 |
| SaMST | ✅ | ✅ | 已有 |

## 评估流程

```bash
# 每个方法跑完 750 张图后, 用我们的 eval pipeline 统一出指标:
cd G:\GitHub\Latent_Style\SchrodingerBridge
python tools/eval_selected_style_metrics.py --image_dir ../Related_Works/<method>/outputs/wikiart_750
```
