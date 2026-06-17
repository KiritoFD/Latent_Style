# 618 外部方法复现计划

> 目标机: RTX 4070 Laptop (8GB VRAM)
> 测试集: WikiArt distinct5, 5×5×30 = 750 images

## 论文索引

| 方法 | arXiv | 会议 | 代码 |
|------|-------|------|:---:|
| StyleGallery | [2603.10354](https://arxiv.org/abs/2603.10354) | CVPR 2026 | ⬜ |
| HAM | [2603.24043](https://arxiv.org/abs/2603.24043) | CVPR 2026 Findings | ⬜ |
| CSGO | [2408.16766](https://arxiv.org/abs/2408.16766) | NeurIPS 2025 | ⬜ |
| SCSA | [2503.04119](https://arxiv.org/abs/2503.04119) | CVPR 2025 Highlight | ⬜ |
| StyleShot | [2407.01414](https://arxiv.org/abs/2407.01414) | ICLR 2025 | ⬜ |
| SaMST | — | arXiv 2025 | ✅ 已有 |

---

## 可直接跑的 (training-free, ~8GB VRAM)

### 1. StyleGallery (CVPR 2026, arXiv 2603.10354)

**原理**: 预训练 SDXL + 语义区域分割 (潜空间自适应聚类) + 块滤波区域匹配 + 能量函数引导扩散采样. 无需训练.
**VRAM**: SDXL inference ≈ 8-10GB, 可能需要 `enable_attention_slicing()`.
**特点**: 支持多张参考图, 区域级个性化风格定制. 论文自建了 benchmark.

### 2. HAM (CVPR 2026 Findings, arXiv 2603.24043)

**原理**: Style noise initialization + Global Attention Regulation (GAR) + Local Attention Transplantation (LAT). 在扩散模型的不同 attention 层用不同策略: GAR 调节全局 attention, LAT 做本地 attention 替换.
**VRAM**: 与 StyleGallery 类似.

### 3. CSGO (NeurIPS 2025, arXiv 2408.16766) ★最重要

**原理**: 数据构建管线生成 content-style-stylized 三元组, IMAGStyle 数据集 (210K triplets). 端到端训练, 独立的内容/风格特征注入. 支持 image-driven / text-driven / text-editing 三种模式.
**特点**: 有预训练权重可下载 (代码已公开), 可能可以直接推理.
