# 618 外部方法复现计划

> 目标机: RTX 4070 Laptop (8GB VRAM)
> 测试集: WikiArt distinct5, 5×5×30 = 750 images
> 仓库根: `G:\GitHub\Latent_Style\Related_Works\`

## 论文与代码索引

| 方法 | arXiv | 会议 | GitHub | 克隆 |
|------|-------|------|--------|:---:|
| StyleGallery | [2603.10354](https://arxiv.org/abs/2603.10354) | CVPR 2026 | [iiiiiiiword/StyleGallery](https://github.com/iiiiiiiword/StyleGallery) | ✅ |
| CSGO | [2408.16766](https://arxiv.org/abs/2408.16766) | NeurIPS 2025 | [instantX-research/CSGO](https://github.com/instantX-research/CSGO) | ✅ |
| SCSA | [2503.04119](https://arxiv.org/abs/2503.04119) | CVPR 2025 Highlight | [scn-00/SCSA](https://github.com/scn-00/SCSA) | ✅ |
| StyleShot | [2407.01414](https://arxiv.org/abs/2407.01414) | ICLR 2025 | [open-mmlab/StyleShot](https://github.com/open-mmlab/StyleShot) | ✅ |
| SaMST | — | arXiv 2025 | ✅ 已有 | ✅ |

> HAM (CVPR 2026) — 已移除: 核心机制 (attention 调制) 与我们的 TopoGate 重叠度太高, 单独对比价值有限.

## 克隆命令 (全部)

```bash
cd G:\GitHub\Latent_Style\Related_Works
git clone https://github.com/iiiiiiiword/StyleGallery.git   # CVPR 2026
git clone https://github.com/instantX-research/CSGO.git     # NeurIPS 2025
git clone https://github.com/scn-00/SCSA.git                 # CVPR 2025 Highlight style-transfer repo
git clone https://github.com/open-mmlab/StyleShot.git        # ICLR 2025
```

> 注意: `HZAI-ZJNU/SCSA` 是同名的分类/检测/分割注意力模块仓库，不是 arXiv:2503.04119 的语义风格迁移代码。本地正确仓库放在 `G:\GitHub\Latent_Style\Related_Works\SCSA_style_transfer\`。

## 4070 可行性评估

| 方法 | 推理 | 训练 | 说明 |
|------|:---:|:---:|------|
| StyleGallery | ✅ 已完成 750 | 不要 | Resident batch, 512/8, 已出 CLIP/LPIPS |
| CSGO | ✅ 已完成 750 低显存档 | 不要 | CPU offload + cached image embeddings, 384px/1step；512px/8step 在 8GB 上不稳定 |
| SCSA | ⚠️ 不适合直接评估 | 不要 | 官方代码要求每对图的彩色 semantic mask 和预计算 sem map；WikiArt distinct5 没有这些标注 |
| StyleShot | ✅ 已完成 750 | 不要 | Resident batch, 512/8, 已出 CLIP/LPIPS |
| SaMST | ✅ | ✅ | 已有 |

## 评估流程

```bash
# 每个方法跑完 750 张图后, 用我们的 eval pipeline 统一出指标:
cd G:\GitHub\Latent_Style\SchrodingerBridge
python tools/eval_selected_style_metrics.py --image_dir ../Related_Works/<method>/outputs/wikiart_750
```

## 618 当前复现结果

| 方法 | 输出目录 | 状态 | 结果 |
|------|------|------|------|
| StyleGallery | `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\stylegallery_wikiart5_618\images` | 完成 750 | full CLIP-style 0.697547, LPIPS 0.710688 |
| StyleShot | `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\styleshot_wikiart5_618\images` | 完成 750 | full CLIP-style 0.806562, LPIPS 0.698320 |
| CSGO | `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\csgo_wikiart5_618_cpuoffload_384_1step\images` | 低显存档完成 750 | full CLIP-style 0.654125, LPIPS 0.820927；CPU offload + cached image embeddings, 384px/1step。512/8 resident 到第 3 张后严重变慢/不稳定 |
| SCSA | `G:\GitHub\Latent_Style\Related_Works\SCSA_style_transfer` | 不纳入当前 WikiArt 750 公平评估 | 缺少 WikiArt pair-level semantic mask；使用伪 mask 会改变论文设置 |
