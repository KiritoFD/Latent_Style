# 618 外部方法复现计划

> 目标机: RTX 4070 Laptop (8GB VRAM)
> 测试集: WikiArt distinct5, 5×5×30 = 750 images
> 仓库根: `G:\GitHub\Latent_Style\Related_Works\`

---

## 论文与代码索引

| 方法 | arXiv | 会议 | GitHub | HuggingFace | 克隆? |
|------|-------|------|--------|-------------|:---:|
| StyleGallery | [2603.10354](https://arxiv.org/abs/2603.10354) | CVPR 2026 | ⬜ 未公开 | — | ⬜ |
| HAM | [2603.24043](https://arxiv.org/abs/2603.24043) | CVPR 2026 Findings | ⬜ 未公开 | — | ⬜ |
| CSGO | [2408.16766](https://arxiv.org/abs/2408.16766) | NeurIPS 2025 | [instantX-research/CSGO](https://github.com/instantX-research/CSGO) | [InstantX/CSGO](https://huggingface.co/InstantX/CSGO) | ✅ |
| SCSA | [2503.04119](https://arxiv.org/abs/2503.04119) | CVPR 2025 Highlight | ⬜ 未公开 | — | ⬜ |
| StyleShot | [2407.01414](https://arxiv.org/abs/2407.01414) | ICLR 2025 | ⬜ 未公开 | — | ⬜ |
| SaMST | — | arXiv 2025 | ✅ 已有 | — | ✅ |

---

## 克隆命令

```bash
cd G:\GitHub\Latent_Style\Related_Works

# CSGO (NeurIPS 2025) — 唯一已公开代码
git clone https://github.com/instantX-research/CSGO.git

# SaMST — 已有
# 路径: G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\
```

---

## 4070 可行性评估

### CSGO — 可推理, 训练需确认

**推理**: 预训练权重在 HuggingFace (`InstantX/CSGO`), 可直接加载. VRAM ~8GB.
**训练**: 需要 IMAGStyle 数据集 (210K 三元组, 数据集未公开标注"Coming Soon"). 训练成本未知, 不推荐本地训练.

### StyleGallery / HAM / SCSA / StyleShot

代码未公开 (arXiv 页面无 GitHub 链接). 可能需要等作者放出或联系作者.

---

## 评估流程 (CSGO 可用)

```bash
cd G:\GitHub\Latent_Style\Related_Works\CSGO

# 1. 安装依赖 (看 requirements.txt)
pip install -r requirements.txt

# 2. 准备 WikiArt 内容图 + 每风格选 5 张参考图
mkdir wikiart_test
# 从 WikiArt distinct5 测试集拷贝 30 张测试图
# 从 WikiArt 训练集每风格选 5 张风格参考图

# 3. 跑推理 (看 demo 或 inference 脚本)
python inference.py --content_dir ./wikiart_test/content --style_dir ./wikiart_test/style_refs

# 4. 用我们的 eval pipeline 出指标
cd G:\GitHub\Latent_Style\SchrodingerBridge
python tools/eval_* --image_dir ../Related_Works/CSGO/outputs
```
