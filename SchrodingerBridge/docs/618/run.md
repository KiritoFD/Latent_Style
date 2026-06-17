# 618 外部方法复现计划

> 目标机: RTX 4070 Laptop (8GB VRAM)
> 测试集: WikiArt distinct5, 5×5×30 = 750 images
> 指标: CLIP-S, LPIPS (与我们的 eval pipeline 一致)

---

## 可直接跑的 (training-free, ~8GB VRAM)

### 1. StyleGallery (CVPR 2026)

**原理**: 用预训练 SDXL, 语义区域分割 (DINO/SAM), 聚类区域匹配, 无需训练.
**VRAM**: SDXL + DINO ≈ 8-10GB. 4070 可能需加 `enable_attention_slicing()`.
**时间**: ~2-4s/image → 750 images ≈ 1h.

```bash
git clone https://github.com/xxx/StyleGallery
cd StyleGallery
# 准备 WikiArt 内容图 + 每风格选 5 张参考图
# 跑 750 张
python run_style_transfer.py --content_dir ./wikiart_content --style_dir ./wikiart_style_refs
# 用我们的 eval pipeline 出指标
python tools/experiments/rerun_full_eval_for_run.py --image_dir ./outputs
```

### 2. HAM — Heterogeneous Attention Modulation (CVPR 2026)

**原理**: 调制扩散模型中的交叉/自注意力, 无需训练.
**VRAM**: 与 StyleGallery 类似.
**时间**: ~2s/image → 750 images ≈ 25min.

### 3. Scheduled Style Injection (CVPR NTIRE 2026)

**原理**: 探索在扩散去噪过程的不同时间步注入风格特征的效果.
**VRAM**: 轻量 — 只需一次扩散推理.

---

## 需要训练但 4070 可能能跑

### 4. CSGO (NeurIPS 2025) ★最重要

**原理**: 端到端可训练统一框架, IMAGStyle 数据集 (210K 三元组).
**训练**: 需要看代码. 如果支持 bf16 + gradient checkpointing, b4-b8 可能能在 4070 上跑.
**备选**: 只下载预训练权重做推理 (如果放出了), 不训练.

---

## 不在 4070 上跑的

| 方法 | 原因 |
|------|------|
| SigStyle | 需要 DreamBooth 微调 + 个人化 T2I |
| StyleShot | 训练需要大规模开放域数据 |

---

## 复现优先级

| 优先级 | 方法 | 操作 | 预期时间 |
|:---:|------|------|:---:|
| 1 | StyleGallery | 直接跑 750 张 | 1h |
| 2 | HAM | 直接跑 750 张 | 30min |
| 3 | CSGO | 找预训练权重做推理 | 1-2h |
| 4 | CSGO | 轻量微调 (如放权重) | 半天 |

---

## 跑完后

```bash
# 用我们的 eval 统一出指标
for method in stylegallery ham csgo; do
    python tools/experiments/rerun_full_eval_for_run.py \
        --image_dir ./outputs/$method \
        --dataset distinct5_512
done
# 写入 all_experiments.csv
```
