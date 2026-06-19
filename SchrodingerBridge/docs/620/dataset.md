# dataset.md — 数据极度不平衡的影响与预处理策略

> 针对实际数据集（Photo: 6187, Hayao: 1752, Monet: 972, Vangogh: 600, Cezanne: 850）的极度不平衡问题。
> 在旧的在线 Minibatch OT 路线下，这会导致灾难性的模型坍缩；在新的 620 离线配对路线下，如果不加处理，依然会造成严重的风格饥饿（Style Starvation）。

---

## 1. 现状诊断：不平衡带来的负面影响

我们的内容池（Photo）有约 6187 张，而风格池非常悬殊（Vangogh 只有 600 张，Hayao 有 1752 张）。

### 1.1 在历史旧路线（Minibatch OT）中的影响
旧代码通过 `balance_target_styles_per_batch = true` 试图在 Batch 内强制每种风格数量一致（比如 Batch=80 时，每种风格各 16 张）。
* **致命后果**：因为 Vangogh 总共才 600 张，在大量的 Epoch 中，这 600 张图被极高频地重复采样。Minibatch OT 会被迫把很多根本不相关的 Photo 强行匹配给 Vangogh 的这几张图，导致梯度方向极其混乱（也是造成 LPIPS 崩盘的原因之一）。

### 1.2 在 620 新路线（离线 DINOv2 弱配对）中的潜在影响
如果不加干预，直接在所有风格图中进行全局 Top-K 检索：
* **概率碾压（风格饥饿）**：Hayao 的样本数是 Vangogh 的 3 倍。在余弦空间中，一张 Photo 匹配到 Hayao 的概率远大于匹配到 Vangogh。
* **后果**：模型对 Hayao 的特征学习得很充分，但对于 Vangogh 和 Cezanne 的“笔触专家”或 Cross-Attention 权重几乎得不到足够的梯度更新。最终结果就是：不管输入什么图，生成的结果都更偏向宫崎骏的色调。

---

## 2. 解决方案：分层约束的离线配对 (Stratified Offline Pairing)

为了解决这个问题，我们绝对不能在 DataLoader 里简单地“增加 Vangogh 的采样权重”（那只会导致过拟合那 600 张图的具体内容）。
**我们必须在“离线配对 (Pairing Oracle)”阶段解决它。**

### 2.1 修改 DINOv2 离线配对逻辑
我们需要将全局匹配改为**域内独立匹配 (Intra-domain Matching)**：

```python
# 伪代码：离线构建 mapping.json
for content_img in all_photos:
    content_feat = extract_dino(content_img)
    
    # 强制在每个风格子集中独立寻找 Top-K
    for style_domain in ["Hayao", "monet", "vangogh", "cezanne"]:
        style_feats = get_domain_feats(style_domain)
        # 在该 domain 内部找到与当前 photo 语义最接近的 top-8
        top8_in_domain = cosine_similarity(content_feat, style_feats).topk(8)
        
        mapping[content_img][style_domain] = top8_in_domain
```

### 2.2 修改 Dataset 采样逻辑 (Stratified Target Sampling)
在 `src/dataset.py` 中，当我们获取到一个内容图 $z_c$ 时：

```python
def __getitem__(self, idx):
    z_c_path = self.photos[idx]
    
    # 1. 强制均匀随机选择一个目标风格域 (各 25% 概率)
    # 这彻底抹平了数据集数量的不平衡！
    target_domain = random.choice(["Hayao", "monet", "vangogh", "cezanne"])
    
    # 2. 从预存的该域 Top-8 候选集中，随机选择一张作为最终目标
    z_s_path = random.choice(self.mapping[z_c_path][target_domain])
    
    z_c = load_latent(z_c_path)
    z_s = load_latent(z_s_path)
    style_rgb = load_image(z_s_path) # 用于提取 Cross-Attention 需要的特征
    
    return {"z_c": z_c, "z_s": z_s, "style_image": style_rgb, "domain": target_domain}
```

---

## 3. 为什么这样处理是正确的？

1. **从流形匹配的角度**：这相当于我们并行地在训练 4 个独立的 Flow Matching 映射：$\text{Photo} \to \text{Hayao}$，$\text{Photo} \to \text{Monet}$ 等。由于每次前向传播的域是均匀采样的（25%），每个风格接收到的总梯度量是绝对平等的，避免了“大类吃小类”。
2. **从过拟合的角度**：虽然 Vangogh 只有 600 张，但因为我们在配对时使用了 DINOv2 的语义对应（比如只把有树的内容图匹配给 Vangogh 画的树），网络学习到的是**Vangogh 画树的笔触规律**，而不是简单记住了这 600 张图的死像素。Cross-Attention 机制本身极大地提高了对少量数据的利用效率（因为它是 Patch-to-Patch 的寻址，600 张图包含了 $600 \times 256$ 个有效纹理 Patch）。

## 4. 落地建议
在 620 的预处理脚本 `tools/prematch_dino.py` 中，**必须加入按文件夹（Domain）隔离计算相似度的逻辑**，并在生成的 JSON 中保留 `domain` 层级。
这是以极低的代码成本，完美化解了悬殊数据量带来的优化灾难。
