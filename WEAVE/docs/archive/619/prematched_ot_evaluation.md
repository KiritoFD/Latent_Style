# 619: 预匹配 OT 方案评估

> 外部建议: DINOv2离线做pixel级Sinkhorn对齐 → 训练时加载预配对 → 纯Flow Matching

---

## 一、方案解决了什么

对照我们的 5 个致命缺陷:

| 缺陷 | 预匹配 OT 如何解决 |
|------|------------------|
| 1. time/style 纠缠 | **没解决** — 需要独立修改 backbone |
| 2. 伪交叉注意力 | **没解决** — 需要独立修改 attention 层 |
| 3. 闭集查表 | **部分解决** — 用离线特征, 但推理时需要 style_condition |
| 4. Minibatch OT不稳定 | ✅ **彻底解决** — 离线预计算, 配对固定 |
| 5. 训练中 ODE 展开 | ✅ **彻底解决** — 单步MSE, 不需要 terminal SWD |

---

## 二、方案的合理之处

### 2.1 DINOv2 做离线对齐 — 非常合理

DINOv2 的特征天然携带语义结构。在潜空间做 Sinkhorn pixel 级对齐 → 目标 latent 的每个位置都"对应"内容 latent 的相近语义位置。

**这替代了我们 tokenizer 的空间路由功能**: tokenizer 试图在训练中学习"哪块内容应该匹配哪块风格"。预匹配直接在预处理中完成这个匹配。训练时只需要学"怎么把内容变成风格"——纯回归任务。

### 2.2 训练极简化 — 非常合理

```python
z_t = (1 - t) * z_c + t * z_tgt_aligned
v_true = z_tgt_aligned - z_c
loss = MSE(model(z_t, t, cond), v_true)
```

没有 OT, 没有 tokenizer, 没有 ODE 展开, 没有 SWD。**这是 Flow Matching 的标准训练范式**。模型只需要学速度场——一切复杂路由都被预处理吸收了。

### 2.3 空间惩罚 — 非常好的细节

```python
cost_matrix = cos_dist + 0.1 * spatial_dist
```

加空间距离惩罚防止像素"飞到另一端"。这是 SCSA 论文中"局部性约束"的简单实现。

---

## 三、方案未解决的问题

### 3.1 Backbone 仍然需要改

即使有了预配对, `model(z_t, t, cond)` 的实现仍然需要:
- **独立 time modulation** (AdaLN on ResBlock) — 当前 `_compute_style_code` 把 time+style 混在一起
- **真实 cross-attention** — 当前 `CrossAttnAdaGN` 用的是全局 learned tokens, 不是从参考图编码的特征
- `cond` 应该作为 cross-attention 的 K, V, 而不是加到 AdaGN 上

### 3.2 推理时的风格条件怎么来?

预匹配方案在训练时保存 `style_condition` (如 CLIP/DINO embedding). 但我们推理时只有 `style_id`。

**解决方案**:
- A: 对每个 style_id 预先计算平均 style_condition → 推理时用平均值. 损失实例级变化, 但简单.
- B: 训练一个轻量 `StyleEncoder(style_image) → cond` → 推理时需要至少 1 张风格参考图. 更灵活, 但需要改设定.
- C: `StyleEncoder` 在训练时也从 aligned_latent 的原始风格图中提取, 但让它可以接收随机噪声 → 推理时不需要参考图. 类似 VAE 的思路.

**推荐 A+B**: 默认用 A (闭集), 可选 B (开放域). 这和我们之前讨论的"多模态"方向一致.

### 3.3 空间对齐可能"过强"

Sinkhorn pixel 级对齐把风格图的所有 patch 都重排到内容图的位置。但这意味着:
- 风格图中有 30% 的 patch 可能被丢弃 (OT mass 集中在匹配好的区域)
- 风格图的结构被完全破坏 → 目标 latent 不再是一个"自然图像" → 模型可能学到 unnatural transformations

**缓解**: 混合策略 — 部分对齐, 部分随机. `z_tgt_mixed = 0.7 × z_tgt_aligned + 0.3 × z_tgt_random`. 保持一定的不匹配度 → 模型需要学得更鲁棒.

---

## 四、如果从零实现 — 最小可用路径

### Phase A: 离线预处理 (DINOv2)

```
for each content image:
    for each style_id:
        pick top-20 similar style images (CLIP/DINO CLS matching)
        for each selected style image:
            compute pixel-level Sinkhorn plan between DINOv2 features
            warp style latent via plan → aligned target latent
            save (z_c, z_tgt_aligned, cond) as .pt file
```

**输出**: ~25000 pair files, ~875MB

### Phase B: 模型架构 (干净实现)

```python
class SimpleStyleTransferModel(nn.Module):
    def __init__(self):
        self.encoder = ContentEncoder()      # 标准 UNet encoder
        self.decoder = ContentDecoder()      # 标准 UNet decoder
        self.style_cross_attn = CrossAttn()  # 真实 cross-attention
        self.time_proj = TimeMLP()           # 独立时间调制
        self.style_proj = StyleMLP()         # 独立风格条件
        
    def forward(self, z_t, t, cond):         # cond 是 style condition
        time_emb = self.time_proj(t)         # → 注入 AdaLN
        feat = self.encoder(z_t, time_emb)
        style_kv = self.style_proj(cond)     # → 注入 CrossAttn K,V
        feat = self.style_cross_attn(feat, style_kv)
        return self.decoder(feat, time_emb)
```

### Phase C: 训练 (极简)

```python
for batch in dataloader:
    z_c = batch["z_c"]
    z_tgt = batch["z_tgt"]
    cond = batch["cond"]
    
    t = rand(batch_size)
    z_t = (1 - t) * z_c + t * z_tgt
    v_pred = model(z_t, t, cond)
    loss = MSE(v_pred, z_tgt - z_c)
    loss.backward()
```

---

## 五、与当前代码的差距

| 当前代码 | 预匹配方案 | 差距 |
|---------|-----------|------|
| tokenizer routing | DINOv2 离线对齐 | 彻底替换 |
| online OT matching | 离线预计算 | 消除不确定性 |
| _terminal_swd ODE展开 | 单步 MSE | 消除梯度问题 |
| 伪 cross-attention | 真实 cross-attention | 需重写 attention 层 |
| time/style 混合 | 独立 time/style 注入 | 需重写 backbone |

**离可直接训练差 3 天开发 + 2 天预处理**。
