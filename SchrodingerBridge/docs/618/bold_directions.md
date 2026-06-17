# 大胆的改进方向 — 不调参, 改范式

> 当前死局: blend=1.0 锁死 attention, tokenizer 5 向量查表, 所有人有参考图我们没有.
> 以下方向不追求"fine-tune 参数", 而是改变模型的底层设计逻辑.

---

## 方向1: 反转 TopoGate — Style-Locked Attention

**现状**: TopoGate 锁 content self-attention → style 进不来.
**反转**: 锁 style self-attention, 把 content 作为调制信号.

$$A_{\text{final}} = 0.3 \times A_{\text{self-content}} + 0.7 \times A_{\text{style}}$$

先用 style features 生成一个"风格画布", 内容只作为微调信号.
Content 从"不可破坏的铁律"降级为"软约束".

**预期**: style 直接突破 0.72, LPIPS 可能升到 0.40-0.50. 需要配合更强的 content correction (PC solver).

**代码量**: 改一个参数 + 扫描.

---

## 方向2: 抛弃 Tokenizer — 直接匹配传输

**现状**: tokenizer 输出 spatial_map → UNet modulation. Tokenizer 是瓶颈.
**替代**: 不用 tokenizer, 直接用 OT-matched 目标图做 **多尺度 latent 特征匹配**.

训练时:
1. OT 匹配找到 matched_target $z_t$
2. 同时从 content $z_0$ 和 matched $z_t$ 提取 UNet encoder 多尺度特征
3. Content 的低频特征 + target 的高频特征 → 混合后解码
4. Loss = 混合特征的 SWD vs target 分布

**不需要 tokenizer, 不需要 style_id, 不需要 spatial_map**. 风格信息直接从 matched target 的特征中"借"过来.

**预期**: style 0.72+, LPIPS 取决于频率分离的质量.

**代码量**: 中等 — 需要改 model forward 的特征混合逻辑.

---

## 方向3: 对抗风格鉴别器 — Fool-the-Discriminator

**现状**: Terminal SWD 是弱风格信号. SWD 只能匹配分布矩, 对高频纹理不敏感.
**替代**: 加一个轻量风格鉴别器 $D_s$, 判别输出是否属于目标风格.

$$\mathcal{L}_{\text{style}} = -\mathbb{E}_{z \sim p_{\text{generated}}}[\log D_s(z, \text{style\_id})]$$

$D_s$ 是一个浅层 CNN (比 UNet 小 10 倍), 只判断"这张 latent 是否像目标风格".

**这提供了比 SWD 强得多的风格梯度**. 鉴别器可以捕捉 SWD 漏掉的高频笔触特征.

**预期**: style 突破 0.73+. 训练不稳定风险 → 加 gradient penalty.

**代码量**: 低 — 加一个鉴别器类 + 对抗 loss.

---

## 方向4: 内容约束后置 — Content as Corrector, Not Anchor

**现状**: 整个架构围绕内容保持设计 — residual + skip + TopoGate + velocity.
**反转**: 训练时放开风格 (降低 blend 到 0, 自由风格化), 推理时用 PC solver 修正.

**训练**: blend=0 (完全自由风格化), 不加 kinetic, 不加任何结构约束
**推理**: solver_pc + latent_lowpass corrector 把宏观结构拉回 content

**这将大幅释放 style 能力** — 训练时模型可以自由地"画"任何风格, 不受 content 约束.
推理时 PC solver 作为事后校正.

**预期**: style 0.73+, LPIPS 取决于 PC solver 参数.

**代码量**: 极低 — 改 config 即可验证.

---

## 方向5: 从 matched_target 学风格 — Instance-Level Style Encoding

**现状**: tokenizer 的 style values 是 Embedding(5, D) — 5 个固定向量.
**替代**: 每次训练迭代, OT 匹配后, 用一个 StyleBankEncoder 从 matched_target 中编码风格特征:

```python
# 训练循环中:
matched_target = ot_match(content, target_style_images)
style_code = StyleBankEncoder(matched_target)  # 从实际风格图中编码
# style_code 替代 style_id lookup
```

StyleBankEncoder 是一个轻量网络 (几层 Conv), 从风格实例中提取纹理/笔触/色彩信息.

**这个改变让 tokenizer 从"查表"升级为"编码"** — 风格表征不再是 5 个固定向量, 而是从实际风格图中动态提取.

**预期**: style 0.71-0.73, 配合降低 blend 效果更佳.

**代码量**: 中等 — 新增 StyleBankEncoder 类, 修改 losses.py 的风格注入逻辑.

---

## 方向6: 多模态风格表征 — 参考图+ID 混合

**现状**: 只有 style_id, 没有参考图 → 风格表征极弱.
**方案**: 在**推理时**允许可选的参考图输入. 训练时仍然用 style_id, 但设计一个 reference encoder 分支.

```
训练: style = Embedding(style_id)  # 现有
推理: style = ReferenceEncoder(ref_image)  # 新, 可选
```

ReferenceEncoder 只在推理时激活, 编码参考图的风格 → 注入 tokenizer.
训练时仍用 style_id — 保持无配对训练的优势.

**这是最小代价获得最大收益的方向**. 代码量小 (加一个可选 encoder), 但直接把"无参考图"的劣势变成了"可选参考图"的优势.

**预期**: 有参考图时 style 0.74+, 无参考图时保持现有 0.67.

**代码量**: 低 — 加一个 encoder + 推理时的分支选择.

---

## 方向优先级

| 优先级 | 方向 | 原因 |
|:---:|------|------|
| 1 | 方向4: blend=0 + PC solver | 零代码, 立即验证 |
| 2 | 方向1: 反转TopoGate | 改一个参数 |
| 3 | 方向5: matched_target编码 | 最根本的改进 |
| 4 | 方向6: 多模态 | 最小代价最大收益 |
| 5 | 方向3: 对抗鉴别器 | 风格信号最强 |
| 6 | 方向2: 抛弃tokenizer | 最激进, 风险最大 |
