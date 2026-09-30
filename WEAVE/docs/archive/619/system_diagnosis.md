# 619: 从第一性原理重审我们的模型 — 代码审计后的系统诊断

> 外部审查提出了 5 个致命缺陷, 逐项代码审计**全部确认属实**.

---

## 一、风格迁移的第一性原理

### 什么是风格迁移?

给定内容图 $C$ 和目标风格 $S$, 生成图像 $T$ 满足:
1. $T$ 的结构 = $C$ 的结构 (猫还是猫)
2. $T$ 的纹理/笔触 = $S$ 的纹理 (印象派的笔触)

### 实现这一点需要什么?

| 需求 | 为什么需要 | 实现方式 |
|------|-----------|---------|
| **风格感知** | 必须"看到" $S$ 的具体纹理特征 | 从 $S$ 编码空间特征图 |
| **内容感知** | 必须"知道" $C$ 的空间结构 | 从 $C$ 编码空间特征图 |
| **结构保持** | 不能改变 $C$ 的语义 | attention/残差/校正机制 |
| **风格传输** | 把 $S$ 的纹理贴到 $C$ 的结构上 | 空间交叉注意力 |
| **训练稳定性** | 梯度不能爆炸/消失 | 无 ODE unrolling, 固定配对 |

---

## 二、我们的实现 vs 需要什么 — 逐项审计

### 缺陷 1: 时间风格纠缠

**需要**: Time 独立调制 ResBlock (AdaLN), Style 独立做 Cross-Attention

**我们做了**: `_compute_style_code` 把 time 和 style 加成一个 1D 向量 → `CrossAttnAdaGN` 同时用它做 AdaGN scale/shift (时间调制) 和 Cross-Attention K/V bias (风格调制)

```python
# model.py:1459-1460 — 致命加法
time_code = self.time_mlp(sinusoidal_time_embedding(t, ...))
return style_code + time_code  # 混合, 不可分
```

```python
# lancet_blocks.py:128 — 下游同时接收混合向量
scale, shift = self.global_proj(style_code).chunk(2, dim=1)  # 时间+风格混在同一个 code 里
```

**后果**: 模型无法区分"t 改变了"还是"风格改变了". 下游的所有调制都是混合信号.

**应改为**:
```python
time_code = self.time_mlp(...)   # 独立时间编码
style_code = self.encode_style_id(...)  # 独立风格编码
# AdaGN 只接收 time_code
# Cross-Attention K/V 只接收 style_code
```

---

### 缺陷 2: 伪交叉注意力

**需要**: Content Q 关注 Style 图像的空间特征图 → 每个像素从风格图中"复制"最匹配的纹理

**我们做了**: 全局可学习 tokens + 1D style_code 偏移 → 作为 K, V

```python
# lancet_blocks.py:130-131
style_bias = self.style_proj(style_code).unsqueeze(1)       # 1D 向量 → 偏移
style_tokens = self.style_tokens_basis.unsqueeze(0) + style_bias  # 全局 tokens + 偏移
k = self.k_proj(style_tokens)  # 从学习参数投影
v = self.v_proj(style_tokens)  # 从学习参数投影
```

**后果**: attention 关注的不是风格图像的空间特征, 而是一组固定的学习 token. 这是"查表 2.0", 不是真正的 cross-attention.

**应改为**:
```python
# 从实际风格图像编码空间特征图
style_feat = self.style_encoder(style_image)  # [B, C, H, W]
style_feat_flat = style_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
k = self.k_proj(style_feat_flat)
v = self.v_proj(style_feat_flat)
```

---

### 缺陷 3: 闭集查表

**需要**: 从任意输入图像中提取风格 → 泛化到未见风格

**我们做了**: `nn.Embedding(num_styles, D)` — 查表

```python
# semantic_tokenizer.py:283
self.style_values = nn.Embedding(self.num_styles, self.num_clusters * self.spatial_dim)
```

**后果**: 只能记忆训练集里的 5 种风格. 没有风格泛化能力. 即使训练集内, 也只有 5 个固定向量 — 没有内容适应性.

**应改为**: 从风格参考图 (或 matched_target) 中实时编码风格特征.

---

### 缺陷 4: Minibatch OT 不稳定

**需要**: 稳定的 (content, target) 配对 → 速度场学习目标一致

**我们做了**: 每 batch 内动态做 OT 匹配

**后果**: 同一张内容图在不同 epoch 匹配到不同目标 → 速度场学习目标反复跳变 → 模型输出"平均"色块

**应改为**: 
- 离线预计算配对 (fixed pairing per epoch)
- 或使用 Independent Coupling (不做 OT, 让模型自主学习)
- 或跨 batch 匹配 (更大的候选池)

---

### 缺陷 5: 训练中 ODE 展开

**需要**: 训练时只做单步预测, 推理时再积分

**我们做了**: `_terminal_swd` 调用 `model.integrate()` **在 autograd 内**

```python
# losses.py:2082 — 训练时展开 ODE
endpoint = model.integrate(content, style_id=..., num_steps=self.terminal_num_steps)
```

**后果**: 梯度流经 `num_steps` 层展开的 ODE → 爆炸/消失 → 大量 clamp/nan_to_num 掩盖 → 有效风格梯度消失

**应改为**:
```python
# 用单步预测, 不用 ODE 展开
pred_endpoint = model.predict_transport_base(x_t, t=t, style_id=style_id)
loss += SWD(pred_endpoint, target)
```

---

## 三、修复优先级

| 优先级 | 缺陷 | 修复 | 预期收益 |
|:---:|------|------|:---:|
| 1 | 时间风格纠缠 | 分开 time/style 注入路径 | 训练稳定, style 可学 |
| 2 | 训练中 ODE 展开 | 用单步预测算 SWD | 消除梯度爆炸, 释放 style 梯度 |
| 3 | 伪交叉注意力 | 用真实风格特征做 K,V | 风格细节可学 |
| 4 | Minibatch OT | 离线配对或 Independent Coupling | 目标稳定 |
| 5 | 闭集查表 | 从 matched_target 编码 | 泛化能力 |

---

## 四、如果从头实现 — 最小可行产品

一个干净的风格迁移模型只需要:

```
Content Image → Content Encoder → spatial features ──┐
                                                      ├→ Cross-Attention → Decoder → Output
Style Image  → Style Encoder  → spatial features ────┘
                                        ↑
                                  独立的 time modulation
                                  (AdaLN on ResBlocks)
```

**训练**: 单步预测. $L = MSE(pred\_x1, matched\_target) + SWD(pred\_x1, style\_distribution)$
**推理**: ODE/SDE 积分.

没有 tokenizer, 没有 minibatch OT, 没有 ODE unrolling.
