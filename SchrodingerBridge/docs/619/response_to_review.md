# 619: 外部审查的深度回应 — 方案修正

> 外部审查指出预匹配 OT 方案中"Sinkhorn 重排 VAE latent"是致命错误.
> 同时指出: 如果有真正的 Cross-Attention, 离线 OT 对齐是冗余的.
> 这两个判断都是对的.

---

## 一、致命错误: VAE latent 离散重排

### 为什么不行

VAE 的卷积 decoder 假设输入 latent 在空间上是光滑的. 相邻 pixel 来自原始图像的相邻区域.
如果用 Sinkhorn plan 把 pixel 从不同位置"飞"过来 → 空间连续性被破坏 → decoder 输出 checkerboard artifacts.

### 修正

**不做 latent 重排**. 预处理的唯一输出: **弱语义配对列表** (content_path, style_path), 不需要对齐.

---

## 二、冗余: 离线 OT vs Cross-Attention

### 为什么冗余

Cross-Attention 的公式本身就是软 OT:
$$A = \text{softmax}(Q_{\text{content}} K_{\text{style}}^T / \sqrt{d})$$
$$Output = A \times V_{\text{style}}$$

这等价于在特征空间做了一次 Sinkhorn 式的"软分配". 离线 OT 是硬对齐, Cross-Attention 是软对齐. 后者更灵活, 且随训练优化.

**结论**: 如果实现了真正的 Cross-Attention, 离线 OT 对齐不仅冗余, 而且**阻碍**注意力学习最优匹配.

---

## 三、修正后的黄金架构

```
                    ┌──→ AdaLN (独立 time 调制)
Content Image → Encoder → UNet Blocks → Decoder → Output
                              ↑
Style Image → StyleEncoder → Cross-Attention K,V (独立 style 注入)
                              ↑
                         Independent Coupling:
                         z_t = (1-t)*z_c + t*z_s
                         Loss = MSE(v_pred, z_s - z_c)
```

### 预处理
- DINOv2/CLIP 计算弱语义配对 → 保存配对列表
- **不做 latent 重排**

### 模型架构
- **Time**: AdaLN-Zero (DiT/SD3 范式) — 每个 ResBlock 前用 `time_mlp(t) → scale, shift, gate`
- **Style**: StyleEncoder (DINOv2) 提取风格图像的空间特征 → Cross-Attention K,V
- **结构保持**: 不显式约束 — 依赖模型从 Flow Matching 的线性路径中学结构保持

### 训练
```python
z_t = (1 - t) * z_c + t * z_s       # Independent Coupling
v_pred = model(z_t, t, style_img)   # Cross-Attention sees style image
loss = MSE(v_pred, z_s - z_c)       # Simple FM loss
```

### 需要实验确定的三项

| 问题 | 选项 | 推荐 |
|------|------|:---:|
| Coupling | A: Independent B: Paired C: Score-matching | A (简单, FM理论保证) |
| Style注入 | A: Cross-Attention (空间) B: AdaLN (全局) | A (能学到笔触细节) |
| Time注入 | A: AdaLN-Zero B: Channel concat | A (DiT/SD3验证) |

### 可选扩展
如果 Independent Coupling 下结构保持不够:
```python
loss += w_content * L1(pred_x1, content)  # 单步预测的结构约束
```

---

## 四、与当前代码的差距

当前代码需要**完全重写**以下模块才能达到黄金架构:

| 模块 | 现状 | 需改为 |
|------|------|--------|
| `_compute_style_code` | time+style 混合 1D | time 独立 AdaLN, style 独立 Cross-Attn |
| `CrossAttnAdaGN` | learned tokens + 1D bias | 真实风格图像空间特征 |
| `_terminal_swd` | ODE展开 | 移除或改为单步预测 |
| `_ot_match_targets` | minibatch Sinkhorn | 移除或离线弱配对 |
| Tokenizer | Embedding lookup | 移除或替换为 StyleEncoder |
