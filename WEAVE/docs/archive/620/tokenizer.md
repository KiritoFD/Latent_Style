# tokenizer.md — 风格表征层设计: 从查表到空间交叉注意力

> 承接 math.md C2+C5. 本文件回答: **给定一张目标风格图, 模型如何"看见"它的笔触**.
> 仓库已有 5 种 tokenizer 设计, 全部被 DPI 上限定理 (math.md §4) 判决.
> 本文给出以 True Cross-Attention 为核心的新 tokenizer 设计.

---

## 1. 历史路线全表 (含数学上限)

来自 `style_tokenizer.py`, `semantic_tokenizer.py`, `docs/616`, `docs/618/why_style_weak.md`, `docs/612-lookback/analysis.md`:

| Tokenizer 家族 | $C_s$ 维度 | 信息量 | DPI 上限 | 实测 style | 实测 LPIPS | 状态 |
|----------------|-----------|--------|---------|-----------|-----------|------|
| `legacy_factorized` (5×Embedding(256)) | 256 | <1 KB | 极低 | 0.67 (618 全 7 组) | 0.29-0.30 | 主线, 已锁死 |
| `direct_atom_residual` (concept atoms) | ~256 | <1 KB | 极低 | 0.789 (WikiArt512) | 0.360 | 在大数据集上能跑; 跨任务差 |
| `concept_atoms` / `global_vq` | ~256 | ~1 KB | 极低 | 同上 | 同上 | 与 direct_atom_residual 等价 |
| `PureLatentSpatialTokenizer` (16 spatial) | ~2K | ~8 KB | 低 | ZERO ROI (不变) | 不变 | **已退役**, 白耗 1.2GB VRAM |
| `SMoETranslatorTokenizer` (32 expert×W∈R^{D×D}) | ~2K effective | ~8 KB | 低 | 未完成 | 未完成 | 612-phase2 提案, 未跑通 |
| `AffineConnectionTokenizer` (γ/β per style+cluster) | ~1K effective | ~4 KB | 低 | 未实现 | 未实现 | 616/design §3 提案 |

**核心事实**: 所有"在 latent 内部自组织"路线都没有突破 DPI 上限. 它们的 $I(S;C_s)$ 量级都在 KB 级, 远小于风格图本身的 ~400KB.

> 619/model/04 用 DPI 解释这点是完全正确的数学.
> 612/plan-612.md 的"自组织语义字典"叙事被仓库实验 ZERO ROI 证伪.

---

## 2. 第一性结论: 风格表征必须走空间序列

由 math.md §4 的 DPI 不等式:
$$I(S;Y) \le I(C_s;Y) \le I(S;C_s).$$

要 $Y$ (生成图) 包含 $S$ (风格图) 的纹理信息, 必须 $I(S;C_s)\!\sim\!$ 风格图本身信息 (几百 KB).
唯一在计算上可达的途径: **$C_s$ 直接是 $S$ 的预训练 encoder 空间特征序列** $F_s\in\mathbb{R}^{N\times D}$, $N\!\approx\!256, D\!\approx\!384$.

→ math.md C2 的具体形式就是 True Cross-Attention.

---

## 3. 选定方案: 真空间交叉注意力 (True Spatial Cross-Attention)

### 3.1 数据通路

```
Style Image (3×512²)
    │
    ▼
[Frozen Encoder] (DINOv2 vit_small_patch14, 中间层)
    │
    ▼
F_s ∈ R^{256 × 384}        (16×16 patch token 序列)
    │
    ▼ (训练时冻结, 推理时按需重算)
proj_d → R^{256 × D_model}  (投影到 UNet 维度)
    │
    ▼ 注入主干
[Cross-Attention K, V]
```

### 3.2 Cross-Attention 的数学

在 UNet 每个 decoder block (或 DiT block):
$$Q = W_Q\,h, \quad K = W_K\,F_s, \quad V = W_V\,F_s,$$
$$\text{Attn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V \;\in\;\mathbb{R}^{HW\times D}.$$

**关键性质**:
- Attention map $A = \text{softmax}(QK^\top/\sqrt d)$ 是**可微的 Kantorovich 软传输计划** (619/model/01 §2.3 已指出). 每个内容位置 $h_{hw}$ 可以软寻址**任意** $F_s$ 中的风格 token.
- 信息通量 $\sim HW\times D$ (不是 1D 向量), 突破 DPI 瓶颈.
- 它本身是空间感知的 — 内容图左侧的 query 会自动关注风格图左侧相似的 token, 这是 latent Euclidean OT 永远做不到的.

### 3.3 为什么选 DINOv2 而不是 CLIP

| 选项 | 优势 | 劣势 |
|------|------|------|
| Frozen DINOv2 vits14 | 强语义+空间结构, 训练快(算力省) | 极高频笔触表征略弱 |
| Frozen CLIP image encoder | 语义强 + 语言对齐 | StyleShot 论文证明 CLIP 风格表征弱 |
| 可训练 ResNet/ConvNeXt | 高频细节捕获强 | 小数据集易过拟合, 泛化差 |
| 不用 encoder (旧 latent 路线) | 0 外部依赖 | DPI 死锁, 实测 ZERO ROI |

→ 619/model/05 的方案 A (Frozen DINOv2) 是正确的起点. 训练时计算量极低 (frozen), 推理时每张图一次 DINOv2 forward (<10ms on 3060).

### 3.4 与 619/system_diagnosis "缺陷 2 / 3" 的精确对接

诊断列了"伪交叉注意力"和"闭集查表"两个缺陷:
- **伪交叉注意力**: 当前 `lancet_blocks.py:130-131` 的 `style_tokens_basis + style_bias` 是**全局可学习 token + 1D bias** → 等价于"色彩调制" → DPI 瓶颈.
- **闭集查表**: `nn.Embedding(5, 256)` 退化 → 完全无泛化.

本方案的 Cross-Attention 输入是真实空间特征序列, 这两个缺陷同时被消除. **这是数学要求, 不是工程偏好**.

---

## 4. 残余设计问题: 是否还需要"tokenizer" 这个词

历史 tokenizer 的产物有两个:
1. `global_code`: 1D 风格向量 → 注入 AdaGN scale/shift.
2. `spatial_map`: [B, C, H, W] 风格空间图 → 加在 UNet 中间特征上.

在新方案下:
- `global_code` 完全由 DINOv2 CLS token + 一个 MLP 生成 (单实例全局风格摘要). 这是**可选**的, 主要用于全局色调匹配.
- `spatial_map` 的功能被 Cross-Attention 直接吸收 — 不再需要显式生成, 因为 K/V 注入本身就在做空间对齐.

→ **新架构没有独立 tokenizer 模块. 风格表征的"翻译"功能合进 Cross-Attention 的 $W_K, W_V$ 投影矩阵里**.
这和 619/model/05 的"模块解耦"叙事不同 — 我们更激进地认为"tokenizer" 这个抽象本身就是错误.

### 4.1 可选保留: SMoE-style 多专家投影 (容量扩展)

如果纯 Cross-Attention 在 8 epoch 后 style 仍卡 (< 0.72), 引入 mixture-of-experts 的 K/V 投影:
$$K = \sum_{k=1}^K \alpha_k(F_s)\,W_K^{(k)} F_s,\quad V = \sum_{k=1}^K \alpha_k(F_s)\,W_V^{(k)} F_s.$$
- 每个专家 $(W_K^{(k)}, W_V^{(k)})$ 学一种"如何从 DINO 特征中提取笔触类型".
- $\alpha_k$ 是 router (类似 PureLatentSpatial 的 routing), 基于 $F_s$ 全局池化.
- 这是 612-phase2/SMoE 提案与 616/AffineConnectionTokenizer 的"翻译"思想在 Cross-Attention 框架下的兼容版本.

但**第一轮不启用**. 先看纯 Cross-Attention 能跑到哪儿, 再决定是否上 MoE.

---

## 5. 时空解耦的精确实现 (math.md C5)

主干 block 前向:
```python
def block_forward(h, t_emb, F_s):
    # === C5a: Time → AdaLN, 只有 AdaLN 看到 t_emb ===
    scale_t, shift_t, gate_t = self.adaln_mlp(t_emb).chunk(3, dim=-1)
    h = h * (1 + scale_t[:, :, None, None]) + shift_t[:, :, None, None]
    
    # === Self-Attention (内容自拓扑, 不带外部条件) ===
    h_res = h
    h_sa = self.self_attn(h)
    h = h + gate_t[:, :, None, None] * h_sa
    
    # === C5b: Style → Cross-Attention, 只有 CA 看到 F_s ===
    Q = self.q_proj(h.flatten(2).transpose(1, 2))     # [B, HW, D]
    K = self.k_proj(F_s)                                # [B, N, D]
    V = self.v_proj(F_s)                                # [B, N, D]
    A = softmax(Q @ K.transpose(-2, -1) / sqrt(D))
    h_ca = (A @ V).transpose(1, 2).view(B, D, H, W)
    
    # === Style gate (可选, zero-init 让训练初期纯 self-attn) ===
    h = h + self.style_gate * h_ca
    
    # === FFN + Residual ===
    h = h + self.ffn(h)
    return h
```

关键点对照 math.md:
- **C5** time/style 参数 disjoint: `adaln_mlp` 只含 time 参数, `k_proj/v_proj` 只含 style 参数. $\nabla_{\text{ada}}\mathcal{L}$ 与 $\nabla_{\text{ca}}\mathcal{L}$ 在 disjoint 矩阵上.
- **C2** $I(S;C_s)$ 由 $F_s$ 直接保证 (DINO 输出, 不被 1D 瓶颈).
- **zero-init style_gate**: 训练开始时模型接近"无 style 注入", 防止随机 cross-attention 把训练初期搞乱. `style_gate` 的 L2 梯度自然推动它在第一次 SWD 信号到来时缓慢打开.

### 5.1 与 616/TopoGate 的关系

TopoGate 在 612-phase2 的实测中 (618/why_style_weak.md) 是"weak runtime lever" — `plain_forward_delta` 只有 1e-3, 没达到"完美结构保护"的承诺.

新方案下 TopoGate 可选保留, 作为 self-attention 的混合门控:
$$A_{\text{final}} = \alpha\,A_{\text{self-content}} + (1-\alpha)\,A_{\text{self-current}}.$$

但**第一轮不启用**. 在新框架下, 结构保护主要靠:
1. vertical FM (math.md C3) — 训练目标侧的硬约束.
2. velocity 残差性质 ($z_1 = z_0 + \int v$) — 推理侧自然锚定 $z_0$.
3. Cross-Attention 的软约束 ($Q\!=\!h$ 自带内容拓扑信号).

TopoGate 留作 Phase B 的可选增量.

---

## 6. 是否需要 StyleEncoder 训练 (待 A/B)

| 选项 | encoder | 总训练参数 | 风险 |
|------|---------|-----------|------|
| A (第一轮): DINOv2 frozen | 0 (encoder) | 大幅减少 | 高频纹理可能不足 |
| B (升级): DINOv2 + 可训练 adapter | ~1M (adapter) | 微调 | 训练复杂度略增 |
| C (备选): 可训练轻量 CNN | ~11M | 中等 | 小数据集过拟合 |

A→B 的过渡路径: 把 DINOv2 输出过一层 1×1 卷积 + 小 MLP (`adapter`), 让网络对 DINO 特征做"风格化线性变换". 这是 616/AffineConnectionTokenizer 思想的轻量版, 但作用在**DINO 特征而不是 latent F_s 的高频残差**上 (避开 latent 切分问题).

第一轮跑 A. style 上限到达 0.72+ 就停; 不到 0.72 升级到 B.

---

## 7. 评估指标

- `clip_style` (transfer): 第一轮目标 ≥ 0.72.
- `clip_style` (all-pairs): 第一轮目标 ≥ 0.72.
- `LPIPS`: 保持 < 0.40 (vertical FM 应该天然守住).
- `cross_attn_entropy`: 监控 attention map 是否塌缩为常数 (高 entropy=均衡用所有风格 token, 低 entropy=只盯少数). 健康: 4.5–6.0 (log domain).
- `style_gate_value`: 应在 5-10 epoch 内从 0 涨到 0.05–0.2 量级.

如果 `cross_attn_entropy` 早期 <2.0 (塌缩), 说明 router 失效 → 升级 §4.1 MoE.
如果 `style_gate_value` 卡在 <0.01, 说明单步 MSE 梯度对 style 路径无驱动力 → 检查 OT.md 是否生效 (即 target stability).

---

## 8. 与历史文档的精确对接

参考并采纳:
- 619/model/01 §2.1 (闭集查表=纤维压缩点): ✅ 保留诊断.
- 619/model/01 §2.3 (伪交叉注意力=均匀传输计划): ✅ 保留诊断.
- 619/model/04 §3 (DPI 信息瓶颈): ✅ 直接作为本方案数学基础.
- 619/model/05 §1+§2 (StyleEncoder 蓝图): ✅ 框架采纳, 细节简化为单个 frozen DINOv2.
- 619/model/06 §2 (Multi-scale SMoE Tokenizer): ⚠️ 不采纳为初始方案, 作为 §4.1 的 capacity 升级手段保留.

不采纳:
- 612/plan-612.md PureLatentSpatialTokenizer: ❌ 实测 ZERO ROI, 且本身违反 DPI.
- 616/design.md §3 AffineConnectionTokenizer: ❌ 作用在 latent 高频残差上, latent 切分本身有截断频率问题; 改造为作用在 DINO 特征上的 adapter (本方案 §6 B).
- 619/problem.md §2.2 "Latent 离散重组致命错误": ✅ 采纳诊断, 本方案不做任何 latent 重排.
- 619/solution.md §2 "交叉注意力=自发语义对应": ✅ 采纳, 这是 §3.2 的直接推论.

---

## 9. 实施步骤 (按优先级)

1. **写 StyleEncoder 包裹层** (`src/style_encoder.py`): frozen DINOv2 vits14 + 可选 adapter.
2. **替换 `_compute_style_code` 调用**: 删除 `style_code + time_code`, forward 签名改为 `forward(z_t, t, F_s)`.
3. **新 block 实现** (`src/blocks_fm.py`): 按 §5 模板, AdaLN(time) + CrossAttn(F_s).
4. **退役旧 tokenizer 抽象**: `style_tokenizer.py`, `semantic_tokenizer.py` 不删但不接入 forward.
5. **数据 pipeline 改造**: DataLoader 返回 `(z_c, z_s, style_image_rgb)`.
6. **Phase A smoke**: 5 epoch @ B=16 验证 attention 学得动.

第一轮若达标, 进入 bridge.md 的 SDE 设计; 若不达标, 升级 §4.1 MoE 或 §6 方案 B.
