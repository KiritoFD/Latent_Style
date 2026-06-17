# 618 外部方法设计启发

> 阅读这些论文/代码的目标: 找到能反哺我们模型设计的思想, 不是单纯复现.

---

## 一、StyleGallery (CVPR 2026) — 语义区域分割 + 聚类匹配

### 核心机制

1. **语义区域分割**: 用 DINO/SAM 将内容图和风格图分割成语义区域 (天空/建筑/水面...)
2. **聚类区域匹配**: 对每个语义区域, 在目标风格的对应区域中找最匹配的 patch
3. **区域级风格注入**: 只在匹配的区域注入风格, 跨语义区域的风格不混合

### 对我们的启发

**我们已经在做类似的事情**: Tokenizer 的 $K$ 个 cluster → 空间路由 → 每个 cluster 对应一种"语义-风格"配对。
TopoGate 进一步保证信息只在内容 self-attention 通路中流动。

**StyleGallery 比我们多做的**: 
- 使用外部的 DINO/SAM 做语义分割 (我们有 TopoGate 的内生 attention, 不需要外部模型)
- 显式的区域匹配 (我们有 OT 隐式匹配)

**可以借鉴的**:
- **区域级 SWD**: 把我们的 fiber-wise SWD 从 "per cluster" 改为 "per spatial region"。
  用 TopoGate 的 attention map 做 soft mask, 按区域分别算 SWD.
  这和我们的 fiber-wise SWD 本质上一样, 只是 mask 的来源不同.

---

## 二、HAM — Heterogeneous Attention Modulation (CVPR 2026)

### 核心机制

在扩散模型的不同层 (cross-attn, self-attn) 用不同的策略注入风格:
- Cross-attention 层: 替换 K, V 为风格图像的 K, V
- Self-attention 层: 共享内容图的 Q, K, 用风格 V

### 对我们的启发

**我们已经在做类似的事情**: TopoGate 的 $A_{\text{final}} = \alpha A_{\text{self-content}} + (1-\alpha) A_{\text{cross}}$ 就是一个 attention 调制策略.

**HAM 比我们多做的**:
- 对不同层 (不同分辨率) 使用不同的调制策略
- 对 cross-attn 和 self-attn 用不同的注入方式

**可以借鉴的**:
- **多尺度 TopoGate**: 当前 TopoGate 在所有层用同一个 blend。
  HAM 的启示: 不同层用不同的 blend。低层 (high-res, 细节) 用较小的 blend (更依赖内容 self-attn);
  高层 (low-res, 全局) 用较大的 blend (更多风格注入).
  这和我们在 616 讨论的"多尺度 TopoGate"一致.

---

## 三、SCSA — Semantic Continuous-Sparse Attention (CVPR 2025 Highlight)

### 核心机制

解决语义风格迁移中"区域风格不一致"的问题:
- 引入 continuous-sparse attention: 每个 query 只关注语义相似的 key
- 使用语义 mask 来稀疏化 attention

### 对我们的启发

**我们已经在做类似的事情**: TopoGate 的 self-attention blending 天然限制了跨语义区域的信息混合.

**SCSA 比我们多做的**:
- 显式的 sparse attention (hard mask)
- 需要外部语义分割

**可以借鉴的**:
- **Attention 稀疏化**: 在 TopoGate 基础上, 进一步对 attention matrix 做 top-k 稀疏化.
  即每个 query 只关注 attention score 最高的 k 个 key.
  这能进一步减少"错误的风格信息流入错误区域"的问题.
  实现极简单: `attn = topk_mask(attn, k=8) / attn.sum(dim=-1, keepdim=True)`.

---

## 四、CSGO (NeurIPS 2025) — 统一框架 + 大规模三元组数据

### 核心机制

1. **三元组训练**: (content, style_ref, ground_truth) 三重监督
2. **IMAGStyle 数据集**: 210K 三元组, 覆盖多种风格
3. **统一框架**: 同时支持多种风格迁移范式

### 对我们的启发

**CSGO 和我们做的不一样的地方**:
- 我们做的无配对 (unpaired) 风格迁移, CSGO 需要配对的三元组数据
- CSGO 强调数据集规模 (210K), 我们只有 18K (WikiArt full)

**可以借鉴的**:
- **ground truth 监督**: 虽然我们的设定是 unpaired, 但可以考虑自监督的 pseudo-GT。
  比如: 用当前模型风格化一张图 → 把结果作为下一轮训练的 pseudo-GT → 迭代.
  类似 self-training 或 knowledge distillation.

- **多风格统一训练**: CSGO 同时训练多个风格, 共享 backbone。
  我们也在做 (5 类 WikiArt 一起训), 但 tokenizer 的 style-specific 部分是独立的.

---

## 五、StyleShot (ICLR 2025) — Style-Aware Encoder

### 核心机制

1. **Style-Aware Encoder**: 将风格图像编码为 style code, 与内容解耦
2. **Content Fusion Encoder**: 将 content 特征和 style code 融合
3. **不需要测试时微调** (test-time tuning free)

### 对我们的启发

**我们 tokenizer 的问题**: PureLatentSpatial 试图从 content latent 同时提取 content query 和 style value.
这两个任务是矛盾的——content query 应该编码结构布局, style value 应该编码纹理笔触.
当 tokenizer 不够好时, 两者混淆.

**StyleShot 的设计**:
- Style-Aware Encoder **只从风格图像中提取 style code** — 不碰 content
- Content Fusion Encoder 负责融合

**可以借鉴的**:
- **分离的 style code 提取**: 让 tokenizer 从 TARGET style images (而不是 content latent) 中提取 style values.
  当前 tokenizer 的 `style_values` 是纯 style_id lookup, 没有用到目标风格图像的实际特征.
  
  改进: 在 training 的 OT 匹配后, 对 matched_target 做额外的 style encoding,
  而不是只用 style_id 的 embedding lookup.

---

## 六、可立即落地到我们代码的改动

| 优先级 | 来源 | 改动 | 代码位置 | 预期效果 |
|:---:|------|------|----------|------|
| 1 | HAM + 我们已有的计划 | 多尺度 TopoGate blend | `lancet_blocks.py` | LPIPS 不变, style +0.01 |
| 2 | SCSA | TopoGate attention top-k 稀疏化 | `lancet_blocks.py` | 减少跨区域风格泄漏 |
| 3 | StyleGallery | 区域级 fiber-wise SWD (已部分实现) | `losses.py` | 风格特异性 |
| 4 | StyleShot | 从 matched_target 编码 style code | `semantic_tokenizer.py` | 更准确的风格表征 |

---

## 七、读论文方法建议

不要从头读到尾. 每个论文看三个部分:
1. **Method 的图** (architecture diagram) — 数据流怎么走
2. **核心公式** (1-2 个) — 数学本质是什么
3. **和我们的方法的对比** (上面的表格) — 我们已经在做什么, 缺什么
