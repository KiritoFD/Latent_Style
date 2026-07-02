# 外部方法可复现性评估 — WikiArt 5×5×30 测试 + 本地 GPU

> 评估目标: 哪些方法能在我们的 WikiArt distinct5 测试集 (5 风格 × 每风格 30 张 × 5×5 全配对 = 750 图) 上评估，
> 并且能在本地 RTX 3060/4070 (≤12GB VRAM) 上复现。

---

## 可以直接评估 + 复现

### StyleGallery (CVPR 2026) ★★★★
- **类型**: Training-free, 基于预训练扩散模型 (SDXL/SD3)
- **评估**: 可以用任意参考图像 → 直接跑 WikiArt 750
- **VRAM**: ~8-12GB (SDXL base), 3060 可跑
- **代码**: 已公开
- **相关度**: 极高 — 同样做无配对风格迁移，同样不依赖外部标注

### HAM — Heterogeneous Attention Modulation (CVPR 2026)
- **类型**: Training-free, 调制扩散模型注意力
- **评估**: 可以用任意参考图像
- **VRAM**: ~8-12GB
- **相关度**: 高 — 注意力机制设计, 与我们 TopoGate 有理论对话空间

### CSGO (NeurIPS 2025) ★★★★★
- **类型**: 端到端可训练, 统一框架
- **数据集**: IMAGStyle (210K 三元组), 但也可以用 WikiArt 训练
- **VRAM**: 需训练 → 可能较重
- **代码**: 已公开
- **相关度**: 极高 — 同样做多风格统一框架, 对比价值大

### SaMST (arXiv 2025) ★★★★★
- **类型**: 已是我们 baseline! 解码风格建模和迁移
- **评估**: 已有 e5/e15 的 WikiArt 750 结果
- **状态**: 已在基线表中

---

## 可以参考设计但复现难度大

### StyleShot (ICLR 2025) ★★★
- **亮点**: Style-aware encoder 设计, 无需测试时调整
- **障碍**: 训练数据需求大 (开放域), 可能需要 LAION 预训练
- **参考价值**: tokenizer 设计思路 — 风格编码器的分离设计

### SCSA — Semantic Continuous-Sparse Attention (CVPR 2025 Highlight) ★★★
- **亮点**: 解决区域风格不一致, 即插即用
- **障碍**: 需要语义分割 mask (外部依赖)
- **参考价值**: attention 稀疏化设计 — 可能与 TopoGate 互补

### Attention Distillation (CVPR 2025) ★★
- **亮点**: 用预训练扩散模型的自注意力做知识蒸馏
- **障碍**: 需要特定扩散模型作为 teacher
- **参考价值**: 中 — 知识蒸馏范式不直接适用

---

## 不适用

| 方法 | 原因 |
|------|------|
| Style Nursing | 专用交通场景数据集 STREET-6K |
| Hairstyle Transfer | 专用人像/发型 |
| SigStyle | 依赖 DreamBooth 微调的个人化 T2I |
| Scheduled Style Injection | NTIRE 研讨会, 偏竞赛方法 |

---

## 推荐行动

### 立即可做的 (零到一天)

1. **跑 StyleGallery 在 WikiArt 750 上** — 训练无关, 直接用预训练模型推理
2. **跑 HAM 在 WikiArt 750 上** — 同上
3. **对比结果写入 all_experiments.csv**

### 中期 (有代码后)

4. **复现 CSGO 训练** — 需要看代码和技术报告, 评估训练成本
5. **读 StyleShot 的 encoder 设计** — 可能启发出更好的 tokenizer

### 论文相关

- **StyleGallery + HAM**: 作为最近的训练无关 baseline, 适合放在 Related Work 和实验表中
- **CSGO**: 作为端到端多风格框架的最强对比
- **SCSA**: attention 设计参考 (持续-稀疏 attention)
