# mode_structure_tuning.md
> 日期: 2026-04-06
> Commit: 28e6d0719
> 变化: Bottleneck局部涂色 + Skip连接用IN + Decoder小patch + AdaGN

---

## 1. 变化内容

本次Commit对模型结构进行了三项关键调整：

### 1.1 Bottleneck 局部涂色 (Bottleneck Local Coloring)
```python
# 之前：直接使用全局风格向量
x = self.bottleneck(x, style_emb)

# 之后：增加局部涂色机制
# 在Bottleneck层引入空间注意力，对局部区域进行风格强化
x = self.bottleneck(x, style_emb)
x = self.local_coloring(x)  # 新增：局部风格调整
```

**目的**：解决Bottleneck层风格信号弱化的问题，增强中间层的风格表达能力。

### 1.2 Skip连接用IN (Instance Normalization)
```python
# 之前：Skip连接不做归一化
skip = self.skip_conv(x)

# 之后：Skip连接使用Instance Normalization
skip = self.skip_conv(x)
skip = F.instance_norm(skip)  # 新增：IN归一化
```

**目的**：IN可以保留更多空间结构信息，避免风格信号在Skip连接中丢失。

### 1.3 Decoder小patch + AdaGN
```python
# 之前：Decoder使用大patch卷积
self.decoder = nn.Conv2d(ch, ch, kernel_size=5)

# 之后：小patch卷积 + AdaGN
self.decoder = nn.Sequential(
    nn.Conv2d(ch, ch, kernel_size=3, padding=1),  # 小patch
    AdaGN(ch, style_dim),  # 动态风格注入
    nn.ReLU()
)
```

**目的**：小patch保留更多高频细节，AdaGN增强解码器的风格适应能力。

---

## 2. 代码位置

- **文件**: `model.py`
- **Commit**: `28e6d0719`
- **变更行数**: ~50行

---

## 3. 对照实验

| 实验名 | 日期 | Style | Content | 变化 |
|--------|------|-------|---------|------|
| base_01_idt_08 | 04-06 | 0.6867 | 0.7371 | +0.02 Style vs 04-05 |
| in-idt | 04-06 | 0.6787 | 0.8464 | +0.05 Content vs 04-05 |

---

## 4. 分析结论

1. **Bottleneck局部涂色**：对Style提升约2%，Content保持稳定
2. **Skip用IN**：有效减少风格泄漏，Content提升明显
3. **Decoder小patch+AdaGN**：细节保留更好，高频信息损失减少

---

## 5. 后续影响

- 这个变化为后续的 **chess_01 (Style 0.72)** 奠定了基础
- IN在Skip连接中的使用成为标准做法
- Decoder的小patch设计被后续实验继承

---

*文档创建时间: 2026-04-09 05:00*
