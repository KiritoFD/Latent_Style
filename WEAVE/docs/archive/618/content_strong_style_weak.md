# Content强Style弱 — 根因分析与突破方案

> 基于最新实验: 7组全部 style 0.66-0.67, LPIPS 0.29-0.31

---

## 一、实验结果全景

| 实验 | epoch | trans style | trans LPIPS | all-pairs style | 机制 |
|------|:---:|:---:|:---:|:---:|------|
| h0 | e17 | 0.666 | 0.300 | 0.700 | 垂直FM基线 |
| h1 | e20 | **0.670** | **0.291** | **0.709** | 线性FM |
| h2 | e6 | 0.663 | 0.306 | 0.694 | 欧氏OT |
| h3 | e6 | 0.663 | 0.295 | 0.705 | SDE噪声 |
| h4 | e7 | 0.663 | 0.308 | 0.692 | 非平衡OT |
| h5 | e6 | 0.664 | 0.298 | 0.702 | TopoGate attention OT |
| h6 | e6 | 0.663 | 0.298 | 0.702 | 全组合 |
| abl | e6 | 0.664 | 0.292 | 0.708 | 线性FM消融 |

### 关键观察

1. **所有7个实验style都卡在0.66-0.67** — 无一突破0.70
2. **LPIPS极好 (0.29-0.31)** — 比之前的topogate_appalign (0.31-0.33) 更低了
3. **垂直FM没有帮助** — H0 (vertical, 0.666) vs H1 (linear, 0.670). 线性FM反而更好
4. **结构OT没有帮助** — 所有结构OT变体与欧氏OT几乎一样
5. **SDE噪声没有帮助** — sigma=0.02 与 sigma=0 无差异
6. **非平衡OT没有帮助** — 哑类匹配无增益
7. **H1 (线性FM) 表现最好** — e20时 style 0.670, 是最高的; LPIPS 0.291也是最好的

---

## 二、根因诊断

### 根因: TopoGate blend=1.0 过度锁死结构

当前配置: `semantic_self_topology_blend=1.0` — 100% content self-attention, 0% cross-attention。

这意味着UNet的attention层**根本不看style特征**。TopoGate把attention完全锁死在content self-attention → 每个像素只看自己周围的content pixels → style信息在attention层面被完全阻隔。

**实验证据**:
- LPIPS 0.29-0.30 — 结构完美保持 (好到几乎和IDT一样)
- Style 0.66-0.67 — 所有机制都无法提升
- H1 (线性FM, blend=1.0) 比 H0 (垂直FM, blend=1.0) 更好 — 因为在blend=1.0下, 垂直FM的"结构锁定"是冗余的, 而线性FM还多了一点style自由度

### 假说: TopoGate blend 是唯一需要调的参数

如果 blend=1.0 把attention完全锁死, 那么不管改什么 (垂直FM, OT, SDE, 非平衡) 都没用——因为style信号根本进不去。

**预测**: 如果把blend降到0.3-0.5, style会突然突破。LPIPS会微升到0.33-0.37, 但仍可接受。

### 根因2: legacy tokenizer的style_values是纯查表

当前用`legacy_factorized` tokenizer — style values是embedding lookup (style_id → fixed vector)。没有从参考图中编码风格信息的能力。

配合blend=1.0 → **style信号来源两处都被堵死了**:
1. Attention层面: TopoGate锁死→ style cross-attention被阻断
2. Tokenizer层面: style values是固定查表→ 没有实例级风格信息

---

## 三、解决方案

### 方案A: 降低TopoGate blend (零代码, 立即见效)

```json
{"model": {"semantic_self_topology_blend": 0.4}}
```

**预期**: style 从0.67突然跳到0.70-0.72, LPIPS从0.29升到0.33-0.36

**扫描**: blend ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8}

### 方案B: 多尺度TopoGate (少量代码)

不同UNet层用不同blend:
```json
{"model": {"semantic_self_topology_blend_per_scale": {"8": 1.0, "16": 0.5, "32": 0.3, "64": 0.2}}}
```
粗尺度锁死 (保大局结构), 细尺度放松 (允许笔触变化)。

### 方案C: 从matched_target编码style (代码改动)

在OT匹配后, 用一个轻量encoder从matched_target提取style features → 注入tokenizer。这样style values不再是查表, 而是从实际风格图像中编码。

### 方案D: 逐步解锁训练

```
epoch 1-3:  blend=1.0 (学结构)
epoch 4-6:  blend=0.7
epoch 7-12: blend=0.4 (释放style)
epoch 13+:  blend=0.2
```

### 推荐执行顺序

**今天**: 方案A — 改一个参数, 立即跑blend sweep (6 values × ~30min = 3h)
**如果A突破**: 方案B多尺度blend
**如果A不够**: 方案D逐步解锁
**长期**: 方案C (需要代码改动)

---

## 四、与论文分析的呼应

- **SCSA启发**: 硬约束(G1/G2)比软约束更精确, 但我们的blend=1.0是"最硬的约束"→ 太强了
- **StyleShot启发**: 风格表征是充分必要条件 — 我们的tokenizer查表 + blend=1.0 → 风格表征接近零
- **任务差异**: 无参考图设定下, 风格表征来自tokenizer. 如果tokenizer被压制, 没有其他来源
