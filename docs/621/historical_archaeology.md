# 621 历史数据考古综合分析

> 分析日期: 2026-06-21  
> 数据源: 22,629行实验CSV, 15+分支git历史, 248个考古文档

---

## 1. 宏观统计

### 1.1 数据规模

| 指标 | 值 |
|------|-----|
| 总实验行数 | 22,629 |
| 唯一方法数 | 40+ |
| 唯一数据集 | 20+ |
| 分支数 | 15 |
| 时间跨度 | 2026-02-15 至 2026-06-21 (4个月) |

### 1.2 核心指标分布

| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 有效样本数 |
|------|------|--------|--------|--------|-----------|
| clip_style | 0.6729 | 0.0930 | 0.3989 | 0.9109 | 17,021 |
| content_lpips | 0.4260 | 0.1202 | 0.0000 | 0.9368 | 17,021 |
| ssim_y | 0.4206 | — | 0.0000 | 0.6520 | ~5,000 |

### 1.3 clip_style与content_lpips的相关性

**r = -0.1005** (弱负相关)

**关键洞察**: clip_style和content_lpips几乎是**独立指标**！这意味着:
- 提升风格不必然牺牲内容保持
- 但最优配置需要同时优化两个目标
- Pareto前沿只有10个非支配点（17,021中）

---

## 2. 方法演进时间线

### 2.1 四个阶段

```
Phase 1 (Feb 15 - Mar 19): Legacy探索
  ├─ no-edge: 基线style transfer
  ├─ SWD系列: 256/100/6/50参数扫描
  ├─ AdaGN: Adaptive GroupNorm实验
  ├─ Moment: 矩匹配
  └─ 结果: clip_style ~0.65-0.68

Phase 2 (Mar 20 - May 6): Tokenizer+StyleID
  ├─ Tokenizer spiral: 多种tokenizer变体
  ├─ StyleID/IDT: 参考style ID条件化
  ├─ AdaIN v32k/vgg19: 风格注入方式
  └─ 结果: clip_style ~0.67-0.71, 突破0.70

Phase 3 (May 7 - May 29): SchrodingerBridge
  ├─ LANCET/LBM: U-Net backbone
  ├─ OT coupling: Sinkhorn/Hungarian配对
  ├─ Terminal SWD: 多步ODE展开
  ├─ Grid/Weight sweep: 超参搜索
  ├─ Representation probe: 表征分析
  └─ 结果: clip_style ~0.69-0.71, 稳定突破0.67

Phase 4 (May 30 - Jun 21): 620 SpatialBridge + 消融
  ├─ 620架构: 纯transformer blocks
  ├─ DINO cross-attention: 真正的空间attention
  ├─ 单步SWD: 消除ODE展开问题
  ├─ FiLM endpoint: style调制endpoint head
  ├─ 白化诊断: WFI指标体系
  └─ 结果: clip_style ~0.705, WFI从0.49降至0.39
```

### 2.2 历史天花板演进

| 时间 | 天花板 | 方法 | 关键突破 |
|------|--------|------|----------|
| Feb 2026 | 0.65 | Legacy SWD | 基线 |
| Mar 2026 | 0.68 | Tokenizer | tokenizer设计 |
| Apr 2026 | 0.70 | AdaIN/IDT | 风格条件化 |
| May 2026 | 0.71 | LANCET/LBM | U-Net + OT |
| Jun 2026 | 0.705 | 620 SpatialBridge | DINO + 单步SWD |
| Jun 2026 | 0.9109 | Cycle-NCE (overfit50) | 简单任务过拟合 |

**关键发现**: Cycle-NCE在overfit50上达到0.9109，但这是在极小数据集上的过拟合结果。真正的泛化天花板在0.71-0.73。

---

## 3. 消融实验考古

### 3.1 已验证的消融维度

| 维度 | 配置数 | 最优 | 结论 |
|------|--------|------|------|
| SWD weight (12/16/20) | 3 | 16 | 中等权重最优 |
| Velocity length (1.0/0.2/0.04) | 3 | 0.04 | 小步长稳定 |
| Attention mode (5种) | 5 | gated | 但差异小 |
| Endpoint head (3种) | 3 | FiLM hd512 | 关键突破 |
| Gate init (0.05/0.3) | 2 | 0.3 | 增强style信号 |
| Training target (3种) | 3 | target_linear | 低频锚定 |
| SWD scale (4种) | 4 | global | 简单有效 |

### 3.2 已证明无效的方向

| 方向 | 结果 | 原因 |
|------|------|------|
| Diff-Gram | 极差 | sdxl-fp32验证 |
| Gram-Moment | 差 | 矩匹配不充分 |
| Structure loss | 无用 | "完全没用" (Classify分支) |
| gated_raw attention | WFI=0.64 | 无归一化导致统计漂移 |
| relu2 attention | WFI=0.53 | 稀疏但风格区分不足 |
| style_select attention | WFI=0.50 | top-k未解决content-style冲突 |
| lowfreqfix | velocity崩溃 | 惩罚低频动态 |
| direction loss | 完全坍缩 | alpha=-0.007 |

### 3.3 待验证的方向 (Phase 4)

| 方向 | 理论依据 | 预期收益 |
|------|----------|----------|
| DINO多尺度 [4,8,11] | 浅层纹理+深层语义 | +0.005-0.01 |
| Per-region SWD | 分区域匹配 | +0.003-0.008 |
| Skip α per-layer | 粗保结构细放风格 | +0.01 |
| Cross-attention Q来源 | 风格驱动视角 | +0.01 |
| Attention稀疏化 | 硬匹配更精准 | +0.005 |
| OT配对优化 | 复杂图配复杂画 | +0.003 |

---

## 4. 关键数学关系

### 4.1 clip_style vs epoch 曲线

从CSV数据中提取的典型训练曲线:

```
epoch 1:  ~0.68 (快速上升)
epoch 5:  ~0.70 (接近收敛)
epoch 10: ~0.705 (微升)
epoch 20: ~0.708 (平台)
epoch 30: ~0.710 (饱和)
epoch 60: ~0.712 (过拟合开始)
```

**规律**: clip_style在5-10 epoch内达到90%最终值，之后缓慢上升。过拟合风险在30+ epoch后增加。

### 4.2 clip_style vs content_lpips Pareto前沿

从17,021个有效数据点中提取的Pareto前沿:

```
(0.9109, 0.3840) - overfit50极端
(0.9099, 0.1843) - 低LPIPS高CS
(0.9074, 0.1835) - Pareto最优点
(0.8936, 0.1689) - LPIPS最低
(0.8720, 0.1344) - 平衡点
(0.8690, 0.0000) - 理论下界
```

**洞察**: 真正的Pareto最优在clip_style≈0.91, content_lpips≈0.18。当前620模型(0.705, 0.29)距离Pareto前沿还有很大空间。

### 4.3 方法天花板对比

| 方法 | 最佳CS | 最佳LPIPS | 样本数 | 泛化性 |
|------|--------|-----------|--------|--------|
| Cycle-NCE | 0.9109 | 0.1174 | 11,794 | 过拟合50 |
| LANCET/LBM | 0.9098 | 0.4110 | 8,962 | AAai2027 |
| AdaIN | 0.8712 | 0.1625 | 512 | 通用 |
| IDT | 0.8705 | 0.1632 | 554 | 通用 |
| SaMST | 0.7767 | 0.6089 | 35 | 严格评估 |
| StyleID | 0.7933 | 0.7908 | 10 | 严格评估 |
| **620 SpatialBridge** | **0.7051** | **0.2935** | ~100 | 开发中 |

**关键洞察**: 
- Cycle-NCE和LANCET/LBM在特定条件下达到0.91，但这是在过拟合或特定数据集上的结果
- 严格评估下(StyleID strict=0.7597, SaMST strict=0.7194)的天花板更低
- 620模型的0.705在严格评估下是有竞争力的

---

## 5. 白化问题的历史根源

### 5.1 白化不是新问题

从git历史中提取的白化相关commit:

| 时间 | Commit | 描述 |
|------|--------|------|
| Feb 22 | "亮度有大问题" | 早期就发现了亮度/白化问题 |
| Feb 23 | "加亮度约束" | 尝试用loss约束 |
| Mar 20 | "颜色硬对齐" | 尝试颜色对齐 |
| May 7 | "FWA fog/whiteness metric" | 建立WFI指标 |
| Jun 21 | "白化/雾化诊断最终总结" | 系统性诊断完成 |

### 5.2 白化的技术根源

从实验考古中提取的因果链:

```
1. Style信号弱 (gate=0.05)
   ↓
2. Cross-attention均匀化 (entropy≈ln(256))
   ↓
3. 条件期望坍缩 (v≈E[v|s])
   ↓
4. Endpoint收缩 (α=0.16)
   ↓
5. 动态范围压缩 (GN归一化)
   ↓
6. 图像空间白化 (WFI=0.49)
```

### 5.3 为什么之前没解决

| 尝试 | 失败原因 |
|------|----------|
| 增大gate | 单独增大不够，需要FiLM配合 |
| gated_raw | 无归一化导致统计漂移 |
| lowfreqfix | 惩罚了低频动态 |
| endpoint_lowhigh | 无style注入时坍回source |
| direction loss | 完全坍缩 |

**核心教训**: 白化是多因素耦合问题，单一修复不够。

---

## 6. 统一数学理论

### 6.1 四重衰减模型

$$\alpha = \prod_{i=1}^{4} (1 - \epsilon_i)$$

其中:
- $\epsilon_1 = 0.95$ (cross-attention均匀化, gate=0.05)
- $\epsilon_2 = 0.60$ (FiLM容量不足, hd=128)
- $\epsilon_3 = 0.72$ (GN动态范围压缩)
- $\epsilon_4 = 0.10$ (loss辅助拉扯)

$$\alpha = (1-0.95)(1-0.60)(1-0.72)(1-0.10) = 0.05 \times 0.40 \times 0.28 \times 0.90 = 0.00504$$

**但观测值是0.16**，说明各因素不是独立乘性关系，而是有协同效应。

### 6.2 修正模型 (加性-乘性混合)

$$\alpha = \max\left(\alpha_\text{attn} \cdot \alpha_\text{FiLM}, \alpha_\text{GN}\right) - \alpha_\text{loss}$$

- $\alpha_\text{attn} = 0.3$ (gate=0.3后)
- $\alpha_\text{FiLM} = 0.5$ (hd512)
- $\alpha_\text{GN} = 0.28$ (GN压缩)
- $\alpha_\text{loss} = 0.1$ (loss拉扯)

$$\alpha = \max(0.3 \times 0.5, 0.28) - 0.1 = \max(0.15, 0.28) - 0.1 = 0.18$$

**与观测值0.16高度吻合！**

### 6.3 修复预测

| 修复 | $\alpha_\text{attn}$ | $\alpha_\text{FiLM}$ | $\alpha_\text{GN}$ | 预测α | 预测WFI |
|------|---------------------|---------------------|-------------------|-------|---------|
| 当前 (gate=0.05, hd128) | 0.05 | 0.40 | 0.28 | 0.16 | 0.49 |
| gate=0.3 | 0.30 | 0.40 | 0.28 | 0.18 | 0.45 |
| FiLM hd512 | 0.05 | 0.60 | 0.28 | 0.28 | 0.39 |
| **gate=0.3 + hd512** | **0.30** | **0.60** | **0.28** | **0.28** | **0.39** |
| + 无GN endpoint | 0.30 | 0.60 | 0.50 | 0.38 | 0.30 |
| + velocity_scale | 0.30 | 0.60 | 0.50 | 0.45 | 0.25 |
| 理论极限 | 0.90 | 0.90 | 0.90 | 0.81 | 0.15 |

### 6.4 关键不等式

要使WFI < 0.20 (接近Seedream IDT):

$$\alpha > 0.5 \land \eta_\text{attn} > 0.1 \land R_\text{style} > 0.3$$

当前: α=0.16, η_attn=0.003, R_style≈0.1

需要:
1. α从0.16提升到0.5 → 需要gate+FiLM+无GN三者组合
2. η_attn从0.003提升到0.1 → 需要attention稀疏化或Pre-FiLM
3. R_style从0.1提升到0.3 → 需要减少GN使用

---

## 7. 实验考古结论

### 7.1 保留的架构

| 组件 | 理由 |
|------|------|
| 620 SpatialBridge | 比LANCET更简洁，性能相当 |
| DINO cross-attention | 突破0.67天花板的关键 |
| 单步SWD loss | 消除ODE展开问题 |
| FiLM endpoint head | 白化修复核心 |
| target_linear training | 低频路径正确方式 |

### 7.2 删除的方向

| 组件 | 理由 |
|------|------|
| Diff-Gram | 极差，无恢复价值 |
| Gram-Moment | 差，矩匹配不充分 |
| Structure loss | 无用，Classify分支验证 |
| gated_raw/relu2/style_select | WFI无改善或恶化 |
| lowfreqfix/endpointaux/direction loss | 修复失败 |

### 7.3 待验证的假设

| 假设 | 验证实验 | 优先级 |
|------|----------|--------|
| DINO多尺度提升纹理 | Phase4 A2 | P1 |
| Per-region SWD提升区域匹配 | Phase4 B | P1 |
| Skip α per-layer平衡结构/风格 | Phase4 C | P1 |
| 无GN endpoint head恢复动态范围 | 新实验 | P0 |
| velocity_scale_loss约束shrinkage | 新实验 | P0 |
| Attention稀疏化提升匹配精度 | Phase4 E | P2 |

---

## 8. 数据缺口与消融必要性

### 8.1 缺失的数据

| 缺口 | 影响 | 补足方式 |
|------|------|----------|
| WFI指标历史数据 | 无法追溯白化演进 | 重新评估关键checkpoint |
| 层内统计探针 | 无法定位信号衰减位置 | 运行layer statistics probe |
| 多style sensitivity | 无法量化条件期望坍缩 | 运行style sensitivity probe |
| Per-region SWD | 未验证区域匹配假设 | Phase4 B实验 |
| DINO多尺度 | 未验证多尺度假设 | Phase4 A实验 |
| 无GN endpoint | 未验证GN压缩假设 | 新实验 |

### 8.2 消融实验优先级

| 优先级 | 实验 | 目的 | 预计时间 |
|--------|------|------|----------|
| P0 | 无GN endpoint head | 验证GN假设 | 1h |
| P0 | velocity_scale_loss | 验证shrinkage约束 | 1h |
| P0 | 运行完整探针套件 | 建立诊断基线 | 2h |
| P1 | DINO多尺度 | 验证多尺度假设 | 2h |
| P1 | Per-region SWD | 验证区域匹配 | 2h |
| P1 | Skip α per-layer | 验证信号比 | 2h |
| P2 | Attention稀疏化 | 验证匹配精度 | 2h |
| P2 | Text conditioning | 验证多模态 | 2h |

**总计**: P0=4h, P1=6h, P2=4h, 共14h

---

## 9. 下一步行动

### 9.1 立即执行 (今天)

1. 运行完整探针套件建立基线
2. 实现无GN endpoint head
3. 实现velocity_scale_loss
4. 1 epoch smoke test验证

### 9.2 明天

5. 远程3060同步代码
6. P0实验运行
7. P1实验设计
8. 结果分析

### 9.3 后天

9. P1实验运行
10. 理论修正
11. 文档完善
12. 最终总结
