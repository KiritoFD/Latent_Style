# 理论修正：用实验数据推翻和支持假设

> 基于645+实验、18个branch、6个月数据的理论重建

---

## 一、核心理论框架：风格迁移的三难困境

### 传统认知（1月）
```
风格强 ←→ 内容保  (二难)
```

### 数据修正后（6月）
```
        风格强 (clip_style↑)
       /        \
      /          \
内容保(LPIPS↓) — 训练稳(loss收敛)

三难困境：不能同时满足三者
```

**数据证据**：

| 方案 | 风格 | 内容 | 稳定性 | 验证 |
|------|------|------|--------|------|
| XPred Pattn | **0.729** | 0.637 LPIPS | 差(数值爆炸) | Inmortal family |
| LatAff s0.35 | 0.677 | **0.314** LPIPS | 好 | FiberBundle Phase2 |
| 620 notext_b8 | 0.665 | **0.287** LPIPS | 好(但白化) | 本地eval |
| Fiber-SDE σ=0.08 | 0.711 | 0.337 LPIPS | **不用训练** | ODE baseline |
| LANCET K e1 | 0.701 | 0.362 LPIPS | 好 | Distinct5 |

---

## 二、被推翻的理论假设

### 假设1: "新架构应该超越旧架构"

**原始信念**: 620 Spatial Bridge (DINO + CrossAttn + AdaLN) 是619诊断后的Golden Path实现，应该比LANCET更好。

**数据现实**:
```
LANCET K e1:    clip_style = 0.701, LPIPS = 0.362
620 local b8:   clip_style = 0.665, LPIPS = 0.287
620 remote best: clip_style = 0.675, LPIPS = 0.278
```

**620 clip_style低0.03-0.04，但LPIPS好0.07-0.08** — 620不是"更差"，而是在三难困境中选了不同的平衡点。

**修正理论**: 新架构未必在所有指标上超越旧架构。620的DINO encoder提供了更丰富的style信号，但gate=0.047说明模型学到了"少注入style更安全"的策略。**架构能力≠实际输出**，需要训练策略配合。

### 假设2: "Text条件能提升风格迁移质量"

**原始信念**: T5-base text encoder提供语义理解，应该帮助模型区分不同风格。

**数据现实**:
```
620 T5-base b4 e8:   clip_style = 0.666, LPIPS = 0.338
620 no-text b8 e8:   clip_style = 0.665, LPIPS = 0.287
```

**差0.001，在噪声水平内。**

**修正理论**: Text条件在当前架构下无效。原因分析：
1. **DINO patches已经是语义丰富的style表示** — T5提供的信息可能冗余
2. **gate=0.047说明style通道几乎关闭** — 不论加什么条件信号，都被gate截断
3. **Text条件需要在style注入有效的前提下才有用** — 当前瓶颈不在"style理解"而在"style注入"

**预测**: 如果先解决gate/白化问题，text条件可能变为有效。

### 假设3: "更大模型=更好效果"

**原始信念**: 增加维度(depth/width)应该增加表达力→更好风格迁移。

**数据现实**:
```
620 Capacity sweep (1-epoch smoke test):
  64×4:   clip_style = ~0.667
  64×6:   clip_style = ~0.667
  128×4:  clip_style = 0.668
  128×6:  clip_style = ~0.667

CGW 8-arch sweep (60-epoch full training):
  arch_1 through arch_8: clip_style 0.680-0.691 (21 sub-configs)
```

**容量3×增加→clip_style差异0.001。**

**修正理论**: **模型容量不是瓶颈**。overfit50 consistently best也支持这个结论——小数据上模型能学得很好，说明容量够用，问题是泛化/注入策略。

### 假设4: "训练越久越好"

**原始信念**: 更多epoch→更好收敛→更好结果。

**数据现实**:
```
620 film_v5_hd512:
  1 ep: WFI = 0.391 (最优)
  3 ep: WFI = 0.427
  5 ep: WFI = 0.453
  → 白化随训练加重

LANCET AAAI:
  1 ep: LPIPS = 0.427
  7 ep: LPIPS = 0.451
  → 内容保持随训练恶化

style_oa_8:
  60 ep: clip_style = 0.724
  → 但LPIPS=0.519，内容已经崩溃
```

**修正理论**: **训练时间存在最优停止点，过训练导致风格注入过度或白化**。风格迁移不是传统优化问题——loss下降不等于输出更好。

### 假设5: "SWD是好的style loss"

**原始信念**: SWD匹配统计分布，理论上应该改善风格质量。

**数据现实**:
```
Cycle-NCE era:
  SWD 30 vs 60: clip_style差0.003 (weight sweep)
  SWD patches [3,5] vs [7,11,15,19,25]: 无系统差异
  "SWD is really very bad" (Classify branch)
  
620 era:
  SWD 0/2/8/16 sweep (1ep): clip_style 0.660-0.669
  SWD 16+edge0: 微弱最优 0.669

re-SWD branch:
  "style-8's SWD directly went to NaN" — 数值不稳定
```

**修正理论**: SWD在**训练中**有效（提供梯度方向），但在**评估中**不可靠（clip_style和SWD不总一致）。且SWD对高频方向过敏感，可能导致训练不稳定。

---

## 三、被数据确认的理论

### 理论1: Cross-attention是风格注入的关键机制

**数据证据**:
```
01月: 纯Conv (SA-Flow)        → clip_style ~0.60
01月: 1-token cross-attn       → ~0.68 (但softmax=1.0)
03月: 64-token cross-attn      → 0.72 (突破!)
06月: XPred+Pattn             → 0.729 (Inmortal frontier)
06月: 620 (cross-attn, 但gate低) → 0.665 (回退=gate问题不是attn问题)
```

**但数据也显示**: cross-attn不总是好的——在gate被截断时，attn的输出被乘以0.047→几乎为零。**Cross-attn × gate = 实际注入量**，两者必须联合考虑。

### 理论2: Style-Content强耦合是根本障碍

**数据证据**:
```
Style8_Moment+SWD: corr(clip_style, LPIPS) = +0.94

具体表现:
  最好clip_style=0.729 → LPIPS=0.637 (XPred Pattn)
  最好LPIPS=0.287     → clip_style=0.665 (620 notext)
  好平衡点             → clip_style=0.697, LPIPS=0.319 (LANCET F)
```

**更深的理解**: 这种耦合不是偶然的。在latent space中，style和content共享同一组dimensions——修改style方向的同时必然扰动content方向。**解耦需要架构级别的分离**，而不仅仅是loss平衡。

### 理论3: ODE路径质量 ≠ 学习到的路径质量

**数据证据**:
```
Fiber-SDE σ=0.08 (不训练):  clip_style = 0.711, LPIPS = 0.337
LANCET best trained:        clip_style = 0.701, LPIPS = 0.362
620 best trained:           clip_style = 0.675, LPIPS = 0.278
```

**不训练的ODE路径比训练后的更好！** 这说明：
1. **模型在学习过程中"走偏"了** — 训练loss引导的方向不是最优输出方向
2. **SWD梯度和v_target梯度正交** — cos≈-0.024 (620诊断数据)
3. **训练loss最小化≠输出质量最大化** — 优化目标和评估目标不一致

### 理论4: Domain Style表示远优于Instance Style

**数据证据**:
```
Style8_Moment+SWD branch:
  Domain 1×1: ratio = 5.77× (最强)
  Instance 1×1: ratio = 1.15×
  Domain 3×3: ratio = 5.35×
  Instance 3×3: ratio = 1.23×
```

**Domain style(按风格类别的全局表示)比Instance style(单张图的表示)有效5.8倍**。这解释了为什么DINO patches有效——DINO提供的是domain-level语义信息。

### 理论5: 模型容量不是瓶颈

**数据证据**:
```
CGW 8-arch sweep:     clip_style 0.680-0.691 (差0.011)
620 capacity sweep:    clip_style ~0.667-0.668 (差0.001)
overfit50 consistently best: clip_style 0.59→0.81 (小数据大信号)
```

**overfit50能到0.81但full training只到0.72** — 模型有capacity学到好的style迁移，但泛化时丢失了。问题不在capacity，在**训练信号质量**和**注入策略**。

---

## 四、新理论：从数据中提炼

### 新理论1: Gate Collapse — 风格注入的保守偏好

**观察**: 620的所有实验中，style_gate收敛到0.047-0.050，不论gate_init设为0.05/0.3/0.5。

**解释**: 模型在训练中发现"少注入style"是降低training loss的安全策略。因为：
1. **L_flow = MSE(v_pred, v_target)** — v_target的方向是source→target，但幅度很小(alpha=0.163)
2. **L_swd惩罚风格不足** — 但SWD梯度与v_target梯度正交(cos≈-0.024)
3. **模型选择优先满足L_flow** — 因为L_flow权重=1而L_swd权重=8，但梯度方向更一致

**预测**: 需要改变loss平衡或gate机制，使style注入有利可图。

### 新理论2: Training-Output Mismatch — 优化目标错误

**观察**: Fiber-SDE不训练就达0.711，训练后反而更差。

**解释**: 
- FM loss引导模型学习velocity field v_theta(z_t, t)
- 但评估时只看endpoint z_1 = z_0 + integral(v_theta)
- **v_theta可以在训练中完美拟合但endpoint质量差** — 因为积分放大了小误差
- 类似于ODE数值稳定性问题：单步误差小但累积误差大

**类比**: 射击——准星精度高(v_theta好)但弹着散布大(endpoint差)，因为风速(积分)放大了误差。

### 新理论3: 风格迁移的有效维度极低

**观察**: 21个CGW configs几乎一样(0.680-0.691)，36个620消融也几乎一样(0.660-0.669)。

**解释**: 在latent 4×64×64=16384维空间中，有效的style方向只有极少数。这意味着：
1. **大部分architecture变体在相同的低维style子空间中工作** — 维度远小于4×64×64
2. **SWD只捕捉了少数统计维度** — 其余维度被忽略
3. **Cross-attention的多token是在这个低维空间中选择** — 64-token可能已经over-parameterized

**预测**: 有效style维度可能在10-50之间，而不是16384。找到这些关键维度比扩大模型更有效。

### 新理论4: 白化=Endpoint Shrinkage=训练loss的隐含偏好

**观察**: 
- Endpoint alpha = 0.163 (只走16%目标方向)
- 高频alpha = -0.050 (往反方向走)
- 3-epoch WFI恶化(0.43→0.47)

**解释**: 
1. **Endpoint prediction vs velocity prediction的tradeoff** — endpoint直接预测目标，但训练不稳定；velocity通过积分到达目标，但累积误差
2. **Shrinkage是velocity模型的隐式正则** — 模型学到"缩短路径=更安全"
3. **高频方向更容易被shrink** — 因为高频在latent space中variance小→梯度噪声大→模型偏好忽略

**与历史的联系**: 这和04月的"IN杀注意力"是同一类问题——**模型倾向于保守(白化/均匀attention/低gate)**，因为保守策略降低training loss但不改善输出。

---

## 五、修正后的行动优先级

### 旧优先级（1月-06月）
```
1. 架构改进 (cross-attn, new backbone)
2. Loss调优 (SWD权重, 新loss)  
3. 训练策略 (batch, lr, epoch)
4. 评估指标 (clip_style, LPIPS)
```

### 新优先级（基于数据修正）
```
1. 解决Gate Collapse — style注入为什么被截断？
2. 解决Training-Output Mismatch — 为什么训练loss↓但输出不改善？
3. 找到有效style维度 — 在16384维中哪些是关键？
4. 解耦style-content — 架构级别的分离，不是loss平衡
5. 才考虑架构/loss/训练策略
```

**核心转变**: 从"调参数找更好配置"转向"理解为什么模型做出保守选择"。

---

## 六、历史重演模式

| 时间 | 问题 | 根因 | 当时解决方案 | 是否真正解决 |
|------|------|------|------------|------------|
| 01月 | 1-token attn无选择性 | softmax=1.0 | 64-token vocabulary | ✅ 暂时解决 |
| 02月 | Cycle loss不稳定 | 对抗loss在latent space | 改MSE | ✅ 解决 |
| 04月 | IN杀注意力 | 白化→均匀attention | 去IN, twin-norm | ✅ 解决 |
| 05月 | Black-dot | 数值爆炸 | 删heuristic loss | ✅ 解决 |
| 06月 | Gate Collapse | 模型偏好保守 | **未解决** | ❌ |
| 06月 | Endpoint Shrinkage | 模型偏好保守 | **未解决** | ❌ |
| 06月 | WFI随训练恶化 | 训练-输出不一致 | **未解决** | ❌ |

**模式**: 每次都是"模型选择保守策略"——均匀attention、低gate、shrinkage、白化。**真正的瓶颈不是架构，而是训练信号不足以鼓励模型"大胆"注入style。**
