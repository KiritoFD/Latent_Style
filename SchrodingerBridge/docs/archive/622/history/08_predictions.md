# 下一步预测：基于统一理论 + 620完整数据

> 2026-06-23 | 基于836行数学理论 + 187条620 eval数据 + 645+历史实验

## 理论基础

统一数学理论（[latent_style_mathematical_theory.md](../../621/theory/latent_style_mathematical_theory.md)）识别出4个耦合机制形成**自强化退化吸引子**：

1. **Gate Collapse** — gate→0.048，style注入近乎关闭
2. **Endpoint Shrinkage** — α=0.163，只走16%目标方向
3. **Norm-Induced Whitening** — GN消除style的一阶/二阶统计量
4. **Training-Output Mismatch** — 逐step训练但评估endpoint，误差累积

模型在endpoint仅保留**0.5%原始style信息**。

---

## 620远程数据关键发现（187条eval）

### 1. 架构变体无关紧要

| 变体 | clip_style_transfer | LPIPS | 结论 |
|------|-------------------|-------|------|
| base_swd12 | 0.6724 | 0.2968 | baseline |
| adapter_swd12 | 0.6715 | 0.2916 | +adapter ≈ 0 |
| gate12_adapter | 0.6714 | 0.2915 | +gate12 ≈ 0 |
| moe_swd12 | 0.6711 | 0.2905 | +moe ≈ 0 |

**Δ < 0.0013**：在gate=0.05的前提下，任何架构改动都被退化吸引子吸收。

### 2. SWD宽度有最优值

| SWD宽度 | clip_style | LPIPS | 结论 |
|---------|-----------|-------|------|
| SWD-4 | 0.6704 | 0.2794 | 内容最好但style弱 |
| SWD-8 | 0.6720 | 0.2899 | 均衡 |
| **SWD-12** | **0.6724** | 0.2968 | **style最优** |
| SWD-16 | 0.6722 | 0.3058 | 无增益 |

### 3. NFE/sigma不是质量杠杆

- NFE=4/8/16: clip_style差异 < 0.0001
- sigma=0.00/0.02: clip_style差异 < 0.0005
- 这些是效率参数，不是质量参数

### 4. WFI-Style tradeoff确认

| 实验 | WFI | clip_style | LPIPS |
|------|-----|-----------|-------|
| film_gate03_5ep | **0.410** | 0.6675 | 0.3236 |
| film_v2_5ep | 0.451 | 0.6686 | 0.3340 |
| film_v4_gated | 0.487 | 0.6673 | 0.3241 |
| film_formal | 0.509 | **0.6735** | 0.3104 |

**WFI↑ → clip_style↑**：更好的style意味着更差的白化，因为当前模型通过"平均化"来部分满足style方向。

### 5. 训练普遍导致内容退化

- 39个实验中仅2个（lowswd_formal, lowmix05_gate12）LPIPS随训练改善
- 其余37个全部退化，平均ΔLPIPS = +0.025
- style/content最佳比通常在epoch 3-4，而非最终epoch

---

## 11个可证伪预测

### Gate Collapse相关

| ID | 预测 | 验证方法 | 证伪条件 |
|----|------|---------|---------|
| P1 | gate=0.3单独使用不会改善CLIP-style | 训练gate_init=0.3的模型 | 如果CLIP-style > 0.69 |
| P2 | gate必须配合style-correct方向才有效 | gate=0.3 + style_select_attn | 如果gate=0.3+任何方法>0.72 |
| P3 | gate warmup(0→0.3)比固定0.3更稳定 | 训练gate_schedule模型 | 如果warmup不稳定或效果差 |

### Endpoint Shrinkage相关

| ID | 预测 | 验证方法 | 证伪条件 |
|----|------|---------|---------|
| P4 | 移除GN endpoint使α从0.16→0.38 | 训练no-GN-endpoint模型 | 如果α<0.25 |
| P5 | velocity_scale_loss使α→0.45 | 训练+scale_loss模型 | 如果α<0.30 |

### 退化吸引子相关

| ID | 预测 | 验证方法 | 证伪条件 |
|----|------|---------|---------|
| P6 | 单轴修复被吸引子吸收，必须≥3轴同时修复 | 逐个修复gate/GN/FiLM | 如果任何单轴>0.72 |
| P7 | 有效style维度k≈10-50 | SVD sweep: top-1/5/10/50/100 | 如果无plateau或plateau>100 |
| P8 | sub-space projection(top-50)改善α→0.65 | 训练subspace-restricted模型 | 如果CLIP-style<当前最佳 |

### Training-Output Mismatch相关

| ID | 预测 | 验证方法 | 证伪条件 |
|----|------|---------|---------|
| P9 | 直接endpoint预测优于velocity预测 | 训练endpoint head | 如果endpoint<velocity |
| P10 | Multi-step SWD减少mismatch | SWD at t=0.25/0.5/0.75/1.0 | 如果CLIP-style改善<0.005 |

### Text条件相关

| ID | 预测 | 验证方法 | 证伪条件 |
|----|------|---------|---------|
| P11 | Text在gate=0.05时无效，在gate修复后有效 | gate=0.3+T5 vs gate=0.05+T5 | 如果gate=0.05+T5改善>0.01 |

---

## 6阶段修复路线图

| 阶段 | 干预 | 预测α | 预测WFI | 预测CLIP-style | 优先级 |
|------|------|-------|---------|---------------|--------|
| **当前** | gate=0.05, hd128, GN endpoint | 0.16 | 0.49 | 0.699 | — |
| **Stage 1** | + FiLM hd512 | 0.28 | 0.39 | 0.701 | 高 |
| **Stage 2** | + gate=0.3 (warmup) | 0.28 | 0.39 | 0.710 | 高 |
| **Stage 3** | + 移除GN endpoint (用RMSNorm) | 0.38 | 0.30 | 0.715 | 高 |
| **Stage 4** | + velocity_scale_loss | 0.45 | 0.25 | 0.720 | 中 |
| **Stage 5** | + AdaGN (style-modulated norm) | 0.55 | 0.20 | 0.725 | 中 |
| **Stage 6** | + subspace projection (top-50) | 0.65 | 0.18 | 0.735 | 低 |
| **Target** | All combined | >0.5 | <0.20 | >0.72 | — |

### 关键洞察：必须多轴同时修复

理论证明（Proposition 8.2）：**单轴修复被退化吸引子吸收**。

证据：
- gate=0.3单独 → CLIP-style反而降到0.696
- gated_raw_attn → WFI升到0.64
- relu2_attn → WFI=0.53
- 所有36个620消融在单轴上差异<0.015

**最少需同时修复3个轴**（gate + GN + FiLM capacity），总衰减率需 < ln(2) ≈ 0.693。

---

## 关于Text条件的结论

### 为什么Text在当前架构无效

1. **Gate Collapse直接阻断**：gate=0.05意味着95%的T5 text信息被tanh截断
2. **信息论证明**：cross-attention在η=0.997时仅能传输0.024 bits/position，T5 token再多也传不过去
3. **实证**：T5 vs no-T5差0.001（620_t5base实验）

### Text何时会有效

预测P11：gate修复到>0.2后，T5 text提供**语义维度的补充style信号**：
- DINO: 视觉纹理/色彩style（低层）
- T5: 语义style描述"oil painting"/"watercolor"（高层）
- 两者互补，预期在gate修复后T5可额外贡献+0.01~0.02

### 当前应该做什么

**先不碰Text**。Text调试应放在Stage 2（gate修复）之后。当前优先级：

1. **Stage 1: FiLM hd512** — 在当前gate下测试capacity增加的效果
2. **Stage 2: gate warmup + style_select** — 解决注入通道关闭问题
3. **Stage 3: 移除GN** — 解决norm消除style问题
4. **Stage 4+: 再开Text** — 在gate>0.2的条件下测试T5

---

## 最可能的突破路径

### 路径A：多轴联合修复（最保守，成功率最高）

1. 同时修改：gate_init=0.3(warmup) + FiLM hd512 + RMSNorm替代GN
2. 预测：α从0.16→0.38, CLIP-style从0.699→0.715
3. 风险：三轴联合可能导致训练不稳定
4. 缓解：gate warmup + 梯度裁剪 + 小学习率

### 路径B：Endpoint直接预测（最大胆，理论最干净）

1. 绕过ODE积分，直接预测z_g = z_s + Δz
2. 预测：消除Training-Output Mismatch，直接优化endpoint质量
3. 风险：可能失去ODE的平滑性优势
4. 缓解：保留velocity branch作为辅助loss

### 路径C：Style子空间投影（最精准，但需先测k值）

1. 先用SVD sweep确定有效style维度k
2. 将style injection限制在top-k子空间
3. 预测：减少浪费，提高注入效率
4. 风险：如果k>100则无效
5. 前置条件：需要P7实验结果

### 推荐：**先A后B**

- A是必要条件（不修gate/GN，任何方法都被吸收）
- B是充分条件（消除mismatch后可直接优化目标）
- C是优化手段（在A+B成功后进一步提升）

---

## 远程实验待补数据

### 缺WFI的35个实验

需要打包图片回本地，补白化/雾化评估：

**优先级高**（接近最佳style）：
- 620_lowswd_formal (clip_style=0.6751, 无WFI)
- 620_lowmix05_diag_b64 (0.6765, 无WFI)
- 620_swd12_b80 (0.6725, 无WFI)

**优先级中**（代表baseline）：
- 620_base_swd8_b80 (0.6720)
- 620_adapter_swd12 (0.6715)
- 620_gate12_adapter (0.6714)

**优先级低**（明显差或smoke test）：
- 其余29个

### 补评估方法

1. 远程打包：`tar czf /mnt/i/Github/Latent_Style/exp/620_eval_images.tar.gz 620_spatial_bridge/*/full_eval/`
2. SCP拉回本地
3. 本地运行WFI评估脚本

---

## 关键数值参考

| 常数 | 值 | 来源 |
|------|-----|------|
| Gate收敛值g* | 0.047-0.050 | 所有620实验 |
| Endpoint shrinkage α | 0.163 | 620诊断 |
| Cross-attn entropy η | 0.997 | 620 probe |
| Style信息保留率 | 0.5% | 信息论推导 |
| 不训练ODE CLIP-style | 0.711 | Fiber-SDE σ=0.08 |
| 训练后最佳CLIP-style | 0.6765 | lowmix05_diag e1 |
| SWD宽度最优 | 12 | 4/8/12/16消融 |
| NFE对质量影响 | ≈0 | 4/8/16消融 |
| 架构变体影响 | Δ<0.0013 | adapter/gate12/moe |
| WFI-CLIP-style相关性 | 正相关 | 4个film实验 |
