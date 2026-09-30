# 完整实验影响图：每个改动的量化效果

> 2026-06-23 | 基于645+实验 / 22629 CSV记录 / 187条620远程eval / 30+本地620消融
> 所有数据点均有原始文件可追溯

---

## 一、总览：改动分类体系

6个月的所有实验改动可归入**7个正交轴**：

```
A. 架构轴 — 模型结构变化
B. 条件轴 — Style信号来源与注入方式
C. 容量轴 — 参数量/隐藏维度
D. 损失轴 — 训练目标函数
E. 训练轴 — 学习率/epoch/batch
F. 评估轴 — 指标定义与协议
G. 数据轴 — 数据集/分辨率/风格数
```

每个改动的影响用**Δ三元组**表示：`(Δclip_style, ΔLPIPS, ΔWFI)`
- 正值=恶化（WFI↑=更白化, LPIPS↑=内容差）
- 负值=改善

---

## 二、A轴：架构改动

### A1. 注意力模式 (style_attn_mode)

**背景**：Cross-attention从style tokens到content的注意力权重分布

| 模式 | WFI | CLIP-S | LPIPS | ΔWFI vs gated | 场景 |
|------|-----|--------|-------|---------------|------|
| **softmax** | **0.3736** | 0.7023 | 0.3397 | **-0.0189** | 消融1ep |
| style_select | 0.3751 | 0.7015 | 0.3366 | -0.0174 | 消融1ep |
| sparsemax | 0.3779 | 0.7018 | 0.3354 | -0.0146 | 消融1ep |
| gated_raw | 0.3850 | 0.7017 | 0.3453 | -0.0075 | 消融1ep |
| relu2 | 0.3856 | 0.7020 | 0.3434 | -0.0069 | 消融1ep |
| gated (baseline) | 0.3925 | 0.7020 | 0.3400 | 0 | 消融1ep |

**早期结果（无endpoint-FiLM时）**：
| 模式 | WFI | CLIP-S | LPIPS | 场景 |
|------|-----|--------|-------|------|
| gated | 0.4902 | 0.6987 | 0.3300 | E1 baseline |
| gated_raw | **0.6435** | 0.6987 | 0.2973 | E1 |
| relu2 | 0.5340 | 0.6964 | 0.3102 | E1 |
| style_select | 0.5005 | 0.6982 | 0.3331 | E1 |

**结论**：
- 在有endpoint-FiLM的baseline上，所有注意力模式都通过WFI<0.40门控
- **softmax最佳**（WFI 0.3736），gated反而最差（0.3925）
- 在无endpoint-FiLM时，gated_raw造成灾难性白化（0.64）
- **关键转折**：endpoint-FiLM引入后，注意力模式不再是白化瓶颈
- CLIP-S差异：全部 < 0.001，注意力模式对style transfer质量无影响

**数学解释**：
- η = 5.531 / ln(256) = 0.997 → 99.7%均匀
- 注意力信息容量 C = ln(N)(1-η) = 5.545 × 0.003 = 0.017 nats ≈ 0.024 bits
- **无论选哪种注意力模式，在gate=0.05时通过cross-attn的style信息都被压缩到0.024 bits**
- endpoint-FiLM绕过了这个瓶颈，直接从style→endpoint调制

### A2. GroupNorm位置与类型

**背景**：GN(1) = LayerNorm效果，消除一阶/二阶style统计量

| 配置 | WFI | CLIP-S | LPIPS | ΔWFI | 说明 |
|------|-----|--------|-------|------|------|
| 无endpoint-FiLM (有GN) | 0.4902 | 0.6987 | 0.3300 | baseline | E1 |
| + endpoint-FiLM (绕过GN) | 0.4283 | 0.7066 | 0.3226 | **-0.0619** | E2 P0 |
| + endpoint-FiLM hd512 | 0.3906 | 0.7015 | 0.3382 | **-0.0996** | E3 H2 |

**GN对style信号的衰减**：
- R_style per GN layer: ~0.2（保留20%的style差异）
- 经过L=2层GN: R_style^(2) ≤ 0.04
- 经过L=8层GN(4 blocks × 2): R_style^(8) ≤ 6.6×10⁻⁵ ≈ 0

**数学**：
```
GN(h) = (h - μ) / σ → Var[GN(h)] = 1, E[GN(h)] = 0
对不同style s1, s2:
  h(s1) ~ N(μ1, σ1²), h(s2) ~ N(μ2, σ2²)
  GN(h(s1)) ~ N(0,1), GN(h(s2)) ~ N(0,1)
  R_style → 0
```

**结论**：GN是白化的**第二大致因**（仅次于gate collapse），endpoint-FiLM通过绕过GN直接注入style到endpoint，效果ΔWFI=-0.062

### A3. Endpoint Head结构

| 结构 | WFI | CLIP-S | LPIPS | ΔWFI vs hd512 |
|------|-----|--------|-------|---------------|
| velocity (无FiLM) | **0.3769** | 0.7020 | 0.3315 | -0.0146 |
| lowhigh hd128 | 0.3801 | 0.7023 | 0.3422 | -0.0114 |
| lowhigh hd512 (baseline) | 0.3915 | 0.7019 | 0.3432 | 0 |
| lowhigh nofilm | 0.3957 | 0.7012 | 0.3399 | +0.0042 |
| lowhigh hd256 | 0.3990 | 0.7013 | 0.3408 | +0.0075 |

**结论**：
- **velocity head最简单且WFI最好**（0.3769）
- hd128 > hd256 > hd512：容量↑不等于质量↑
- **非线性发现**：hd128在WFI和CLIP-S上都优于hd512

### A4. 跨代际架构变迁

| 代际 | 架构 | clip_style天花板 | 时间 |
|------|------|-----------------|------|
| Gen1 | Thermal (SA-Flow) | 0.59 | 01-02月 |
| Gen2 | Cycle-NCE (CGW sweep) | 0.69 | 03月 |
| Gen3 | Cross-Attention (64-token) | **0.72** | 03月 |
| Gen4 | Style8 (Cycle-NCE full) | 0.724 | 03月 |
| Gen5 | SB Cleanup | 0.694 | 05月 |
| Gen6 | LANCET/LBM (K/Predictor) | 0.701 | 06月 |
| Gen7 | XPred+Pattn+Stokes | **0.731** | 06月 |
| Gen8 | 620 SpatialBridge | 0.6765 | 06月 |

**Gen3突破分析**（+0.03的来源）：
- 从Cycle-NCE的CGW到cross-attention：引入了64-token style representation
- 之前：style=1个全局向量(24d identity + 32d texture + 24d geometry)
- 之后：style=64个空间token(每个64d)，信息容量从~80d→~4096d
- **但这没有改变gate collapse**——Gen7的0.731是XPred直接预测endpoint

---

## 三、B轴：条件信号

### B1. Style条件来源（最关键的发现之一）

| 来源 | WFI | CLIP-S | LPIPS | contrast | saturation | 说明 |
|------|-----|--------|-------|----------|------------|------|
| **latent intrinsic** | **0.3842** | 0.7020 | 0.3417 | **3.54** | **0.249** | 消融1ep |
| DINO patches (无adapter) | 0.6407 | **0.7097** | **0.2773** | 1.70 | 0.115 | 消融1ep |
| DINO + adapter | 0.6076 | 0.7063 | 0.2618 | 1.84 | 0.117 | 消融1ep |

**重大反转**：
- 历史结论：DINO patches是620的必要条件
- 消融结论：DINO patches导致严重白化（WFI 0.64），latent intrinsic通过所有门控
- **DINO在CLIP-S上+0.008，但WFI恶化+0.26**

**数学解释**：
- DINO patches: 256个token × 768d = 196608维style信号
- Latent intrinsic: 4×64×64 = 16384维，但经过VAE KL正则化
- DINO信号维度过高→cross-attention被稀释→gate学到的还是0.05→注入的DINO信号被tanh截断→剩余信号导致均值偏移(brightness↑0.745)但对比度崩塌(1.70)

**历史vs现在对比**：
| 配置 | CLIP-S | LPIPS | 时期 |
|------|--------|-------|------|
| H6 intrinsic (无endpoint-FiLM) | 0.6717 | 0.3678 | 早期 |
| 当前intrinsic (有endpoint-FiLM) | 0.7020 | 0.3417 | 消融 |
| 差异 | +0.0303 | -0.0261 | endpoint-FiLM的增益 |

### B2. Block-level StyleFiLM

| 配置 | WFI | CLIP-S | LPIPS | Δ |
|------|-----|--------|-------|---|
| stylefilm ON | 0.3785 | 0.7020 | 0.3321 | baseline |
| stylefilm OFF | 0.3782 | 0.7021 | 0.3322 | Δ=0.0003 |

**结论**：**完全冗余**。在endpoint-FiLM存在时，block-level StyleFiLM贡献0.0003 WFI差异。

### B3. Text条件 (T5)

| 配置 | CLIP-S | LPIPS | 说明 |
|------|--------|-------|------|
| T5 enabled | 0.666 | ~0.29 | 620本地 |
| T5 disabled | 0.665 | ~0.29 | 差异0.001 |

**结论**：在gate=0.05时，T5 text条件完全无效（差0.001）。理论预测：gate>0.2后T5可提供语义维度+0.01~0.02。

---

## 四、C轴：容量

### C1. 隐藏维度 (base_dim)

| 配置 | WFI | CLIP-S | LPIPS | 参数量 | 说明 |
|------|-----|--------|-------|--------|------|
| 64x4 (baseline) | 0.3887 | 0.7021 | 0.3382 | ~1.70M | 消融 |
| 64x6 | **0.3828** | 0.7021 | 0.3426 | ~2.05M | 消融 |
| 128x4 | 0.3921 | **0.7026** | 0.3393 | ~6.50M | 消融 |
| 128x6 | 0.3895 | 0.7019 | 0.3436 | ~9.50M | 消融 |

**结论**：dim=128仅CLIP-S +0.0005，但WFI恶化。**容量不是瓶颈**。4×参数量≈0效果。

### C2. Endpoint hidden dim

| hd | WFI | CLIP-S | LPIPS |
|----|-----|--------|-------|
| 128 | **0.3801** | **0.7023** | 0.3422 |
| 256 | 0.3990 | 0.7013 | 0.3408 |
| 512 | 0.3915 | 0.7019 | 0.3432 |

**结论**：hd128最优。非线性——更多参数不更好。

---

## 五、D轴：损失函数

### D1. SWD权重

| SWD weight | WFI | CLIP-S | LPIPS | 场景 |
|------------|-----|--------|-------|------|
| 0 | 0.3921 | 0.7007 | 0.3384 | 消融1ep |
| 2 | 0.4001 | 0.7013 | 0.3304 | 消融1ep |
| 8 (baseline) | 0.3959 | 0.7018 | 0.3369 | 消融1ep |
| 16 | 0.4013 | **0.7028** | 0.3395 | 消融1ep |

**远程长训练（8 epoch）**：

| SWD宽度 | clip_style | LPIPS | 说明 |
|---------|-----------|-------|------|
| SWD-4 | 0.6704 | **0.2794** | 内容最好 |
| SWD-8 | 0.6720 | 0.2899 | 均衡 |
| **SWD-12** | **0.6724** | 0.2968 | **style最优** |
| SWD-16 | 0.6722 | 0.3058 | 无增益 |

**结论**：SWD存在style-WFI tradeoff：更高权重→更高CLIP-S但更高WFI。长训练中SWD-12最优。

### D2. NSWD噪声 (sigma)

| sigma | WFI | CLIP-S | LPIPS | 说明 |
|-------|-----|--------|-------|------|
| 0.02 (baseline) | 0.3959 | 0.7018 | 0.3369 | 消融 |
| **0.00** | **0.4105** | 0.7007 | 0.3398 | **WFI恶化+0.015** |

**结论**：**sigma=0.02是强制的**。移除噪声恶化WFI约0.015-0.029。

**数学**：NSWD理论：添加高斯噪声到投影后计算SWD，使得排序不稳定→梯度不再正交于v_target。
- cos(grad_SWD, v_target) = -0.024（无噪声时）
- sigma=0.02时期望交换率~46%，打破排序不变性→梯度方向性恢复

### D3. Edge loss

| edge | WFI | CLIP-S | LPIPS | 说明 |
|------|-----|--------|-------|------|
| 0.1 (baseline) | 0.3959 | 0.7018 | 0.3369 | 消融 |
| **0.0** | **0.3786** | **0.7020** | **0.3336** | **三赢** |

**结论**：**edge loss有害，应移除**。唯一同时改善WFI、CLIP-S、LPIPS的改动。

### D4. SWD+Edge组合

| SWD | edge | sigma | WFI | CLIP-S | LPIPS |
|-----|------|-------|-----|--------|-------|
| 16 | 0 | 0.02 | **0.3885** | **0.7030** | 0.3396 |
| 8 | 0 | 0.02 | 0.3786 | 0.7020 | 0.3336 |
| 16 | 0.1 | 0.02 | 0.4013 | 0.7028 | 0.3395 |

**结论**：SWD=16 + edge=0是CLIP-S最高且通过WFI门控的组合。

### D5. 损失函数演变时间线

| 时期 | Loss组成 | 效果 |
|------|---------|------|
| 01月 | L_flow | 0.59, 不收敛 |
| 02月 | + L_cycle + L_identity | 0.59, MSE>对抗 |
| 03月 | + L_style(gram) + L_swd | 0.69, gram无效但swd边际 |
| 04月 | + L_edge + L_kinetic + L_structure | 0.72, **structure完全无用** |
| 05月 | 精简到 L_flow + L_swd + L_edge | 0.694, 清理942→340行 |
| 06月 | **L_flow + L_swd(8) + σ=0.02, 无edge** | 0.702, 当前最优 |

---

## 六、E轴：训练策略

### E1. Gate初始化

| gate_init | WFI | CLIP-S | LPIPS | 说明 |
|-----------|-----|--------|-------|------|
| **0.05** | **0.3757** | 0.7020 | 0.3413 | 单独最优 |
| 0.3 | 0.3908 | 0.7022 | 0.3446 | 消融 |
| 0.5 | 0.3833 | 0.7022 | 0.3415 | 消融 |

**组合验证**：
| gate | 配置 | WFI | CLIP-S | LPIPS | 说明 |
|------|------|-----|--------|-------|------|
| 0.05 | hd128+film_off+edge0 | **0.4062** | 0.6994 | 0.3186 | **FAIL** |
| 0.3 | hd128+film_off+edge0 | **0.3757** | 0.6995 | 0.3422 | **PASS** |

**关键教训**：**单因素最优≠组合最优**。gate=0.05单独好，但在简化配置下FAIL。

**Gate Collapse机制**：
- 所有620远程实验（8 epoch）：gate收敛到[0.047, 0.050]
- gate=0.3初始化→训练后回落到0.294（1 epoch smoke）
- 证明：dL/dg > 0 在小g时（增加gate增加flow loss）
- 但gate=0.3在组合中仍比0.05好——因为初始偏差足够大

### E2. 训练Epoch

**1-epoch smoke vs 8-epoch formal**：

| 实验 | epoch | clip_style | LPIPS | WFI | 说明 |
|------|-------|-----------|-------|-----|------|
| film_formal | 1 | 0.6688 | 0.2886 | — | 远程 |
| film_formal | 4 | 0.6735 | — | — | 远程 |
| film_formal | 8 | **0.6735** | 0.3104 | 0.509 | 远程 |

**WFI随训练恶化**（hd128, lr=2e-4）：
| epoch | WFI | CLIP-S | LPIPS |
|-------|-----|--------|-------|
| 1 | 0.4271 | 0.7067 | 0.3236 |
| 2 | 0.4532 | 0.7095 | 0.3505 |
| 3 | 0.4680 | 0.7099 | 0.3768 |

**结论**：更多epoch在lr=2e-4下**单调恶化WFI**。Style/content最优比通常在epoch 3-4。

**远程39实验训练曲线模式**：
- 37/39实验LPIPS随训练恶化
- 仅2个（lowswd_formal, lowmix05_gate12）LPIPS改善
- clip_style通常ep7-8达峰，但内容已退化

### E3. 学习率

| lr | clip_style | LPIPS | 说明 |
|----|-----------|-------|------|
| 2e-4 | 0.7051 | 0.2935 | swd16_vlen0.04 5ep |
| 1e-4 | 0.6691 | 0.2828 | swd16_lr1e4 10ep（更慢学习） |

**结论**：lr=1e-4更慢但方向可能更稳。长训练可能需要更低lr。

### E4. NFE

| NFE | clip_style | LPIPS | 说明 |
|-----|-----------|-------|------|
| 4 | 0.6723 | 0.2966 | swd12 e8 |
| 8 | 0.6724 | 0.2968 | swd12 e8 |
| 16 | 0.6723 | 0.2971 | swd12 e8 |

**结论**：**NFE对质量零影响**。4步足够，这是效率参数。

---

## 七、F轴：评估与指标

### F1. CLIP-style vs LPIPS相关性

- Style8 branch: r = +0.94（强耦合）
- Full dataset (17021 points): r = -0.10（弱反相关）
- 620远程39实验: clip_style和LPIPS正相关

**解释**：在620架构内，更多style注入→更高clip_style但更多内容损失。跨架构时相关性弱。

### F2. WFI与style质量

| 实验 | WFI | clip_style | WFI-clip相关性 |
|------|-----|-----------|---------------|
| film_gate03_5ep | 0.410 | 0.6675 | 最低style最低WFI |
| film_v2_5ep | 0.451 | 0.6686 | |
| film_v4_gated | 0.487 | 0.6673 | |
| film_formal | 0.509 | 0.6735 | 最高style最高WFI |

**结论**：在当前架构内，**WFI和clip_style正相关**（更多style→更白化）。这是三难困境的直接体现。

### F3. Source vs Generated WFI

| 指标 | source | generated (gated) | 变化 |
|------|--------|-------------------|------|
| contrast_ratio | 11.57 | 2.40 | **-79.3%** |
| dynamic_range | 51.35 | 36.95 | -28.0% |
| saturation | 0.316 | 0.139 | **-56.0%** |
| brightness | 0.527 | 0.569 | +8.0% |
| WFI | 0.322 | 0.490 | **+0.168** |

---

## 八、G轴：数据集

### G1. 数据集演变

| 版本 | 分辨率 | 风格数 | 格式 | 关键发现 |
|------|--------|--------|------|---------|
| V0 | 64×64×4 | ~9 | .pt | 类别不平衡严重 |
| V1 | 64×64×4 | 2 | .pt | monet2photo基础建立 |
| V2 | 32×32×4 | 4 | .pt | 4风格不够 |
| V3 | 16×16×4 | 5 | .pt | batch=256 GPU预加载 |
| V7 | 512px | 5+ | 图片 | SaMST per-style 0.76-0.79 |
| V10 | 64×64×4 | open-set | DINO | 256 tokens |

### G2. Per-style vs Universal

| 方式 | 最高clip_style | 说明 |
|------|---------------|------|
| SaMST per-style | 0.793 (Symbolism) | 每风格独立模型 |
| SaMST per-style avg | 0.760 | 5风格平均 |
| LANCET universal | 0.731 | 单模型all-style |
| 620 universal | 0.6765 | 当前架构 |

**结论**：per-style vs universal差距~0.06，这是架构容量的理论上界。

---

## 九、跨轴交互效应

### IX.1 最重要的交互：单因素最优≠组合最优

**案例1：gate_init**
- 单独：0.05最优（WFI 0.3757）
- 组合（hd128+film_off+edge0）：0.05 FAIL（WFI 0.4062），0.3 PASS（WFI 0.3757）

**案例2：endpoint hd**
- 单独：hd512最优（WFI 0.3906 vs hd128 0.4283）
- Phase 5组合：hd128 + gate=0.3 → WFI 0.3757（优于hd512单独的0.3906）

**案例3：DINO**
- 早期（无endpoint-FiLM）：DINO必要（intrinsic 0.6717 vs DINO 0.71+）
- 现在（有endpoint-FiLM）：DINO有害（intrinsic 0.7020 WFI 0.38 vs DINO 0.7097 WFI 0.64）

**通用原则**：改动效果**依赖baseline配置**。每个改动的Δ三元组是baseline条件的函数。

### IX.2 退化吸引子效应

| 单轴干预 | 期望 | 实际 | 原因 |
|---------|------|------|------|
| gate=0.3 alone | CLIP-S↑ | 0.696 (↓) | 放大了错误style方向 |
| gated_raw attn | WFI↓ | 0.64 (↑) | 无归一化→统计漂移 |
| relu2 attn | WFI↓ | 0.53 | 稀疏但style无区分 |
| style_select attn | WFI↓ | 0.50 | Top-k不解决content-style冲突 |
| lowfreqfix | velocity稳定 | v=0.016 | 惩罚了结构所需的低频 |
| endpointaux | 更好endpoint | to_source_rms=0.055 | 崩塌回source |
| direction loss | alpha↑ | α=-0.007 | 过约束→灾难崩塌 |
| structure loss | 内容更好 | "完全无用" | Classify branch验证 |
| Diff-Gram | style更好 | 极差 | sdxl-fp32验证 |
| HF Residual | WFI↓ | 0.4746 | 不改变方向只加噪声 |

**理论**：单轴修复被退化吸引子吸收。需≥2轴同时修复。

---

## 十、改动效果排名

### 按WFI改善排名（620架构内）

| 排名 | 改动 | ΔWFI | ΔCLIP-S | ΔLPIPS | 条件 |
|------|------|-------|---------|--------|------|
| 1 | DINO→latent | **-0.257** | -0.0077 | +0.0644 | 消融1ep |
| 2 | +endpoint-FiLM hd512 | **-0.0996** | +0.0028 | +0.0082 | E1→E3 |
| 3 | edge loss移除 | **-0.0173** | +0.0002 | -0.0033 | 消融 |
| 4 | softmax替换gated | **-0.0189** | +0.0003 | -0.0003 | 消融 |
| 5 | gate 0.3 (组合中) | **-0.0305** | +0.0001 | +0.0236 | Phase5 |
| 6 | NSWD sigma=0.02 | **-0.0146** | +0.0011 | -0.0031 | 对比sigma=0 |
| 7 | StyleFiLM移除 | -0.0003 | +0.0001 | +0.0001 | 消融 |

### 按CLIP-S改善排名（跨架构）

| 排名 | 改动 | ΔCLIP-S | ΔLPIPS | 架构 |
|------|------|---------|--------|------|
| 1 | 64-token cross-attn | **+0.03** | +0.20 | Gen2→Gen3 |
| 2 | XPred endpoint | **+0.03** | +0.26 | Gen6→Gen7 |
| 3 | 620→LANCET回退 | **+0.025** | +0.07 | Gen8→Gen6 |
| 4 | CGW sweep vs Thermal | **+0.10** | -0.08 | Gen1→Gen2 |
| 5 | DINO patches | +0.008 | -0.064 | 620内 |

---

## 十一、关键数值常量汇总

| 常量 | 值 | 来源 |
|------|-----|------|
| Gate收敛值g* | 0.047-0.050 | 所有620远程实验 |
| Endpoint shrinkage α | 0.163 | 620诊断 |
| Cross-attn entropy η | 0.997 (5.531/ln256) | 620 probe |
| Velocity style sensitivity | cos(v_s1,v_s2)=0.9995 | Multi-style probe |
| SWD梯度与v_target夹角 | cos=-0.024 | Gradient probe |
| Style信息保留到endpoint | 0.5% | 信息论推导 |
| 不训练ODE CLIP-style | 0.711 | Fiber-SDE σ=0.08 |
| Pareto前沿点数 | 10/17021 | 22K+实验 |
| clip_style-LPIPS相关性 | +0.94 (620内), -0.10 (跨架构) | |
| 参数量 | 1.55M (620 dim=64) | Model count |
| 训练时间 | ~66h total (3060 12GB) | |
