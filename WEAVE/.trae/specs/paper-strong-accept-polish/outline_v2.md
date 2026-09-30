# AAAI 2027 论文大纲（重写版）

## 核心叙事一句话

**问题**：消费级硬件上做风格迁移，要么训不动（diffusion 重），要么训得慢还跑得差（CUT/SaMam 436 min 且 SaMam 不如 identity）。**根因**：欧氏空间里内容和风格共享基，梯度互相打架，模型选择"白化"。**方案**：Haar 小波把 latent 切成 LL（内容锚）+ LH/HL/HH（笔触纤维），LL 锁死，HF 上跑 flow matching，末端 WCT 注入风格。**结果**：903K 参数，3 分钟，RTX 3060，CLIP-S 0.7213 + LPIPS 0.2868，双杀 SaMam 和 Seedream API。

## 各章节写什么（一句话定位 + 段落级大纲）

---

### Abstract（150 词）

**定位**：结果先行，方法一句话，难处一句话。

1. 第一句：SFM 在 RTX 3060 上 3 分钟训完，CLIP-S 0.7213 / LPIPS 0.2868，750 对 5 域。
2. 第二句：难处是消费级上只能产出"颜色滤镜"，SaMam 甚至不如 identity。
3. 第三句：根因是欧氏空间内容风格共享基，乘性衰减导致退化吸引子。
4. 第四句：方案是 Haar DWT 把 latent 切到正交子空间，LL 锁内容，HF 跑 flow + 末端 WCT。
5. 第五句：结果 903K 参数、3 分钟、真笔触迁移。

**禁止**：任何过程描述（645/22 模块/半年/减法消融）。

---

### §1 Introduction（约 1 页）

**定位**：难处 → 机制诊断 → 方案 → 硬结果。每段一个清晰目的。

**段落 1【难处，200 词】**：
- diffusion 派（CSGO）需 A100 + 21 万配对，消费级跑不起来。
- CUT/SaMam 能消费级训，但慢（322/436 min）。
- 更糟：SaMam CLIP-S 0.5816 < identity 0.6933，在做内容重建不是风格迁移。
- 实践上消费级输出是"颜色滤镜"：内容保住，笔触/纹理没了。
- 硬推风格 → 内容撕裂；加模块（cross-attn / FiLM / 判别器）无效。

**段落 2【Pareto 死锁，80 词】**：
- CLIP-S vs LPIPS 平面上，欧氏方法聚在一条 1:8 斜率的死线上。
- +0.01 CLIP-S 代价 ≈ 0.08 LPIPS。
- 这是结构限制不是容量限制。

**段落 3【退化吸引子，120 词】**：
- 死锁根因：退化吸引子 $\mathcal{M}_0$，velocity 与 style 无关。
- 三个乘性机制：gate 塌缩（0.05）、attention 均匀、GroupNorm 洗掉统计。
- 乘积 ≈ 0.016，匹配观测残差。
- 加模块没用，每个新模块自己也被拉到吸引子。

**段落 4【小波逃逸 / 方案，120 词】**：
- 关键洞察：吸引子存在是因为内容风格共享欧氏基。
- Haar DWT 把 latent 切成 LL（结构/色调）+ LH/HL/HH（笔触）。
- 内容保住 = LL 约束；风格迁移 = HF 问题。两目标不再竞争。
- 损失曲面从沟壑变近凸，100% 容量学笔触，分钟级收敛。

**段落 5【结果，60 词】**：
- 903K 参数，3 分钟，RTX 3060 12GB。
- 750 对 5 域：CLIP-S 0.7213 / LPIPS 0.2868。
- 双杀 SaMam（+0.14 CLIP-S，1/141 训练成本）+ Seedream 4.5 API（+0.19 LPIPS）。

**段落 6【贡献 3 条】**：
1. 诊断 Pareto 死锁到退化吸引子。
2. 提出 SFM：base locking + fiber flow + EOTA。
3. 903K / 3min / 750 对双杀 Mamba baseline + 商用 API。

---

### §2 Related Work（约 0.6 页）

**定位**：每一类相关工作只点出"为什么和我们的不同"，不堆砌。

- **神经风格迁移**：Gatys/AdaIN/WCT/SANet 都在欧氏特征空间 → 强笔触域产颜色滤镜。我们换坐标系。
- **Diffusion 风格迁移**：StyleID/SDEdit 高 CLIP-S 但 LPIPS > 0.45；CSGO 需 A100 + 21 万配对。
- **Mamba 风格迁移**：SaMam/SaMST 把 Mamba 当 style encoder；SaMam 在 750-pair 协议上 CLIP-S 0.5816 < identity 0.6933（做内容重建）。我们用 per-domain token bank，训得快 141×。
- **Flow matching**：rectified flow 的 $\sigma=0$ 特例；我们把 velocity 条件化在 Haar 分解上。
- **小波视觉方法**：传统用于多尺度采样；我们用正交性做内容风格解耦，是坐标系变换不是多尺度 trick。
- **解耦方法**：AdaIN/WCT 走统计、CycleGAN 走 cycle、CUT 走 patch 负对；都加新 loss。我们换坐标系让已有 loss 作用在不相交子空间。

---

### §3 Method（约 2 页，重点章）

**定位**：数学理论 + 架构。这是审稿重点，必须深入。

**§3.1 Problem Setup（100 词）**
- 形式化定义：$x_c$ → $z_c = \mathcal{E}(x_c) \in \mathbb{R}^{4\times32\times32}$，学 $G_\theta(x_c, s) = \mathcal{D}(\hat z_s)$。
- 一句话提 VAE latent 选择：192× 计算节省 + 语义压缩对风格更有利（详见 §3.2）。

**§3.2 Why Latent Space（120 词）**
- 两个理由：192× 算力节省（消费级可训）；VAE 压缩表示提供更好的语义/风格表征（像素空间低层纹理与内容纠缠）。引用 Rombach 2022。
- 像素空间对比实验在 RTX 3060 不可行（192× compute + 65536× attention），用文献论证。

**§3.3 Flow Matching on Latent（100 词 + 公式）**
- $v_\theta(z_t, t, s)$，rectilinear interpolation，FM loss（Eq. 1）。
- 推理：8 步 Heun solver（Eq. 2）。
- Heun vs Euler：+0.0056 CLIP-S；Heun vs RK4：饱和。Solver 阶是 Pareto-breaking 旋钮。

**§3.4 Haar Wavelet Decomposition（150 词 + Lemma 1 + Prop 1）**
- 单级 2D Haar DWT，4 子带半分辨率（Eq. 3）。
- 正交性：LL²+LH²+HL²+HH² = a²+b²+c²+d²。
- 多级：3 级最优（4 级失位置 -0.0003，2 级 -0.0018）。
- **Lemma 1**（小波等距 + Parseval）：$\|\mathcal{W}(z)\|_F^2 = \|z\|_F^2$，子空间无交叉项。
- **Proposition 1**（频谱内容-风格解耦）：内容能量若集中在 LL（≥1-ε），HF 上的变换对内容扰动 ≤ ε‖z^cnt‖²。
- 子带语义：LL 全局色调/布局，LH/HL 方向边缘/笔触，HH 噪声。

**§3.5 Base Locking（150 词）**
- 训练时 $w_{LL}=0$，推理时仍应用（小的、自由漂移的）预测 LL velocity。
- 反直觉：naive 会推理时冻结 LL"保护内容"，但这样会损失 +0.0141 CLIP-S（LL 也携带全局风格如色调/光照）。
- 让 LL 训练时漂移但推理时应用 drift → 既保内容锚（$v_\theta^{LL}$ 小）又开通全局风格通道。
- $w_{LH}=w_{HL}=1$，$w_{HH}=0$（HH head 移除，ablation Δ=±0.0001）。

**§3.6 Fiber Flow: Per-Subband Style Injection（200 词 + Def 1 + Prop 2）**
- 末端 latent $\hat z_s = \text{IDWT}(\hat\ell, \hat{lh}, \hat{hl}, \hat{hh})$，参考 $z_s^{ref}$。
- 每个 HF 子带做 WCT：$\Sigma_{c^{ref}}^{1/2} \Sigma_{\hat c}^{-1/2} (\hat c - \mu_{\hat c}) + \mu_{c^{ref}}$（Eq. 4）。
- LL 不变：$\ell^{new} = \hat\ell$。
- **Definition 1**（谱纤维丛）：总空间 E，投影 π 返回 LL 子带，纤维 $F_b$ 由 LH/HL/HH 参数化。
- **Proposition 2**（WCT 是 fiber-preserving）：π(WCT(z)) = π(z)，LL 不变。
- 几何意义：LL 是 base manifold，HF 是纤维，风格沿纤维迁移，base 不变。

**§3.7 End-of-Trajectory Injection（150 词 + Prop 3）**
- per-step injection 失败：残留 $r_n = (1-\alpha)^n$，n=8 α=0.5 时 r≈0.0039，α=0.5 与 1.0 不可分。
- EOTA：前 n-1 步纯 ODE，末端 $t=1$ 应用一次 WCT。
- **Proposition 3**（α-identifiability）：per-step 下 δ-不可分对覆盖 $[0,1]^2$；endpoint 下 $r_1=1-\alpha$ 单射，所有对可分。
- 与 Schrödinger bridge 视角一致：风格是终端约束不是 per-step 扰动。

**§3.8 Stochastic Frequency Routing（120 词）**
- cross-attn 路由可在 full latent 或 HF tokens 上做。推理恒用 wavelet query。
- 训练时 Bernoulli(p)：p=0.8 用 wavelet，否则 full。
- p 扫描 {0, 0.5, 0.8, 1.0} → CLIP-S {0.7061, 0.7083, 0.7213, 0.7226} / LPIPS {0.2606, 0.2480, 0.2868, 0.3068}。
- p=0.8 是双指标同时过阈的唯一点。
- p=1.0：style encoder 过拟合 HF，丢全局色调；p=0：train/test 失配。

**§3.9 The Degeneration Attractor（200 词 + Def 2 + Thm 1 + Prop 4）**
- 解释为什么小波分解必要。
- **Definition 2**（退化流形 $\mathcal{M}_0$）：$v_\theta$ 与 style 无关，$\nabla_s v_\theta = 0$。
- **Theorem 1**（$\mathcal{M}_0$ 是乘性衰减下的局部极小）：三条件 (i)(ii)(iii)，吸引域半径 $O(\kappa)$。
- 实测：gate≈0.05, attention≈1, norm≈0.32，$\kappa≈0.016$，匹配观测残差；$w_{SWD}/w_{FM}=0.01$，$\sqrt{w_{SWD}/w_{FM}}=0.1 \gg \kappa$，条件 (i) 成立。
- 欧氏基线在 CLIP-S ≈ 0.7175 处结构受限：任何三机制重加权都逃不出，因为乘积仍低于阈值。
- **Proposition 4**（小波逃逸）：小波分解下 Hessian 块对角化，交叉项消失，$\mathcal{M}_0$ 不再是临界流形。

**§3.10 Architecture and Training（100 词）**
- 4 个残差块 C=64，AdaIN conditioning，处理 [LL, LH, HL]（HH 移除）。
- per-subband 3×3 conv head 输出 $v^{LL}, v^{LH}, v^{HL}$。
- style encoder：frozen DINOv2 + 5 tokens × 64 dims per-domain bank。
- 903,248 trainable params。
- 5 epochs，batch 24，Adam lr 1e-4，RTX 3060 12GB。
- end-to-end 3 min 5 sec。

---

### §4 Experiments（约 2 页，重点章）

**定位**：硬结果 + 效率 + 消融。**不做 per-domain 表格**，那是没意义的数据堆砌。

**§4.1 Setup（150 词）**
- 数据集：Distinct5-WikiArt-512，5 域（Early Renaissance, Impressionism, Minimalism, Rococo, Ukiyo-e），每域 3600 训 / 150 测。
- latent：SDXL VAE 4×32×32。训练 18K latents，评测 750 对（5 src × 5 tgt × 30 src）。
- 指标：CLIP-S（CLIP ViT-B/32 cosine，输出 vs target-style prototype）、LPIPS（AlexNet）。
- baselines：12 个 = Identity + AdaIN + WCT + SD-Turbo + SDEdit{0.35,0.40} + StyleID + CUT + SaMST + SaMam + Seedream 4.5 + Ours。
- 训练 baseline 都在同一 18K 上训到收敛。

**§4.2 Main Results（300 词 + Table 1 + Fig scatter + Fig qual）**
- **Table 1**：12 方法 × (CLIP-S, LPIPS, Train min)。一行总结。
- **观察 1（SFM 双杀 SaMam）**：SaMam 0.5816 < identity 0.6933，在做内容重建。SFM +0.14 CLIP-S，1/141 训练成本。排除失败风格方法后，SFM 在真正迁移风格的方法中 LPIPS 最低。
- **观察 2（SFM 双杀 Seedream 4.5）**：商用 API 0.7198/0.4767，SFM 匹配 CLIP-S（+0.0015）且 LPIPS -0.19。API 是亿参数 diffusion 服务，SFM 是 903K 参数 3 分钟训完。
- **观察 3（SFM vs Identity）**：identity 是内容保持上限（LPIPS=0）。SFM LPIPS 0.2868 是风格迁移代价；CLIP-S 0.7213 > identity +0.028，证明真风格被迁移。AdaIN/WCT/SaMST 都 ≤ identity CLIP-S，没迁移风格。
- **观察 4（高 CLIP-S baseline 不是赢家）**：StyleID/SDEdit CLIP-S 最高但 LPIPS > 0.45 内容撕裂。Pareto 散点图（Fig scatter）显示这些点被支配，SFM 在左上无欧氏方法到达的区域。
- Fig qualitative：SFM 迁移笔触/纹理保内容；StyleID 撕裂；AdaIN 颜色滤镜；SaMam 内容重建无风格。
- 一句 CI 说明：SaMST/Seedream 的 bootstrap 95% CI 在 supplement；SFM/SaMam per-pair predictions 随代码发布。

**§4.3 Efficiency（200 词 + Table 2）**
- **Table 2**：4 方法 × (Params, Train time, Hardware, Speedup)。
- SFM 3.08 min vs SaMam 436 min（141×）vs CUT 322.6 min（105×）vs SaMST 39.5 min（12.8×）。
- 训练成本拆解：5 epochs × 18K latents × batch 24 = 3750 forward-backward，1 min 12 sec 纯训练 + 2 min 数据加载/优化器初始化/checkpoint。
- 为什么损失曲面良性：LL 锁死后 velocity 只需学笔触（低复杂度目标）。欧氏基线把算力浪费在内容重建与风格注入的非凸协商上。
- 收敛分析：HF loss 近凸（$u_t^{HF}$ 由 style 主导，$v_\theta^{HF}$ Lipschitz）；欧氏 loss 目标混合内容风格，梯度在 $\mathcal{M}_0$ 上冲突成鞍。欧氏基线 50-100 epoch 还到不了 identity；SFM 3 epoch 到最优，5 epoch 稳定。
- **Latent vs pixel space**：像素空间对比在 12GB 不可行（192× compute + 65536× attention）。引用 Rombach 2022 论证 latent 在等算力下质量更优。我们的 latent 选择既是计算必需又是表征优势。

**§4.4 Ablation: Component Analysis（250 词 + Table 3）**
- **Table 3**：从 SFM baseline 逐个移除/替换组件。分组：小波分解、base locking、风格注入、随机路由、solver、容量。
- **小波分解（活性成分）**：移除 DWT（回欧氏）→ -0.0096 CLIP-S（吸引子激活）。DWT 级数 1/4 不如 3。Daubechies-2 vs Haar 差异小（基函数非关键，正交性才是）。
- **Base locking**：推理时锁 LL → -0.0039 CLIP-S（杀全局风格通道）。训练 $w_{LL}=1$ → -0.0039（内容过重 trade-off）。训 HH head → ±0.0001（dead head 移除）。
- **风格注入**：per-step AdaIN（非 EOTA）→ +0.0148 CLIP-S 但 LPIPS 爆炸到 0.3843（α-decay）。AdaIN vs WCT → -0.0053（丢通道相关）。注入 LL → LPIPS 0.3856（颜色漂移传播）。
- **随机路由**：p=0 → -0.0152（train/test 失配）；p=0.5 → -0.0130；p=1.0 → +0.0013 CLIP-S 但 LPIPS 0.3068（style encoder 过拟合 HF）。p=0.8 唯一双过阈。
- **Solver**：Euler -0.0003（结构 DOF 失）；RK4 +0.0052 但饱和。
- **容量（1:8 Pareto 映射）**：depth/dim/gate 都沿 1:8 斜率移动，证实死锁。wavelet 分解 + solver 阶 + p 是仅有的结构 DOF。
- **1:8 trade-off 解释**：$\mathcal{M}_0$ 上 $v = \bar v + \kappa \delta v_s$，容量扰动 κ，κ 是 $\mathcal{M}_0$ 上唯一自由度，所以每个容量旋钮都走同一条线。wavelet 把模型移出 $\mathcal{M}_0$，使 κ 无关。

---

### §5 Discussion and Limitations（约 0.4 页）

**定位**：回答审稿会问的问题，不堆数据。

- **为什么 3 分钟能训完？**：不是因为模型小或数据小，是因为 LL 锁死后损失曲面变凸。欧氏下模型要协商两个矛盾梯度（内容 vs 风格），协商是非凸的耗时。小波下 LL 梯度为零，HF 梯度都指向风格，无需协商。
- **为什么是"真"风格迁移？**：AdaIN/WCT 在欧氏空间移整个 feature map 的均值/协方差 → 颜色滤镜。SFM 在 HF 子带上 per-subband WCT，LL 不变 → 笔触方向/边缘锐度/纹理粒度迁移，色调保住。这是"Real Style"的含义。
- **局限**：(1) 风格是 domain-level 不是 per-image；(2) Haar 是最简正交小波，双正交/学习小波可能更细；(3) 内容有强 HF 结构（文字/细图案）时可能被覆盖；(4) 5 域评测有限。

---

### §6 Conclusion（80 词）

**定位**：一句话总结 + 启示。

- Pareto 死锁是欧氏空间内容风格纠缠的结构后果，小波域正交分解使其消解。
- SFM：3 分钟 RTX 3060，903K 参数，双杀 Mamba baseline + 商用 API。
- 启示：社区不需要更大模型或更多配对数据，需要正确的坐标系。

---

### Ethics + Reproducibility + Checklist（约 0.5 页）

- Ethics：domain-level 非 per-image，降低伪造风险；WikiArt 公共领域。
- Reproducibility：公开数据 + 公开 VAE；3 min RTX 3060；97 sec 评测；3.5MB checkpoint；代码随发表发布。
- Checklist 7 条：贡献、数据、代码、硬件、协议、baseline、统计显著性（single-seed + SaMST/Seedream CI in supplement + SFM multi-seed in supplement）。

---

## 删除的内容

- **§4.5 Per-Domain Results 整节删除**：Table 4（per-domain）+ Table 5（5×5 transfer matrix）+ 两段分析。理由：
  1. 没有清晰的论证目的（为什么要逐风格比 IDT？论证什么？）
  2. 数据堆砌不服务核心叙事
  3. 占用大量版面却无信息增量
  4. 主结果 Table 1 + 散点图已足够支撑论点
- **Supplement §D.4 5×5 matrix 同步删除**。

## 保留的图表

- Figure 1（framework_sfm_main）：架构图。
- Figure 2（scatter）：CLIP-S vs 1-LPIPS 散点，SFM 在左上无欧氏方法到达区。
- Figure 3（qualitative）：定性对比。
- Table 1（main）：12 方法主对比。
- Table 2（efficiency）：训练效率。
- Table 3（ablation）：组件消融。

## 版面预算

- Abstract: 0.2 页
- §1 Introduction: 1 页
- §2 Related Work: 0.6 页
- §3 Method: 2 页（重点）
- §4 Experiments: 2 页（重点）
- §5 Discussion: 0.4 页
- §6 Conclusion: 0.2 页
- Ethics + Reprod + Checklist: 0.5 页
- **正文总计：~7 页**
- References + Checklist: ~2 页
- **总计：~9 页**

---

## 待用户确认

1. §4.5 Per-Domain Results 整节删除是否同意？
2. §3 顺序是否合理（Setup → Why Latent → FM → DWT → Base Locking → Fiber Flow → EOTA → Routing → Attractor → Arch）？
3. 是否还需要补充什么内容？
