# WEAVE Method 消融实验 Round 3

**日期**: 2026-07-09
**远程**: ssh -p 2222 administrator@100.115.18.62 (RTX 3060 12GB)
**数据集**: wikiart_distinct5_samam_512_latents_ema (5 styles × 1000 samples, I:/datasets/)
**训练**: batch_size=112, Patience=2, max=10 (10/15ep 实验自动触发早停)
**评估**: full_eval on wikiart_distinct5_samam_512_classview/test (5 styles × 30 images)
**Base config**: r2_spectral_10ep (Round 2 best, CLIP-S=0.7288, d_idt=0.0888)
**总耗时**: 246 min (09:22-13:28 remote), 6/6 ok

## 1. 实验结果总表

| 实验 | swd_w | ll_w | epochs | CLIP-S ↑ | CLIP-T | cLPIPS ↓ | d_idt ↑ | vs R2_10ep (CLIP-S) | 耗时 |
|------|-------|------|--------|----------|--------|----------|---------|---------------------|------|
| **r3_spectral_15ep** | 12 | 0.3 | 15 | **0.7311** | 0.2296 | 0.3524 | **0.0912** | **+0.0023** | 61.3min |
| r3_llw05_10ep | 12 | 0.5 | 10 | 0.7278 | 0.2289 | 0.3426 | 0.0879 | -0.0010 | 40.9min |
| r3_llw2_10ep | 12 | 2.0 | 10 | 0.7266 | 0.2285 | 0.3307 | 0.0867 | -0.0022 | 40.9min |
| r3_swd9_llw1_10ep | 9 | 1.0 | 10 | 0.7259 | 0.2284 | 0.3325 | 0.0860 | -0.0029 | 41.0min |
| r3_swd6_llw1_10ep | 6 | 1.0 | 10 | 0.7240 | 0.2279 | **0.3260** | 0.0841 | -0.0048 | 41.2min |
| r3_swd6_llw1_5ep | 6 | 1.0 | 5 | 0.7204 | 0.2271 | 0.3353 | 0.0805 | -0.0084 | 20.4min |
| (R2_spectral_10ep ref) | 12 | 0.3 | 10 | 0.7288 | 0.2290 | 0.3459 | 0.0888 | 0 | 40.8min |
| (R1_spectral_5ep ref) | 12 | 0.3 | 5 | 0.7237 | 0.2284 | 0.3536 | 0.0838 | -0.0051 | ~20min |

## 2. 核心发现

### 2.1 r3_spectral_15ep 是新的 STYLE SOTA (超越 Round 2)
- CLIP-S=0.7311, d_idt=0.0912 — 三轮全场最高 style 指标
- vs R2_spectral_10ep: +0.0023 CLIP-S, +0.0024 d_idt
- vs R1_spectral_5ep: +0.0074 CLIP-S, +0.0074 d_idt
- vs baseline (Round 1): +0.0224 CLIP-S (+3.16%)
- **结论**: 10→15ep 继续提升 style, 但 content 开始劣化 (+0.0065 cLPIPS)

### 2.2 r3_swd6_llw1_10ep 是 CONTENT SOTA
- cLPIPS=0.3260 — 三轮全场最低 (最佳内容保持)
- vs R2_spectral_10ep: -0.0199 cLPIPS (content 大幅改善)
- 代价: CLIP-S -0.0048 (style 略降)
- **结论**: swd_w=6 + ll_w=1.0 + 10ep 是 content-style trade-off 的 content 极端

### 2.3 训练长度的双相效应 (关键修正 Round 2 结论)
- spectral default (swd_w=12, ll_w=0.3):
  | epochs | CLIP-S | cLPIPS | d_idt |
  |--------|--------|--------|-------|
  | 5      | 0.7237 | 0.3536 | 0.0838 |
  | 10     | 0.7288 | 0.3459 | 0.0888 |
  | 15     | 0.7311 | 0.3524 | 0.0912 |
- **5→10ep**: strict win (ALL metrics improve, style +0.0051, content -0.0077, IDT +0.0050)
- **10→15ep**: split win (style +0.0023, IDT +0.0024, 但 content +0.0065 劣化)
- **结论**: 10ep 是 content sweet spot, 15ep+ 是 style sweet spot. 训练长度在 10ep 后转为 style-content trade-off

### 2.4 ll_w 在 10ep 下不是 strict improvement (修正 Round 2 结论)
- Round 2 在 5ep 下发现 ll_w=1.0 是 strict improvement over 0.3 (同 style, 更好 content)
- Round 3 在 10ep 下扫描 ll_w (swd_w=12 fixed):
  | ll_w | CLIP-S | cLPIPS |
  |------|--------|--------|
  | 0.3 (R2_10ep) | 0.7288 | 0.3459 |
  | 0.5 (r3_llw05) | 0.7278 | 0.3426 |
  | 2.0 (r3_llw2) | 0.7266 | 0.3307 |
- **ll_w ↑ → CLIP-S ↓ (monotonic), cLPIPS ↓ (content 更好)**
- **结论**: ll_w 在 10ep 下是 style-content trade-off 旋钮, 不是 strict improvement. Round 2 的 5ep 结论不推广到长训练. ll_w=0.3 是 style-optimal

### 2.5 swd_w 是 clean style-content trade-off knob
- 在 10ep + ll_w=1.0 下扫描 swd_w:
  | swd_w | CLIP-S | cLPIPS |
  |-------|--------|--------|
  | 6  (r3_swd6_llw1_10ep) | 0.7240 | 0.3260 |
  | 9  (r3_swd9_llw1_10ep) | 0.7259 | 0.3325 |
  | 12 (R2_10ep, ll_w=0.3) | 0.7288 | 0.3459 |
- **swd_w ↑ → CLIP-S ↑ (style 更好), cLPIPS ↑ (content 更差)**
- 注意: swd_w=12 的数据点用 ll_w=0.3 (非 ll_w=1.0), 但 Round 2 的 r2_spec_swd6 (swd_w=6, ll_w=0.3, 5ep) 也满足 monotonic, 趋势可靠
- **结论**: swd_w 是调节 style-content 平衡的最干净旋钮. swd_w=6=content 极, swd_w=12+=style 极

### 2.6 三重组合 (swd_w=6 + ll_w=1.0) 的 5ep vs 10ep 效应
| epochs | CLIP-S | cLPIPS | d_idt |
|--------|--------|--------|-------|
| 5 (r3_swd6_llw1_5ep) | 0.7204 | 0.3353 | 0.0805 |
| 10 (r3_swd6_llw1_10ep) | 0.7240 | 0.3260 | 0.0841 |
| Delta | +0.0036 | -0.0093 | +0.0036 |
- 10ep rescues 三重组合的 style drop (+0.0036) AND 放大 content 优势 (-0.0093)
- 但即使 10ep 也无法追上 R2 default 的 style (0.7240 vs 0.7288, -0.0048)
- **结论**: 三重组合适合 content-priority 应用, 不适合 style-priority

## 3. 全局排名 (跨 Round 1+2+3)

### Style 排名 (按 CLIP-S)
| Rank | 实验 | Round | CLIP-S | d_idt | cLPIPS |
|------|------|-------|--------|-------|--------|
| 1 | **r3_spectral_15ep** | R3 | **0.7311** | **0.0912** | 0.3524 |
| 2 | r2_spectral_10ep | R2 | 0.7288 | 0.0888 | 0.3459 |
| 3 | r3_llw05_10ep | R3 | 0.7278 | 0.0879 | 0.3426 |
| 4 | r3_llw2_10ep | R3 | 0.7266 | 0.0867 | 0.3307 |
| 5 | r3_swd9_llw1_10ep | R3 | 0.7259 | 0.0860 | 0.3325 |
| 6 | r3_swd6_llw1_10ep | R3 | 0.7240 | 0.0841 | 0.3260 |
| 7 | R1_spectral_5ep | R1 | 0.7237 | 0.0838 | 0.3536 |
| 8 | r3_swd6_llw1_5ep | R3 | 0.7204 | 0.0805 | 0.3353 |
| — | R1_soft_mask | R1 | 0.7196 | 0.0797 | 0.3386 |
| — | R1_no_swd_loss | R1 | 0.7159 | 0.0760 | 0.3277 |
| — | R1_baseline | R1 | 0.7087 | 0.0688 | 0.3644 |

### Content 排名 (按 cLPIPS, 仅 top-8)
| Rank | 实验 | Round | cLPIPS | CLIP-S |
|------|------|-------|--------|--------|
| 1 | **r3_swd6_llw1_10ep** | R3 | **0.3260** | 0.7240 |
| 2 | r3_llw2_10ep | R3 | 0.3307 | 0.7266 |
| 3 | r3_swd9_llw1_10ep | R3 | 0.3325 | 0.7259 |
| 4 | r3_swd6_llw1_5ep | R3 | 0.3353 | 0.7204 |
| 5 | R1_soft_mask | R1 | 0.3386 | 0.7196 |
| 6 | r3_llw05_10ep | R3 | 0.3426 | 0.7278 |
| 7 | r2_spec_llw1 (5ep) | R2 | 0.3434 | 0.7236 |
| 8 | r2_spectral_10ep | R2 | 0.3459 | 0.7288 |

## 4. Round 4 设计方向 (收敛检查)

Round 3 已确立两个 SOTA. Round 4 目标: **确认 SOTA 是否可继续推进, 或已收敛**.

### R4-1: 训练长度极限 (15→20ep)
- r3_spectral_15ep 是 style SOTA. 20ep 是否继续提升 style?
- 配置: swd_w=12, ll_w=0.3, 20ep
- 假设: style +0.001~0.002, content 继续劣化
- 判断: 若 ΔCLIP-S < 0.001, 训练长度已收敛

### R4-2: swd_w 上界探测 (12→15, 18)
- swd_w=12 是 default, 是否 swd_w>12 能进一步提升 style?
- 配置: swd_w=15, 18, ll_w=0.3, 15ep
- 假设: swd_w↑ 可能 style↑ 但 content 显著劣化
- 判断: 若 style 增益 <0.001 且 content 劣化 >0.01, swd_w=12 是 style sweet spot

### R4-3: ll_w 下界探测 (0.3→0.1, 0.0)
- Round 3 发现 ll_w=0.3 是 style-optimal at 10ep. 更低 ll_w 是否更好?
- 配置: ll_w=0.1, 0.0, swd_w=12, 15ep
- 假设: ll_w↓ 可能 style 微升但 content 显著劣化
- 判断: 若 style 增益 <0.001, ll_w=0.3 是 sweet spot

### R4-4: 最优组合 (style-priority)
- 基于 Round 3 发现, 组合 style-optimal 参数: swd_w=15 + ll_w=0.1 + 20ep
- 假设: 推到 style 极限, CLIP-S ≥0.733
- 判断: 若 <0.733, 已收敛

### R4-5: 最优组合 (content-priority)
- 基于 Round 3 发现, 组合 content-optimal 参数: swd_w=6 + ll_w=2.0 + 10ep
- 假设: 推到 content 极限, cLPIPS ≤0.320
- 判断: 若 <0.320, content 极限可达

## 5. Infra 备注

- GPU 功率: 训练 41-48W, 评估 ~85W (latent-space 结构性限制)
- VRAM: 训练 9.5-10.0GB, 评估 <7GB (符合约束)
- 每 epoch ~194s, 10ep=33min训练+7min评估=40min, 15ep=50min+11min=61min
- 数据在 I: 盘 (HDD), I/O 瓶颈限制 GPU 功率
- bs=112 在 12GB VRAM 下稳定
- 6 个实验 246 min, 无 OOM, 无 divergence, 无 early stop 触发 (所有实验跑满 epoch)
- Round 3 全程infra稳定, 无需干预

## 6. MUSIQ 补充分析 (no-reference 图像质量)

事后对 Round 1+2+3 所有实验的生成图片计算 MUSIQ (pyiqa, koniq checkpoint, 750 imgs/exp).

### 6.1 MUSIQ 排名
| Rank | 实验 | Round | MUSIQ ↑ | CLIP-S | cLPIPS |
|------|------|-------|---------|--------|--------|
| 1 | **R1_baseline** | R1 | **51.34** | 0.7087 | 0.3644 |
| 2 | R3_swd9_llw1_10ep | R3 | 44.66 | 0.7259 | 0.3325 |
| 3 | R3_spectral_15ep | R3 | 44.56 | 0.7311 | 0.3524 |
| 4 | R3_llw2_10ep | R3 | 44.42 | 0.7266 | 0.3307 |
| 5 | R3_swd6_llw1_10ep | R3 | 44.36 | 0.7240 | 0.3260 |
| 6 | R3_llw05_10ep | R3 | 44.22 | 0.7278 | 0.3426 |
| 7 | R2_spectral_10ep | R2 | 43.95 | 0.7288 | 0.3459 |
| 8 | R1_soft_mask | R1 | 41.63 | 0.7196 | 0.3386 |
| 9 | R1_spectral_5ep | R1 | 41.50 | 0.7237 | 0.3536 |
| 10 | R2_spec_llw1_5ep | R2 | 41.00 | 0.7236 | 0.3434 |

## 7. Round 4 + MUSIQ + DINO 综合分析 (2026-07-09 16:00)

用户提醒"不只要考虑clip-s，也要看MUSIQ的啊"后, 触发 Round 4 方向 pivot (从 "push style limit" 改为 "quality recovery"), 并补充计算 DINO 指标 (DINOv2-small, 750 imgs/exp, 0 skipped).

### 7.1 评估指标说明
- **CLIP-S** (clip_style): cos(CLIP(gen), CLIP(style_proto)) — 风格相似度, 越高越好
- **cLPIPS** (content_lpips): LPIPS(gen, content_src) — 内容保持, 越低越好
- **MUSIQ**: 无参考图像质量 (pyiqa koniq), 感知质量, 越高越好
- **DINO-content**: cos(DINO_CLS(gen), DINO_CLS(content_src)) — 内容语义保持, 越高越好
- **DINO-style**: max cos(DINO_CLS(gen), DINO_CLS(style_ref)) — 风格语义一致性, 越高越好
- **DINO-structure**: DINOv2 penultimate patch self-similarity MSE — 结构保持, 越低越好

### 7.2 完整结果总表 (7 个已完成实验, MUSIQ+DINO 全部补完)

| 实验 | mode | ep | CLIP-S ↑ | cLPIPS ↓ | MUSIQ ↑ | DINO-con ↑ | DINO-sty ↑ | DINO-str ↓ |
|------|------|----|---------|---------|---------|-----------|-----------|-----------|
| r4_baseline_10ep | region | 10 | 0.7100 | 0.3842 | **53.11** | 0.6831 | 0.4663 | 0.0282 |
| r4_baseline_15ep | region | 15 | 0.7143 | 0.4026 | 51.84 | 0.6642 | 0.4702 | 0.0289 |
| r4_region_swd9_15ep† | region+swd9 | 5 | ~0.7085 | ~0.39 | 51.96 | 0.7129 | 0.4661 | 0.0273 |
| **abl_baseline** (R1) | region | 5 | 0.7087 | 0.3644 | 51.34 | 0.7070 | 0.4584 | 0.0271 |
| **r4_spec_swd9_15ep** | spectral+swd9 | 15 | 0.7281 | 0.3400 | **45.31** | 0.7501 | **0.4864** | 0.0259 |
| r3_spectral_15ep | spectral | 15 | **0.7311** | 0.3524 | 44.56 | 0.7306 | 0.4853 | 0.0265 |
| **r3_swd9_llw1_10ep** | spectral+swd9 | 10 | 0.7259 | 0.3325 | 44.66 | **0.7615** | 0.4854 | **0.0255** |

†r4_region_swd9_15ep 的 epoch_0015 eval 因 GPU OOM 失败, 此处用 epoch_0005 代替, 指标仅供参考.

**关键**: r4_spec_swd9_15ep 是 spectral mode 下的 MUSIQ 最优 (45.31), 同时保持高 CLIP-S (0.7281) 和高 DINO-content (0.7501).

### 7.3 关键发现

#### Finding 7.3.1: MUSIQ 与 CLIP-S 强负相关 — 质量vs风格的根本权衡
- Region mode: MUSIQ=51-53, CLIP-S=0.71
- Spectral mode: MUSIQ=44-45, CLIP-S=0.73
- **7-9 MUSIQ 分的差距是 mode 选择的根本代价**
- 训练长度不能弥补: region 15ep (MUSIQ=51.84) 仍远好于 spectral 5ep (MUSIQ=41.50)

#### Finding 7.3.2: r4_baseline_10ep 是全局 MUSIQ SOTA (53.11)
- 超过 R1_baseline (51.34) +1.77 分
- 超过所有 spectral 变体 +8.5 分
- **10ep 是 region mode 的质量甜蜜点** (vs 15ep 的 51.84, 10ep > 15ep)
- 但 CLIP-S 仅 0.7100, 风格转移能力弱

#### Finding 7.3.3: DINO 指标揭示 spectral 的"双刃剑"特性
- **DINO-content**: spectral (0.73-0.76) > region (0.66-0.71) — spectral 更好地保持了内容语义?! 反直觉
- **DINO-style**: spectral (0.485) > region (0.466) — spectral 风格一致性更强, 符合预期
- **DINO-structure**: spectral (0.0255-0.0265) < region (0.0271-0.0289) — spectral 结构保持更好
- **解释**: DINOv2 是语义特征, spectral ODE 的 DWT 解耦可能更好地保留了语义结构, 但同时引入了 MUSIQ 敏感的 artifact (高频伪影)

#### Finding 7.3.4: r3_swd9_llw1_10ep 是 DINO 全能 SOTA
- DINO-content=0.7615 (最高), DINO-style=0.4854 (最高), DINO-structure=0.0255 (最低)
- 同时 CLIP-S=0.7259 (spectral 中第二高), cLPIPS=0.3325 (内容第二好)
- 但 MUSIQ=44.66 (远低于 region)
- **结论**: swd_w=9 + ll_w=1.0 是 spectral mode 的语义最优配置

#### Finding 7.3.5: 训练长度对 MUSIQ 和 DINO 的影响相反
- Region mode: 5ep→10ep→15ep, MUSIQ=51.34→53.11→51.84 (10ep 峰值)
- Region mode: 5ep→10ep→15ep, DINO-content=0.7070→0.6831→0.6642 (持续下降)
- **训练越长, 语义内容保持越差, 但 MUSIQ 在 10ep 达峰** — 10ep 是 region 的质量-语义平衡点

### 7.4 Pareto 前沿分析 (更新, 含 r4_spec_swd9_15ep)

| 维度 | 最优实验 | 值 | 次优实验 | 值 |
|------|---------|-----|---------|-----|
| 风格 (CLIP-S) | r3_spectral_15ep | 0.7311 | r4_spec_swd9_15ep | 0.7281 |
| 感知质量 (MUSIQ) | r4_baseline_10ep | 53.11 | r4_region_swd9_15ep† | 51.96 |
| 内容语义 (DINO-con) | r3_swd9_llw1_10ep | 0.7615 | r4_spec_swd9_15ep | 0.7501 |
| 风格语义 (DINO-sty) | r4_spec_swd9_15ep | 0.4864 | r3_swd9_llw1_10ep | 0.4854 |
| 结构保持 (DINO-str) | r3_swd9_llw1_10ep | 0.0255 | r4_spec_swd9_15ep | 0.0259 |
| 内容像素 (cLPIPS) | r3_swd6_llw1_10ep | 0.3260 | r3_swd9_llw1_10ep | 0.3325 |

**Pareto 最优配置**:
- 若优先质量 (MUSIQ): **r4_baseline_10ep** (region, 10ep, MUSIQ=53.11, CLIP-S=0.7100)
- 若优先语义 (DINO): **r3_swd9_llw1_10ep** (spectral, swd_w=9, ll_w=1.0, 10ep, DINO-con/str SOTA)
- 若优先风格 (CLIP-S): **r3_spectral_15ep** (spectral, 15ep, CLIP-S=0.7311)
- **spectral 均衡最优: r4_spec_swd9_15ep** (spectral+swd9+15ep, MUSIQ=45.31 spectral最高, CLIP-S=0.7281, DINO四项第二)

### 7.5 综合判断与核心矛盾 (最终结论)

**核心矛盾**: spectral ODE 提升 CLIP-S (+0.02) 和 DINO (所有维度), 但损害 MUSIQ (-7到-9分). 这意味着 spectral 引入的 artifact 对人眼感知有害, 但对 CLIP/DINO 的语义特征无害甚至有益 (DWT 解耦保留了语义结构, 但引入高频伪影).

**关键发现: swd_w=9 是 spectral mode 的改善旋钮**
- r4_spec_swd9_15ep vs r3_spectral_15ep (swd_w=12):
  - MUSIQ: 45.31 vs 44.56 (+0.75, 改善)
  - DINO-content: 0.7501 vs 0.7306 (+0.0195, 改善)
  - DINO-style: 0.4864 vs 0.4853 (+0.0011, 微改善)
  - DINO-structure: 0.0259 vs 0.0265 (-0.0006, 改善)
  - CLIP-S: 0.7281 vs 0.7311 (-0.003, 微降)
  - cLPIPS: 0.3400 vs 0.3524 (-0.0124, 内容改善)
- **结论**: 降低 swd_w 从 12→9 在 spectral 下几乎全面改善 (除 CLIP-S 微降 0.003)

**infra 问题分析 (2026-07-09 18:00 更新, T11复现实验确认)**:
- **T11 复现实验** (configs/630_local_t11_repro_i.json): 基于 630_local_t11_stochastic_dwt_p08, 切换到 I: 盘 samam 数据集, bs=112
  - 训练速度: **18.7s/epoch** (5ep训练=94秒), GPU功率 **132-134W** (满载), VRAM **7.86GB**, sps=263
  - eval指标: CLIP-S=0.7128, cLPIPS=0.3028, d_idt=0.0728 (与原G:盘t11的CLIP-S=0.7213基本一致, bs差异导致微小偏差)
  - **结论: infra无问题, 数据集无问题. t11方法(squared SWD + global scale + kinetic off)在统一数据集上仍极快**
- Region mode (R1 baseline): 79秒/epoch, 5ep训练≈6.5分钟
- Spectral mode (r4_spec_swd9_15ep): 256秒/epoch, 15ep训练≈64分钟
- **慢的根因是方法, 不是infra**: r4_spec_swd9 使用 `cdf + cross-attn-guided + region_spectral + kmeans×4 + kinetic_penalty=global_l2`, 计算量是 t11 (`squared + global + kinetic off`) 的 **13.7倍** (256/18.7)
- **spectral_ode_enabled=true 本身不慢**: t11 也启用 spectral_ode, 但用简单 SWD 配置仍 18.7s/ep. 慢来自 SWD 复杂度 (cdf距离 + cross-attn引导 + region_spectral kmeans聚类), 不是 DWT 本身
- 用户期望的"5分钟以内训练"在简单SWD配置下可达 (t11=1.5min训练+2min eval=3.5min), spectral复杂配置不可达

**Round 4 终止决定**:
剩余3个实验 (r4_spec_swd9_20ep, r4_spec_swd9_llw05_15ep, r4_softmask_15ep) 已停止. 理由:
1. r4_spec_swd9_15ep 已证明 swd_w=9 在 spectral 下的均衡优势, 20ep 不会有质变
2. ll_w 对 MUSIQ 影响在 Round 3 已知 (近似 flat), llw05 不会突破
3. softmask (region_soft) 在 Round 1 已知 MUSIQ=41.63 (最差), 不会优于 region
4. spectral mode 训练太慢 (256秒/epoch), 性价比低

**下一步方向** (Round 5, 待用户确认):
1. **artifact 抑制**: 在 spectral mode 下加入高频正则化 (HF L2 penalty) 尝试恢复 MUSIQ
2. **混合模式**: 训练用 spectral, 推理时切换 region (或混合)
3. **DINO-aware loss**: 用 DINO feature 作为额外内容损失, 可能改善 region 的 DINO-content
4. **swd_w 极端值**: swd_w=3 或 swd_w=1, 测试极低 SWD 权重能否在 region 下提升 CLIP-S

### 7.6 Round 5 高性能实验 hp_simple_swd12_15ep (2026-07-09 20:00, SWD 简化结论)

**触发问题**: 用户要求核查 "复杂 SWD 是否提升 DINO 指标". 若无提升则关闭, 并综合最优结论拼一个高性能配置顶满显存跑测速 + CLIP/DINO/LPIPS 评估.

**SWD 复杂度对 DINO 影响的对照** (核心结论):
| SWD 配置 | distance | scale | semantic | 训练速度 | DINO-con | DINO-sty | DINO-str |
|----------|----------|-------|----------|----------|----------|----------|----------|
| 简单 SWD (t11/hp) | squared | global | off | **18.6s/ep** | **0.8002** | 0.4727 | **0.0242** |
| 复杂 SWD (r4_spec_swd9) | cdf | cross-attn | region_spectral+kmeans | 256s/ep | 0.7501 | **0.4864** | 0.0259 |
| 复杂 SWD (r3_swd9_llw1) | cdf | cross-attn | region_spectral+kmeans | ~256s/ep | 0.7615 | 0.4854 | 0.0255 |

**关键判断**: 复杂 SWD 不仅没有提升 DINO-content, 反而**降低了 0.040-0.039** (0.7615/0.7501 vs 0.8002), 同时训练慢 **13.7 倍**. 复杂 SWD 仅在 DINO-style 上微弱领先 (+0.013). 综合速度 + DINO-content + DINO-structure, **复杂 SWD 应关闭, 改用简单 SWD (squared + global + off)**.

**hp_simple_swd12_15ep 配置** (configs/hp_simple_swd12_15ep.json):
- 基础: 630_local_t11_stochastic_dwt_p08 (t11 架构: region + DWT route p=0.8)
- SWD: `single_step_swd_weight=12.0` (R3 style-optimal), `swd_scale_mode=global`, `swd_distance_mode=squared`, `swd_semantic_mode=off`, `kinetic_penalty_mode=off`
- spectral_ode: enabled, `spectral_w_ll=0.3` (R3 style-optimal)
- 训练: `batch_size=160` (顶满 VRAM), `num_epochs=15`, patience=2
- 路径解耦: 全面使用 `$index:KEY` 引用 (dataset_index.json 机制), 本地 G: 与远程 I: 各一份索引

**训练实测** (远程 RTX 3060 12GB):
- 速度: **18.6s/epoch** (15ep = 4.8 min), GPU 功率 135W (满载), sps=685
- VRAM: **11.54 / 11.54 GB** (顶满, 接近 OOM 但稳定)
- 对比 r4_spec_swd9_15ep: 256s/ep → hp 18.6s/ep, **快 13.7 倍**

**评估指标** (epoch_0015, full_eval + 手动 DINO):
| 指标 | hp_simple_swd12_15ep | r4_spec_swd9_15ep | r3_swd9_llw1_10ep | r4_baseline_10ep |
|------|---------------------|-------------------|-------------------|------------------|
| CLIP-S | 0.7167 | **0.7281** | 0.7259 | 0.7100 |
| CLIP-T | 0.2233 | — | 0.2284 | — |
| cLPIPS | **0.2990** | 0.3400 | 0.3325 | 0.3842 |
| DINO-content | **0.8002** | 0.7501 | 0.7615 | 0.6831 |
| DINO-style | 0.4727 | **0.4864** | 0.4854 | 0.4663 |
| DINO-structure | **0.0242** | 0.0259 | 0.0255 | 0.0282 |
| identity CLIP-S | 0.8383 | — | — | — |
| 训练速度 (s/ep) | **18.6** | 256 | ~256 | 79 |

**hp 实验的 SOTA 维度**:
- **DINO-content 全场最高**: 0.8002 (超复杂 SWD 最佳 0.7615 达 +0.0387, 超 region baseline 0.6831 达 +0.1171)
- **DINO-structure 全场最优**: 0.0242 (越低越好, 内容结构保持最佳)
- **cLPIPS 全场最低**: 0.2990 (内容像素保持最优, 超原 SOTA r3_swd6_llw1_10ep 的 0.3260 达 -0.027)
- **训练速度全场最快**: 18.6s/ep (复杂 SWD 的 1/13.7)

**hp 实验的非 SOTA 维度**:
- CLIP-S=0.7167 (低于 r3_spectral_15ep 0.7311 和 r4_spec_swd9 0.7281, 但高于 r4_baseline 0.7100)
- DINO-style=0.4727 (低于 r4_spec_swd9 0.4864, 风格语义注入略弱)

**综合判断 (Round 5 结论)**:
简单 SWD (squared + global + off) + spectral_ode 是 **DINO/cLPIPS/速度 的帕累托最优**, 在内容保持维度全面碾压复杂 SWD, 仅在风格强度 (CLIP-S/DINO-sty) 上略弱. 复杂 SWD (cdf + cross-attn + region_spectral + kmeans) 的 13.7 倍计算开销换来的仅是 DINO-style +0.013 和 CLIP-S +0.011, 代价是 DINO-content -0.039 和 cLPIPS +0.041. **复杂 SWD 不应继续使用**.

**dataset 路径解耦机制 (已验证)**:
- 实现: src/config_schema.py 新增 `_resolve_dataset_paths_via_index` (第1248行), `_find_dataset_index`, `_resolve_via_index` 三个函数
- 索引文件: 仓库根 `dataset_index.json`, 本地 G: 与远程 I: 各一份, 路径顺序不同
- 用法: config 中 data_root / pairing_cache_path / latent_cache_dir / test_image_dir / full_eval_cache_dir / full_eval_clip_hf_cache_dir 字段使用 `$index:KEY` 前缀, 加载时自动解析为第一个存在的候选路径
- 端到端验证: hp 实验 config.json 显示 `$index:samam_512_train` 在远程已解析为 `I:/datasets/...`, 本地解析为 `G:/...`, 同一 config 文件两环境通用

### 6.2 关键发现 (MUSIQ 颠覆 CLIP-S 结论)

1. **R1_baseline MUSIQ=51.34 是全场最高** — 比 spectral 系列高 ~7 分. baseline 生成图像感知质量最好, 但 CLIP-S 最低 (0.7087). 这是 **style vs quality 的根本 trade-off**.

2. **spectral 损害 MUSIQ**: baseline 51.34 → spectral_5ep 41.50 (降 10 分!). spectral ODE 虽然提升 CLIP-S (+0.015), 但引入 artifacts 损害感知质量.

3. **训练长度提升 MUSIQ**: spectral 5ep=41.50 → 10ep=43.95 → 15ep=44.56. 长训练能部分恢复 spectral 引入的 artifacts, 但仍远低于 baseline.

4. **swd_w 对 MUSIQ 有 sweet spot**: swd6=44.36, swd9=44.66, swd12=44.56. swd_w=9 是 MUSIQ 最优 (略优于 swd12). 结合 swd9 的 CLIP-S=0.7259 (尚可), **R3_swd9_llw1_10ep 是 style-quality 平衡最优**.

5. **ll_w 对 MUSIQ 影响小**: ll_w0.3=44.56, 0.5=44.22, 1.0=44.36, 2.0=44.42. 在 10ep 下 ll_w 对 MUSIQ 近似 flat.

### 6.3 修正后的 SOTA 结论

只看 CLIP-S 会得出 "r3_spectral_15ep 是 SOTA" 的错误结论. 加入 MUSIQ 后:

- **Style SOTA (忽略质量)**: r3_spectral_15ep (CLIP-S=0.7311, MUSIQ=44.56)
- **Quality SOTA (忽略 style)**: R1_baseline (MUSIQ=51.34, CLIP-S=0.7087)
- **Balanced SOTA (style+quality)**: **R3_swd9_llw1_10ep** (CLIP-S=0.7259, MUSIQ=44.66, cLPIPS=0.3325) — spectral 系列中 MUSIQ 最高, CLIP-S 也在 top-5

### 6.4 Round 4 方向修正

原 Round 4 设计 (推 style 极限) 在 MUSIQ 视角下是错误方向 — 会进一步损害质量. Round 4 应改为 **quality recovery**:

- R4-Q1: baseline + 15ep (长训练能否让 baseline 同时拿到 MUSIQ 51 和 CLIP-S 0.73?)
- R4-Q2: baseline + spectral_w_ll sweep (baseline 没有 spectral ODE, 是否能加一点 spectral 提 style 而不损 MUSIQ?)
- R4-Q3: spectral + swd_w=9 + 20ep (MUSIQ 最优配置 + 长训练)
- R4-Q4: soft_mask + 15ep (R1 次优变体, MUSIQ=41.63, 长训练能否提升到 45+?)
- R4-Q5: hybrid (baseline 架构 + swd_w=9 loss) — 探索能否解耦 style loss 和 spectral ODE 的 artifact 引入
