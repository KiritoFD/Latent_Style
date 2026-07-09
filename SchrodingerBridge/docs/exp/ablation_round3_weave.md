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
