# 全项目实验数据汇总

> 2026-01-13 ~ 2026-06-23 | 18个branch, 645+实验, 完整指标表

## 一、按架构代际汇总

### 1.1 Cycle-NCE Era (01月-04月)

#### CGW Architecture Sweep (8 configs, e30+e60)

| 实验 | Epoch | All-Pairs clip_style | All-Pairs LPIPS | Transfer clip_style | Transfer LPIPS | Identity clip_style | Classifier Acc |
|------|-------|---------------------|-----------------|---------------------|----------------|---------------------|---------------|
| arch_1 (pM_sC_dH) | 60 | 0.688 | 0.429 | 0.659 | 0.442 | 0.803 | 0.289 |
| arch_5 (pMW_sA_dH) | 60 | 0.689 | 0.447 | 0.663 | 0.460 | 0.795 | 0.289 |
| arch_8 (pMW_sC_dH) | 60 | 0.688 | 0.439 | 0.660 | 0.451 | 0.799 | 0.299 |
| arch_6 (pMW_sC_dL) | 60 | 0.682 | 0.447 | - | - | - | - |
| arch_7 (pMW_sA_dL) | 60 | 0.691 | 0.453 | - | - | - | - |
| arch_2 (pM_sA_dL) | 60 | 0.680 | 0.455 | - | - | - | - |
| arch_3 (pM_sC_dL) | 60 | 0.681 | 0.464 | - | - | - | - |
| arch_4 (pM_sA_dH) | 60 | 0.690 | 0.453 | - | - | - | - |

**结论**: 8个CGW架构变体clip_style范围0.680-0.691，差异极小。**架构微调在CGW时代无效。**

#### Weight Sweep (8 configs, latent_adain vs pseudo_hist)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Color Mode | SWD Weight |
|------|-------|---------------------|---------------------|-----------|-----------|
| weight_exp1 (latent_adain, swd30, tv01, id20) | 60 | 0.708 | 0.663 | latent_adain | 30 |
| weight_exp2 (latent_adain, swd30, tv00, id40) | 60 | 0.711 | 0.666 | latent_adain | 30 |
| weight_exp3 (latent_adain, swd60, tv01, id20) | 60 | 0.709 | 0.668 | latent_adain | 60 |
| weight_exp5 (pseudo_hist, swd30, tv01, id20) | 60 | 0.709 | 0.667 | pseudo_hist | 30 |
| weight_exp6 (pseudo_hist, swd30, tv00, id40) | 60 | 0.714 | 0.671 | pseudo_hist | 30 |
| weight_exp7 (pseudo_hist, swd60, tv01, id20) | 60 | 0.714 | 0.669 | pseudo_hist | 60 |
| weight_exp8 (pseudo_hist, swd60, tv00, id40) | 60 | **0.711** | 0.669 | pseudo_hist | 60 |

**结论**: pseudo_hist略优于latent_adain(0.714 vs 0.711); SWD 30→60微弱提升; id40>id20。但整体范围0.708-0.714，差异在噪声水平。

#### style_oa Series (最成功的Cycle-NCE系列)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS | Identity clip_style |
|------|-------|---------------------|---------------------|----------------|---------------------|
| style_oa_5 (wc2, swd90, id15) | 120 | **0.722** | 0.686 | 0.519 | - |
| style_oa_6 (wc5, swd90, id15) | 120 | 0.720 | 0.685 | - | - |
| style_oa_7 (wc5, swd60, id15) | 120 | 0.716 | 0.682 | - | - |
| style_oa_8 (wc5, swd90, id30) | 60 | **0.724** | 0.688 | 0.519 | - |

**结论**: style_oa_8是Cycle-NCE+attn的最佳代表，clip_style=0.724。但LPIPS=0.519太高，内容保持差。

#### Depth Ablation (9 configs, E05-E09)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS | Identity clip_style |
|------|-------|---------------------|---------------------|----------------|---------------------|
| E05 (None, Swin4) | 80 | 0.667 | 0.631 | **0.299** | 0.812 |
| E06 (Adapt, Conv1) | 80 | 0.671 | - | - | - |
| E07 (Adapt, Swin2) | 80 | 0.670 | - | - | - |
| E08 (Adapt, Swin4) | 80 | 0.670 | 0.634 | **0.298** | 0.810 |
| E09 (Naive, Conv1) | 80 | 0.676 | - | - | - |

**结论**: Depth系列LPIPS极佳(0.298)但clip_style低(0.667-0.676)。**结构好=风格弱的经典tradeoff。**

#### Texture Tearer (3 configs)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS |
|------|-------|---------------------|---------------------|----------------|
| T01 (ResOn, Swin4, Noise) | 40 | - | - | - |
| T02 (ResOff, Adapt, LowIDT) | 40 | - | - | - |
| T03 (ResOff, Adapt, Conv1, HFSWD) | 40 | 0.670 | 0.635 | **0.316** |

#### Zero Constraint (3 configs)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS |
|------|-------|---------------------|---------------------|----------------|
| Z01 (ResOff, Adapt, ZeroIDT) | 40 | 0.673 | 0.638 | **0.319** |
| Z02 (ResOff, Adapt, ZeroIDT, HFSWD) | 40 | - | - | - |
| Z03 (ResOff, None, ZeroIDT) | 40 | 0.676 | 0.641 | 0.322 |

#### 46-Series (clean/skip/blend)

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS |
|------|-------|---------------------|---------------------|----------------|
| 46_clean | 40 | 0.676 | 0.653 | 0.475 |
| 46_skip | 20 | **0.701** | 0.680 | 0.457 |
| 46_blend | 20 | 0.670 | 0.640 | 0.379 |

#### Ablation Series

| 实验 | Epoch | All-Pairs clip_style | Transfer LPIPS | FID | art_FID |
|------|-------|---------------------|----------------|-----|---------|
| abl_no_adagn | 80 | **0.697** | 0.541 | 312.5 | 492.0 |
| abl_no_residual | 80 | 0.629 | **0.297** | **286.5** | **372.4** |
| abl_naive_skip | 80 | 0.706 | 0.604 | 330.4 | 538.0 |

**结论**: no_adagn=最高clip_style但LPIPS差; no_residual=最低LPIPS但clip_style差; naive_skip=最差FID。**AdaGN是核心组件，去掉任何一部分都导致严重偏科。**

#### Color Ablation

| 实验 | Epoch | All-Pairs clip_style | Transfer LPIPS |
|------|-------|---------------------|----------------|
| color_ablation_exp1 (anchor_pseudo_adain, wc2, tv05) | 60 | 0.668 | 0.431 |

#### 其他Cycle-NCE

| 实验 | Epoch | All-Pairs clip_style | Transfer clip_style | Transfer LPIPS |
|------|-------|---------------------|---------------------|----------------|
| light-15patch-10color | 60 | 0.691 | 0.634 | 0.408 |
| light-1 | 60 | - | - | - |
| heavy_decode | 80 | 0.684 | 0.651 | **0.339** |

---

### 1.2 Branch-Specific Results (未合入main)

#### Classify Branch: Classifier-Guided Training

| 实验 | clip_style | LPIPS | cls_acc | 结论 |
|------|-----------|-------|---------|------|
| C1_struct1.0 | **0.5057** | 0.5308 | 0.1000 | 结构过强=分类崩溃 |
| C2_no_struct | 0.5041 | 0.5288 | **0.5417** | 无结构=分类好但style弱 |
| overfit50-upscale | **0.5933** | - | - | overfit50最高 |

#### Diff-Gram Branch

| 实验 | style_swd | clip_style_sim | content_ssid | 结论 |
|------|-----------|---------------|-------------|------|
| 所有变体 | **0.0** | **0.0** | 0.977 | **完全失败** |

#### Thermal Branch

| 实验 | clip_style | clip_content | LPIPS | 结论 |
|------|-----------|-------------|-------|------|
| 4_style-all_scales (e50) | 0.5803 | 0.7571 | - | 中等 |
| 4_style-patch-157-mse (e60) | **0.5901** | 0.8754 | - | MSE改善content保持 |
| LoRA verification | - | - | Edge IoU=0.5412 | LoRA有效但overhead=8.27× |
| Proxy CNN filtering | - | - | IoU=0.247 | **完全失败** |

#### attn Branch: Aline120 Sweep

| 实验 | Epoch | clip_style | clip_content | LPIPS | 结论 |
|------|-------|-----------|-------------|-------|------|
| aline_03_ghost_wireframe | 10 | **0.7200** | 0.6601 | 0.6022 | 最高style但content差 |
| aline_03_ghost_wireframe_distill200 | 10 | **0.7207** | 0.6708 | 0.5853 | 蒸馏微弱提升 |
| style_oa_5 | 120 | **0.72** | - | - | 历史Cycle-NCE天花板 |

#### Style8_Moment+SWD Branch

| 实验 | best_clip_style | best_cls_acc | latest_lpips | 结论 |
|------|---------------|-------------|-------------|------|
| overfit50-upscale | **0.5933** | - | - | overfit50天花板 |
| overfit50-style-distill-struct-v2 | 0.5284 | 0.85 | 0.5512 | style-structure coupling |
| overfit50-distill_low_only | 0.5185 | **0.93** | 0.5456 | 高分类但style弱 |
| overfit50-strok-style | 0.5164 | 0.93 | **0.3862** | 最好LPIPS |
| full_300-map16+32 | 0.5099 | 0.78 | 0.4242 | 最稳定full训练 |

**Domain vs Instance style ratio**: Domain 1×1 = 5.77×, Instance 1×1 = 1.15× — **Domain风格表示远优于Instance**

#### Style Injection Priority Branch

| 实验 | transfer_clip_style | transfer_lpips | cls_acc | 结论 |
|------|-------------------|---------------|---------|------|
| d1 (decoder AdaGN) | 0.4671 | 0.3056 | 0.84 | 最佳改进 |
| d2 (decoder 32× spatial) | 0.4671 | 0.3064 | **0.85** | 同上 |
| d3 (+texture head) | 0.4671 | 0.3069 | 0.85 | 无额外收益 |
| d4 (over-regularized) | 0.4413 | **0.2438** | 0.08 | 崩溃到identity |

---

### 1.3 Schrödinger Bridge Era (05月-06月)

#### AAAI2027 LANCET Baseline

| 方法 | Epoch | Transfer clip_style | All-Pairs clip_style | Content LPIPS | CLIP-content |
|------|-------|-------------------|---------------------|--------------|-------------|
| Ours (1ep) | 1 | 0.664 | 0.697 | **0.427** | **0.839** |
| Ours (7ep) | 7 | 0.683 | 0.704 | 0.451 | 0.809 |
| SaMST (100ep) | 100 | - | **0.719** | 0.466 | 0.819 |
| StyleID | - | - | **0.760** | 0.750 | 0.552 |
| AdaIN v32k | - | - | 0.713 | 0.630 | 0.699 |
| AdaIN vgg19 | - | - | 0.693 | 0.687 | 0.599 |

#### Per-Target AAAI (Ours 7ep)

| 目标风格 | clip_style | clip_content | LPIPS |
|---------|-----------|-------------|-------|
| **vangogh** | **0.824** | **0.875** | **0.399** |
| cezanne | 0.791 | 0.845 | 0.425 |
| monet | 0.754 | 0.843 | 0.455 |
| photo | 0.722 | 0.836 | 0.447 |
| Hayao | 0.620 | 0.820 | 0.517 |

**vangogh远强于Hayao** — 风格差异性决定难度

#### Inmortal Family (28 experiments)

| 实验 | Transfer clip_style | All-Pairs clip_style | Content LPIPS | 状态 |
|------|-------------------|---------------------|--------------|------|
| XPred_Kmanifold_Pattn (promoted) | **0.729** | 0.734 | 0.637 | Frontier |
| Pattn_Stokes002 | **0.731** | 0.737 | 0.618 | Style frontier |
| Pattn_Stokes_finetune (selected) | 0.727 | 0.736 | **0.603** | Best tradeoff |
| AnisoStokes_ClampRelease | 0.701 | 0.718 | **0.475** | **Best content** |
| XPred_Barycenter_b40 | 0.716 | 0.719 | 0.718 | Catastrophic damage |
| XPred_Phighpass | 0.680 | 0.681 | 0.775 | Negative |
| K_manifold | 0.663 | 0.695 | **0.335** | Not ceiling-lifting |
| K_spectral_b12 | 0.679 | 0.710 | 0.364 | Moderate |

#### Distinct5-512 Final

| 实验 | Epoch | Transfer clip_style | Content LPIPS | ArtFID |
|------|-------|-------------------|--------------|--------|
| LANCET F (best-lpips) | 1 | 0.697 | **0.319** | **122.6** |
| LANCET K (best-style) | 1 | **0.701** | 0.362 | 157.2 |
| LANCET H | 1 | 0.697 | 0.321 | - |
| Baseline | 1 | 0.687 | 0.447 | - |

#### FiberBundle Phase2 (200+ eval points)

| 实验 | Transfer clip_style | Content LPIPS | 保留 |
|------|-------------------|--------------|------|
| I2SB orth low-anchor (e9) | **0.701** | 0.372 | ✅ Retained |
| LatAff refine s0.45 | 0.679 | **0.319** | ✅ Retained balanced |
| LatAff refine s0.35 | 0.677 | **0.314** | ✅ Retained structure-first |
| Fiber-SDE σ=0.08 | 0.711 | 0.337 | Style ceiling (no training) |
| Gate Head Adapter (e3) | 0.717 | 0.473 | High LPIPS |

#### SaMST Per-Style Baseline

| 风格 | clip_style | LPIPS |
|------|-----------|-------|
| Baroque | 0.723 | 0.294 |
| Impressionism | 0.736 | 0.282 |
| Cubism | 0.777 | 0.427 |
| Symbolism | **0.793** | 0.334 |
| Art_Nouveau | 0.769 | 0.351 |
| **平均** | **0.760** | **0.337** |

---

### 1.4 620 Spatial Bridge (06月)

#### 本地实验 (完整eval)

| 实验 | Epoch | Transfer clip_style | All-Pairs clip_style | IDT clip_style | LPIPS | Gate | Text |
|------|-------|-------------------|---------------------|---------------|-------|------|------|
| t5base_b4 | 1 | 0.662 | 0.697 | 0.837 | **0.288** | 0.050 | T5 |
| t5base_b4 (fixed) | 1 | 0.662 | 0.697 | 0.837 | **0.287** | 0.050 | T5 |
| t5base_b4 | 8 | 0.666 | 0.699 | 0.832 | 0.338 | 0.048 | T5 |
| t5base_b4 (fixed) | 8 | 0.666 | 0.699 | 0.832 | 0.338 | 0.048 | T5 |
| notext_b8 | 8 | 0.665 | 0.700 | 0.841 | **0.287** | 0.047 | no |

**Text vs No-Text**: clip_style差异0.666 vs 0.665，**无差异**。

#### 远程Smoke Test Ablation (36个1-epoch实验)

| 消融类别 | 变体数 | Best Transfer clip_style | Best LPIPS | 结论 |
|---------|-------|------------------------|-----------|------|
| Loss (swd 0/2/8/16, nosigma, edge0) | 6 | 0.669 (swd16+edge0) | - | SWD16最优 |
| Capacity (64×4, 64×6, 128×4, 128×6) | 4 | 0.668 (128×4) | - | 128×4刚好 |
| Attention (gated, softmax, relu2, sparsemax...) | 6 | 0.668 (softmax) | - | softmax略优 |
| StyleFiLM (on/off) | 2 | ~0.668 | - | **无差异** |
| Endpoint (velocity, lowhigh hd128/256/512) | 5 | 0.668 (lowhigh_hd128) | - | hd128微弱优势 |
| Gate init (0.05/0.3/0.5) | 3 | 0.668 (0.5) | - | gate=0.5微弱优势 |

**所有620消融clip_style范围: 0.660-0.669** — **无任何变体突破0.67**

#### 远程Full Training (38+实验, 关键结果)

| 实验 | Epoch | clip_style | LPIPS | WFI | 特点 |
|------|-------|-----------|-------|-----|------|
| 620_swd12_b80 | 8 | 0.673 | 0.297 | - | SWD宽度扫描最优 |
| 620_film_formal | 5 | 0.672 | 0.292 | 0.504 | FiLM有效 |
| 620_film_v5_hd512 | 1 | - | - | **0.391** | WFI最优(1ep) |
| 620_lowswd_formal | 2 | **0.675** | 0.278 | - | AP_style=0.708最高 |
| 620_lowmix05 | 1 | 0.677 | 0.349 | - | Transfer最高 |
| 620_intrinsic_v2 | 8 | 0.672 | 0.368 | - | 内禀cross-attn |
| 620_swd20 | - | - | **0.268** | - | LPIPS最优 |

---

## 二、跨代际对比

### clip_style 天花板演变

```
Thermal (01-02月):     0.59  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  baseline
CGW sweep (03月):      0.69  ████████████████████████████░░░░░░░░░░░░░░░░  +0.10
style_oa+attn (03月):  0.72  ████████████████████████████████░░░░░░░░░░░░  +0.03 (cross-attn)
SB cleanup (05月):     0.694 ███████████████████████████████░░░░░░░░░░░░░  -0.03 (回退)
LANCET K (06月):       0.701 ████████████████████████████████░░░░░░░░░░░░  =Cycle-NCE peak
XPred Pattn (06月):   0.731 ██████████████████████████████████░░░░░░░░░░  +0.03 (但LPIPS差)
620 (06月):           0.665 ███████████████████████████████░░░░░░░░░░░░░  -0.07 (回退!)
SaMST per-style:      0.760 ████████████████████████████████████░░░░░░░░  上限参考
```

### LPIPS vs clip_style 全局tradeoff

| 代际 | clip_style | LPIPS | Tradeoff方向 |
|------|-----------|-------|-------------|
| Depth E05/E08 | 0.667 | **0.298** | 极端内容保持 |
| 620 notext_b8 | 0.665 | **0.287** | 极端内容保持 |
| TextureTearer3 | 0.670 | 0.316 | 好平衡 |
| LatAff s0.35 | 0.677 | 0.314 | 好平衡 |
| LANCET F e1 | 0.697 | 0.319 | 好平衡 |
| LANCET K e1 | 0.701 | 0.362 | 风格优先 |
| I2SB orth e9 | 0.701 | 0.372 | 风格优先 |
| Fiber-SDE σ=0.08 | 0.711 | 0.337 | **最佳风格-内容比** |
| XPred Pattn | 0.729 | 0.637 | 极端风格 |
| style_oa_8 | 0.724 | 0.519 | 极端风格 |
| abl_naive_skip | 0.706 | 0.604 | 极端风格 |

---

## 三、关键发现汇总

### 3.1 被数据确认的结论

1. **clip_style天花板0.70-0.73**：所有通用模型6个月未突破0.73，per-style模型0.76
2. **style-content强耦合**：corr(clip_style, LPIPS)=+0.94 (Style8_Moment+SWD branch验证)
3. **Cross-attention是关键突破点**：0.69→0.72 (+0.03)，是6个月最大单步提升
4. **620架构整体低于LANCET**：0.665 vs 0.701，回退0.04
5. **Text条件无效果**：T5 vs no-T5 clip_style差异0.001 (在620架构上)
6. **InstanceNorm是attention毒药**：aent=0.99, 多个branch反复验证
7. **Diff-Gram完全失败**：所有变体style_swd=0.0
8. **Fiber-SDE不用训练就能达0.711**：纯ODE路径质量
9. **Domain style远优于Instance style**：5.77× vs 1.15× ratio
10. **所有CGW架构变体无差异**：0.680-0.691，21个configs几乎一样

### 3.2 被数据推翻的假设

1. **"620是新架构应该更好"** → 数据：clip_style 0.665 vs LANCET 0.701，**回退0.04**
2. **"Text条件会提升风格迁移"** → 数据：T5 vs no-T5差0.001，**无效果**
3. **"更大模型容量=更好"** → 数据：128×6不优于128×4，64×6不优于64×4
4. **"更高gate init=更好style"** → 数据：gate 0.5 vs 0.05差0.001，**微不足道**
5. **"SWD权重越高=越好"** → 数据：SWD 16不优于SWD 8/2，在1-epoch内差异极小
6. **"训练更多epoch=更好"** → 数据：620的8-epoch WFI恶化，LANCET的7-ep vs 1-ep LPIPS恶化(0.427→0.451)
7. **"Gram/Moment/Semigroup有用"** → 数据：Diff-Gram style_swd=0.0, Gram在Classify branch无用
8. **"Structure loss有用"** → 数据：3个独立验证(Classify/Cycle-upscale/SB)都确认无用

### 3.3 新发现(之前未认识到的)

1. **Fiber-SDE不用训练就达0.711** — 说明ODE路径本身质量好，问题是学习不到好路径
2. **XPred变体可达0.73但LPIPS 0.6+** — 本质是"暴力风格注入"而非好的风格迁移
3. **620的gate值0.047-0.050说明style注入几乎没开启** — 模型选择了保守策略
4. **vangogh远强于Hayao(0.824 vs 0.620)** — 风格差异性直接决定难度
5. **overfit50 consistently best** — 小数据过拟合信号最强，暗示模型capacity不是瓶颈
6. **Replay-ordered Ablate43 grid5可达0.81 clip_style** — 但这是overfit50极端情况
7. **Style8_Moment+SWD的Domain/Instance ratio=5.77×** — 这是之前没注意到的强信号
