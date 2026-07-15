# 架构演变史

## 总览时间线

```
2026-01 ─── SA-Flow (纯Conv U-Net, Flow Matching)
   │         ↓ transformer fail, 回退到conv
   │         ↓ 加OT匹配 (Hungarian)
   │
2026-01 ─── LGT-X (CrossAttn → Layer Collapse → 纯AdaGN)
   │         ↓ cross_attn 1 token → softmax恒=1.0 → 无选择性
   │         ↓ 共享StyleController → Layer Collapse
   │         ↓ 回退纯AdaGN: 稳定但弱
   │
2026-02 ─── Cycle-GAN-Wasserstein (C-G-W)
   │         ↓ 去掉StyleAdaptiveSkip, 换concat_conv
   │         ↓ 加Swin shift window attention
   │         ↓ clip_style ≈ 0.667 (低于前代0.72)
   │
2026-03 ─── LatentAdaCUT (Dual Attention)
   │         ↓ 64-token style vocabulary + sharpen_scale=2.5
   │         ↓ CrossAttnAdaGN + SpatialSelfAttention
   │         ↓ clip_style ≈ 0.72 (回到前高水平)
   │
2026-04 ─── IN灾难 + Bottleneck Painting
   │         ↓ InstanceNorm杀注意力: aent=0.99, amax=1/256
   │         ↓ Q/K必须twin-norm (同GroupNorm处理)
   │         ↓ 从basic重新开始
   │
2026-05 ─── Schrödinger Bridge (LANCET backbone)
   │         ↓ 时间条件向量场 v_theta(z_t, t, style)
   │         ↓ OMF mode → black-dot → 大清洗
   │         ↓ Phase 1 cleanup: losses 942→340行
   │         ↓ 纯Flow Matching: L_flow + L_kinetic + L_terminal_swd
   │
2026-06 ─── Distinct5 (LANCET + Tokenizer)
   │         ↓ solver_pc 33 epochs, ceiling 0.701
   │         ↓ solver_unsb_cycle, round1 sweep
   │         ↓ 架构本身就是瓶颈
   │
2026-06 ─── 616/618/619 诊断
   │         ↓ OT failure: PureLatentSpatial tokenizer = zero ROI
   │         ↓ External: CSGO/StyleShot/StyleGallery分析
   │         ↓ 5个致命缺陷确认, Golden Path定义
   │
2026-06 ─── 620 Spatial Bridge
              ↓ 完全重写: SpatialBridge620
              ↓ DINOv2 style encoder + CrossAttn + AdaLN(time)
              ↓ clip_style ≈ 0.67-0.71
              ↓ 白化问题: WFI 0.39-0.50
              ↓ Endpoint Shrinkage: latent_alpha≈0.163
```

---

## 阶段1: SA-Flow (2026-01-13 ~ 01-27)

### 架构
```
SAFModel (SA-Flow v5):
  time_mlp:     Linear(1→256) → SiLU → Linear(256→256)
  style_emb:    Embedding(num_styles=2, 256)
  cond_encoder: 3× Conv(256,3×3) + SiLU  (处理x_content)
  stem:         Conv(4→256,3×3)
  8× SAFBlock:  Conv(256,7×7) + GroupNorm + condition_add(t_emb+s_emb) + skip(x_cond)
  final:        Conv(256→4,3×3) [ZERO INIT]
```

### Loss
- Flow Matching MSE: `L = MSE(v_pred, x_style - x_content)`
- 插值: `x_t = (1-t)*x_content + t*x_style + noise*0.01`

### 关键事件
| 日期 | Commit | 事件 | 结果 |
|------|--------|------|------|
| 01-15 | `810cdc32` | DiT尝试 | **失败** — PatchEmbed/FinalLayer reshape bug |
| 01-15 | `810cdc32` | 回退SA-Flow v5 | 可训练但弱 |
| 01-19 | `4f79b147` | 加OT匹配(Hungarian) | **有效** — 减少轨迹交叉 |
| 01-22 | `a19ca870` | 大batch=240 + group conv + grad checkpoint | infra飞跃 |
| 01-25 | `3a3dad85` | "good results 250epoch" | 最佳基线 |
| 01-27 | `d82ea9af` | "风格弱，噪点严重" | 纯conv的天花板 |

### 关键教训
1. **DiT patchification的6D reshape极易出bug** — 应对：先用简单conv验证pipeline
2. **OT匹配是Flow Matching的标准改进** — 减少trajectory crossing
3. **纯conv U-Net无法做per-position style retrieval** — 风格注入太弱

---

## 阶段2: LGT-X (2026-01-28 ~ 02-01)

### 架构 (CrossAttn版)
```
LGT-X (Cross-Attention Enhanced):
  StyleCrossAttention:
    Q: image features [B,C,H,W] → [B,N,C]
    K/V: style_emb [B,256] → Linear(256→2C) → 1 token
    Attn: softmax(Q @ K^T) → uniform (1 token → softmax≡1.0)
  
  StyleController: 3个分辨率共享参数生成器
  CCMLite: 低秩通道混合 (rank=12)
  LGTXBlock: cross_attn + Conv + CCMLite
```

### 架构 (纯AdaGN版, 01-31)
```
LGT-X (Robust Edition):
  LGTXBlock: AdaGN(x, style) → SiLU → Conv → AdaGN → SiLU → Conv → residual
  每层独立style_proj (Linear(style_dim→2C))
  无StyleGate, 无shared controller
  Self-attention只在bottleneck 8×8
```

### 关键事件
| 日期 | Commit | 事件 | 结果 |
|------|--------|------|------|
| 01-28 | `c043767` | 加cross_attn + AdaGN + StyleController | "风格强多了" |
| 01-28 | `11798d4` | MSE爆炸 | 学习率问题 |
| 01-31 | `7ef0105` | **去掉cross_attn, 回AdaGN** | 稳定但弱 |
| 01-31 | `5605964` | "Cross-Attention在纯风格迁移中可能导致内容语义过度纠缠" | 理论反思 |

### Cross-Attention失败的根本原因
1. **1 token K/V** → softmax永远=1.0 → 等价于线性投影，无选择性
2. **共享StyleController** → 所有层得到相似参数 → Layer Collapse
3. **CCMLite依赖StyleController** → controller崩溃时CCMLite也无用

### 关键教训
1. **Cross-attention必须有多token K/V** — 1 token = 无attention
2. **独立per-layer参数 > 共享参数生成器** — 防止Layer Collapse
3. **Cross-attention需要条件化到style而非content** — 否则content被覆盖

---

## 阶段3: C-G-W Backbone (2026-02 ~ 03)

### 架构
```
Cycle-G-W:
  去掉StyleAdaptiveSkip (gated erase/rewrite)
  换concat_conv skip fusion + GroupNorm
  body: global_attn, decoder: window_attn
  SWD patches: [7,11,15,19,25] (coarse only)
```

### 关键事件
| 日期 | Commit | 事件 | 结果 |
|------|--------|------|------|
| 02-04 | `a08aa6d` | CNN分类器评估 | **差** |
| 02-06 | `c25c46d` | 简化代码，修正训练目标 | 基线 |
| 02-09 | `eddaa5c` | 从ckpt逆向，回滚到风格发挥作用的版本 | 推动了 |
| 02-10 | `9e7362b` | "风格确实好了，雾也解决了" | **突破** |
| 02-10 | `29ef531` | "Cycle改MSE是对的" | 重要发现 |
| 02-16 | `7535a9c` | overfit50效果很好 | 过拟合有信号 |
| 02-16 | `54d120e` | **structure loss完全没用** | 删除 |
| 03-30 | `ef38af3` | **全部换C-G-W backbone** | 架构大改 |
| 03-30 | `068584f` | Swin shift window attention | 小改进 |

### 关键教训
1. **Cycle loss → MSE是对的** — 对抗loss在latent space不稳定
2. **Structure loss无用** — identity loss已经隐含了结构约束
3. **C-G-W backbone反而退步** — clip_style从0.72降到0.667

---

## 阶段4: LatentAdaCUT Dual Attention (2026-03-29 ~ 04-07)

### 架构
```
LatentAdaCUT with Dual Attention:
  CrossAttnAdaGN:
    style_tokens_basis = Parameter(randn(64, dim) * 0.02)  ← 64 token vocabulary
    style_proj = Linear(style_dim→dim)                       ← style code biases vocabulary
    tokens = basis + style_proj(style_code)
    Q: content features → Linear(C→C) → multi-head
    K/V: tokens → Linear(dim→2C) → split → multi-head
    sharpen_scale = 2.5 (multiplies attention logits)
    FFN: Linear→SiLU→Linear + residual
    gamma = Parameter(zeros) ← zero-init gate

  SpatialSelfAttention:
    global_attn / window_attn modes
    Swin-style shifted window

  AttentionBlock: AdaGN → SelfAttn → FFN
  
  Body: 3× AttentionBlock(global_attn) @ 16×16
  Decoder: 2× SimpleResBlock
```

### IN灾难 (2026-04-06)
```
问题: InstanceNorm使features变成白噪声
  x_norm = (x - mean) / std → 每channel独立标准化 → 零均值单位方差
  Q·K dot product of white noise → Var = 1/d = 1/256 → softmax均匀
  aent = 0.99 (99%最大熵), amax = 1/256 (完全均匀注意力)

修复:
  1. 去掉attention Q/K上的InstanceNorm → 只用GroupNorm
  2. Q/K twin-norm: norm_x和norm_s分别GroupNorm → 同空间
  3. Style通过同一encoder编码 → 不是1×1 conv投影
  4. L2 normalize Q/K → cosine similarity
  5. 可学习temperature → 不是固定0.08
  6. Zero-init gamma/gate → 稳定起步
  7. SWD loss也去掉IN
  8. Identity loss保留IN但加eps=1e-3
```

### 关键事件
| 日期 | Commit | 事件 | 结果 |
|------|--------|------|------|
| 03-29 | `60b3bfe` | **64-token cross-attention + sharpen** | "效果明显" |
| 03-30 | `ef38af3` | C-G-W backbone | clip_style≈0.667(退步) |
| 04-06 | `babf33e` | twin-norm fix | 诊断了根源 |
| 04-06 | `cd8cb2b` | **去掉IN** | 核心修复 |
| 04-07 | `60ee4c6` | "问题太严重，从basic开始" | 重启 |

### 关键教训
1. **64-token vocabulary > 1-token** — 多token才有选择性attention
2. **Sharpen scale防止soft attention** — 2.5× logits使softmax更peaky
3. **InstanceNorm是attention的毒药** — 白化features → 均匀attention
4. **Q/K必须twin-norm** — 同一normalization pipeline才能匹配
5. **Style必须经过同encoder** — 1×1 conv投影引入domain mismatch
6. **Zero-init gate保证稳定起步** — 从AdaGN-only状态渐进学习attention

---

## 阶段5: Schrödinger Bridge (2026-05-07 ~ 05-19)

### 架构
```
TimeConditionedLANCETBridge (extends LatentAdaCUT):
  核心概念: LANCET feature backbone = 时间条件向量场估计器
  v_theta(z_t, t, style) → velocity at time t given bridge state

  Time conditioning: sinusoidal embedding → MLP → added to style_code
  style_code + time_code → cond_emb
  
  推理模式:
    endpoint_map(): 单步 x + v * horizon (Euler at t=1)
    integrate(): 多步Euler积分沿[0,1]概率路径
  
  style_strength = integration horizon (不是heuristic scaling)
```

### Loss演变
```
初始(05-07): L_flow + L_terminal_swd(0.1) + L_color(15) + L_repulsive(1)
05-08加量:   L_kinetic(2) + L_terminal_swd(25) + L_color↓(10) + L_repulsive↓(0.1)
05-08 kitchen sink: +L_patch_nce + L_low_freq + L_cycle + L_semantic_swd
05-09 black-dot: 全频SWD推到边缘 → NCE/rep导致数值爆炸 → 加clamp/sanitize
05-19 大清洗: 去掉所有heuristic losses → L_flow + L_kinetic + L_terminal_swd
```

### Black-Dot问题
- **根因**: 全频SWD高权重 → velocity/endpoint极端值 → NCE/repulsive放大 → NaN/黑点
- **修复**: nan_to_num + clamp + epsilon调整
- **探测实验结果**:
  | Probe | clip_style | LPIPS | 判定 |
  |-------|-----------|-------|------|
  | base | 0.694 | 0.548 | 基线 |
  | +cycle | 0.693 | 0.545 | 可忽略 |
  | +NCE | 0.674 | 0.434 | **摧毁风格** |
  | +repulsive | 0.695 | 0.550 | 无帮助 |

### Phase 1 Cleanup (05-19, 4个commit 15分钟内)
| 操作 | 删减 | 保留 |
|------|------|------|
| losses.py | 942→340行 | L_flow, L_kinetic, L_terminal_swd, L_curvature(可选) |
| ot_cost.py | 413→260行 | 纯full-band SWD, 无micro/macro分解 |
| trainer.py | log columns 33→12 | 核心metrics only |
| config.json | -7 params | w_curvature=0.0 (新加) |

**被删除的losses**: L_color, L_repulsive, L_patch_nce, L_cycle, L_low_freq, L_low_freq_structure, _freq_split, _barycentric_target, _cosine_lock_loss, OMF mode全部

---

## 阶段6: Distinct5 LANCET (2026-06-01 ~ 06-15)

### 架构
- LANCET backbone + factorized tokenizer (identity 24d + texture 32d + geometry 24d)
- solver_pc: 33 epochs training, ceiling clip_style=0.701
- solver_unsb_cycle: 扩展到epoch 30
- round1 sweep: 系统化变体探索

### 关键实验结果
| 变体 | clip_style | LPIPS | 判定 |
|------|-----------|-------|------|
| solver_pc best | 0.701 | - | 天花板 |
| gated-spade | ~0.67 | - | 无突破 |
| attn_pnp | ~0.67 | - | 无突破 |
| SMoE translator | 0.6728 | 0.3272 | 平衡但弱 |

### 结论
**架构本身就是瓶颈** — 不是超参数问题，是LANCET的表达力限制

---

## 阶段7: 616/618/619诊断 (2026-06-16 ~ 06-19)

### 616: OT失败诊断
- **PureLatentSpatial tokenizer = ZERO ROI** — 零收益
- **OT structure cost → TopoGate** — 用endogenous cross-attention替代
- **virtual_length_multiplier放错section** — 被静默忽略
- **结论**: structure(LPIPS 0.31)已解决，style injection是唯一瓶颈

### 618: 外部方法分析
- **4个方法深度分析**: StyleGallery, CSGO, SCSA, StyleShot
- **三层问题框架**: L1(表示) → L2(结构) → L3(注入)
- **关键发现**: IDT CLIP-S=0.68 — Distinct5比wikiart512有更大的style gaps
- **TopoGate blend=1.0 blocks all style in attention** — 致命设计缺陷

### 619: 系统诊断
- **5个致命缺陷确认**:
  1. Latent pixel Sinkhorn重排破坏VAE连续性
  2. 时间/风格条件未分离
  3. Cross-attention不正确
  4. 独立耦合导致mean collapse
  5. OT在Euclidean空间失效
- **Golden Path**: Independent Coupling + AdaLN(time) + CrossAttn(style) + Flow Matching

---

## 阶段8: 620 Spatial Bridge (2026-06-19 ~ 至今)

### 架构
```
SpatialBridge620:
  StyleConditioner620:
    DINOv2 patch tokens → LoRA adapter → memory tokens (256)
    null tokens (std=0.02), modality dropout
    → style_latent [B, 256, dim]

  SpatialBridgeBlock620 (6层):
    AdaLN(time) → SelfAttention → residual
    CrossAttention(style_latent) → tanh(style_gate) * attended → residual
    FFN → residual
    Pre-CrossAttn FiLM (line 237-243)

  Endpoint head:
    predict_endpoint(t=0) → source_low + target_high
    FiLM endpoint heads (H4): style modulation inside head trunk
    可选: endpoint_lowhigh, endpoint_film, ep_hd=512

  Loss: SpatialBridgeObjective620
    L_flow + single_step_swd_weight=8 + edge_weight=0.1
```

### 关键实验结果
| 实验 | clip_style | LPIPS | WFI | 特点 |
|------|-----------|-------|-----|------|
| 620_swd12 (8ep) | 0.6725 | 0.2968 | - | SWD宽度扫描最优 |
| 620_film_formal (5ep) | 0.6723 | 0.2915 | 0.5037 | FiLM有效 |
| 620_film_v5_hd512 (1ep) | - | - | **0.3906** | WFI最优 |
| 620_lowswd_formal (2ep) | **0.6751** | 0.2781 | - | AP_style最高0.7084 |
| 620_intrinsic_v2 (8ep) | 0.6717 | 0.3678 | - | 内禀cross-attn |
| 620_lowmix05 (1ep) | 0.6765 | 0.3492 | - | 最高transfer clip_style |

### 白化问题
- **Endpoint Shrinkage**: endpoint只走16%目标方向 (latent_alpha≈0.163)
- **高频方向为负**: high_alpha≈-0.050
- **Style信号存在但未被利用**: style_sensitivity≈8.75
- **3 epoch WFI恶化**: 0.4271→0.4532→0.4680 — 训练越久白化越严重
- **44个实验clip_style范围极窄**: 0.699-0.707

### 620的关键教训
1. **Open-set style (DINO) 打破了固定风格数限制** — 但style信号仍然弱
2. **FiLM endpoint有效降低白化** — WFI 0.50→0.39，但训练后恶化
3. **Cross-attention entropy=6.24 (near-uniform)** — attention仍然太软
4. **Style gate=0.048** — 注入太弱
5. **clip_style不是好指标** — 白化图可能clip_style很高
6. **WFI才是白化的直接衡量** — mean_pixel, std_pixel, channel_var
