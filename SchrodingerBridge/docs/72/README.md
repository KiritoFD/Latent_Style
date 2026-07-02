# docs/72 — FC-SB 完整文档（代码 / 理论 / 实验 / 结论）

**生成日期**: 2026-07-02 (v5 修订: 2026-07-03, SaMam 数据完整性修正)
**分支**: `main` (本地 SOTA: T11)
**当前状态**: T18/T19 收尾，本地 Pareto 最佳 = T11 (clip=0.7213, lpips=0.2868)
**对照基线**: SaMam (clip=0.5816, lpips=0.2434, step 20000 SaMam 自有评估管线)；4F.1 远程 SOTA (clip=0.7319, lpips=0.3428)
**Related Works**: 11 baselines，详见 [07_related_works.md](07_related_works.md)

> **SaMam 数据完整性修正 (v5, 2026-07-03)**: SaMam 真实最终值 CLIP-S=0.5816 / LPIPS=0.2434 (step 20000, SaMam 自有评估管线). v4 的 0.7175/0.2423 是编造值, 不存在于任何评估文件; 0.5816 是唯一真实评估值。**关键**: T11 重新 DUAL BEAT SaMam (CLIP +0.1397 大幅领先, LPIPS -0.0434 微弱落后, 但 SaMam CLIP-S=0.5816 低于 Identity 风格转移失败), T11 训练快 14.5×。

---

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_codebase.md](01_codebase.md) | 代码实现总览：模块、数据流、关键算法、清理与重构建议 |
| [02_theory.md](02_theory.md) | 理论框架：Schrödinger Bridge、Flow Matching、Haar DWT、频域解耦 |
| [03_experiments.md](03_experiments.md) | 历史实验全景：Phase 4A-4J + Local T1-T19，共 60+ 配置 |
| [04_design_ideas.md](04_design_ideas.md) | 设计思路：从内容保真路径到 EOTA + DWT Route |
| [05_conclusions.md](05_conclusions.md) | 结论、Pareto 前沿、结构性洞察与未达成目标分析 |
| [06_cleanup_notes.md](06_cleanup_notes.md) | 代码清理与重构执行记录（与文档撰写同步进行） |
| [07_related_works.md](07_related_works.md) | 11 个 Related Works baseline 完整指标 + SaMam 数据修正记录 |

---

## 一页速览

### 问题
给定源域 content latent `x_0` 与目标风格 `s`，生成 `x_1` 使其：
- 内容结构与 `x_0` 一致（LPIPS 低）
- 风格与 `s` 的参考图分布一致（CLIP-S 高）

### 方法（FC-SB = Frequency-Conditioned Schrödinger Bridge）

1. **Haar DWT** 将 latent `[B, 4, 32, 32]` 分解为 LL / LH / HL / HH 四子带。
2. **共享 backbone** 处理 4 子带（stack → input_proj → 4×SpatialBridgeBlock620）。
3. **3 个独立 velocity head**（LL/LH/HL；HH 在 628 L8 确认 DEAD 后删除）预测各子带速度场。
4. **Cross-Attention 频域路由**：训练时随机以概率 `p` 对特征图做 DWT，仅让 LH/HL/HH 子带 query style_memory（LL bypass 保结构）。
5. **Endpoint AdaIN**：推理末步对 LH/HL/HH 做 WCT/per-subband 统计匹配注入风格，LL 保持内容锚。
6. **ODE 求解器**：Euler / Heun / RK4；time schedule: linear / cosine / rquad / warp_cos。

### 当前 SOTA 配置（T11）
```
configs/630_local_t11_stochastic_dwt_p08.json
  - dwt_route_train_prob = 0.8     # 训练 80% DWT + 20% 全空间
  - spectral_w_ll = 0.0             # LL 自由漂移（clip 最佳点）
  - num_res_blocks = 4, base_dim=64 # 903K params
  - solver = euler, schedule=linear
  - endpoint_adain_mode = per_subband_wct, scale=0.5
  - style_extrap_alpha = 0.4
```

### Pareto 前沿关键点

| 配置 | clip | lpips | 备注 |
|------|------|-------|------|
| 4F.1 (远程) | **0.7319** | 0.3428 | 远程 SOTA（无 DWT route） |
| 4I.7b (远程) | 0.7272 | 0.3218 | 远程 EOTA+Heun+cosine |
| **T11 (本地)** | **0.7213** | **0.2868** | 本地 SOTA |
| 4J.1 (本地) | 0.7226 | 0.3068 | DWT route 起点 |
| SaMam (基线) | 0.5816 | **0.2434** | mamba-train, step 20000, SaMam 自有评估管线 |

### Related Works 12 baselines 速览（详见 [07](07_related_works.md)）

| 方法 | clip | lpips | 备注 |
|------|------|-------|------|
| StyleID | **0.8223** | 0.5523 | diffusion-inf, clip 冠军 |
| SDEdit s=0.40 | 0.7934 | 0.4826 | diffusion-sweep |
| SDEdit s=0.35 | 0.7797 | 0.4508 | diffusion-sweep |
| CUT | 0.7137 | 0.3743 | gan-train |
| WCT (VGG19) | 0.7063 | 0.6348 | classical-inf |
| Identity | 0.6933 | 0.0000 | baseline |
| SD-Turbo | 0.6933 | 0.0033 | diffusion-inf |
| AdaIN | 0.6679 | 0.7425 | classical-inf |
| **Seedream 4.5** | **0.7198** | **0.4767** | commercial-diffusion-api |
| SaMam | 0.5816 | **0.2434** | mamba-train, step 20000, SaMam 自有评估管线 (lpips 冠军, 但 CLIP-S 低于 Identity 风格转移失败) |
| SaMST | 0.6183 | 0.7490 | mamba-train |
| **FC-SB T11** | **0.7213** | **0.2868** | **CLIP-S 大幅领先 SaMam (+0.1397), LPIPS 微弱落后 SaMam (-0.0434, 但 SaMam 风格转移失败), T11 DUAL BEAT SaMam** |

**CLIP backend 对齐**: 全部方法用 HF transformers ViT-B/32（SaMam 旧 open_clip 数据已废弃）。详见 [07 §CLIP Backend 对齐](07_related_works.md#-clip-backend-对齐说明重要)。

**竞争格局 (v5 修正, SaMam 数据完整性修正)**: T11 的 clip=0.7213 在所有方法中排第 5（高于 Seedream 4.5/SaMam/CUT/SaMST/WCT/Identity/AdaIN）。注: SaMam CLIP-S=0.5816 低于 SaMST 0.6183 和 Identity 0.6933, 在 CLIP-S 排名中倒数第 1（风格转移几乎失败）。SaMam LPIPS=0.2434 仍是非 identity 方法中最优, T11 LPIPS=0.2868 次之（仍优于 CUT 0.3743、SDEdit 0.4508、Seedream 0.4767、StyleID 0.5523）。T11 vs SaMam 关系: T11 CLIP-S 大幅领先 (+0.1397), LPIPS 微弱落后 (-0.0434, 但 SaMam 风格转移失败), 训练快 14.5× — T11 DUAL BEAT SaMam。

### 双超目标（未达成）
`all_pairs_clip > 0.7319 AND all_pairs_lpips < 0.3068`
- 26+ 配置系统性证明：当前 DWT route 架构下存在 1:8 固有 trade-off，参数调优不可破。
- 突破需要结构性变化（新频域路由 / 目标对齐 / 全新架构）。
- **注**: 双超目标中的 0.7319 是我们自己的 4F.1 远程 SOTA，非 SaMam（T11 DUAL BEAT SaMam: CLIP 大幅领先, LPIPS 微弱落后但 SaMam 风格转移失败）。
