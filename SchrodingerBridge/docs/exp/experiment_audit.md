# FC-SB 实验审计报告 (FCSB)

> 核查范围：`exp/FCSB/{early,local_t,phase4}/` 全部实验子目录。
> 参考文档：`docs/72/03_experiments.md`（实验结论主参考）。
> 核查依据：每个目录的 `config.json` 的 `ablation.notes` + `data.data_root` + `training.num_epochs`，结合 `03_experiments.md` 的结论表。
> 输出目的：判定每个实验的 why / conclusion / ckpt 意义，给出"无意义 ckpt 删除建议清单"。
>
> **重要前置说明**：实际盘盘中存在的实验目录数多于任务给定的列表。本报告以任务给定列表为主进行逐项核查，并在 §5 单独标注"盘盘多出的 SOTA 关键目录"——这些恰好是 ckpt 保留价值最高的目录（T11 / 4I.2b / 4I.7b），务必在删除前对照本报告 §5 处理。

---

## 0. 数据集分类总览

通过 `data.data_root` + `data.virtual_length_multiplier`（vlen）判定：

| 数据集标识 | data_root 关键字 | 分辨率 | vlen | 用途 | 涉及实验 |
|---|---|---|---|---|---|
| **distinct5 (主线)** | `G:/.../Dataset/distinct5_512_latents_ema` | 512 | 1.0 | 全量训练，所有 630 系列正式实验 | clean_base_v2_*、全部 phase4、全部 local_t |
| **wikiarts_5 (smoke)** | `F:\wikiart_distinct5_samam_512_latents_ema` | 512 | **0.04** | 4% 子集 smoke test | task1/task3/task4_style_strength 系列 |
| **wikiarts_5 (full)** | `F:\wikiart_distinct5_samam_512_latents_ema` | 512 | 1.0 | 全量早期迭代 | task4_iter/r1~r7 系列 |
| **fewshot6 (4J.6)** | `G:/.../Dataset/fewshot6_512_latents_ema` | 512 | 1.0 | Few-shot Pop_Art 注入 | 630_phase4j6_fewshot_popart* |

> **关键判定**：vlen=0.04 的 smoke test 实验即使有 ckpt 也无任何性能意义（仅 4% 数据 + 1-2 epoch）；wikiarts_5 (full) 的 task4_iter 早期迭代已被 distinct5 + DWT 路线全面取代。**所有有性能意义的 ckpt 都来自 distinct5 (主线)**。

---

## 1. Early 早期实验 (`FCSB/early/` + `wiki5/`)

| 实验目录 | 数据集 | why（为什么做） | conclusion（结论） | ckpt | 意义判定 |
|---|---|---|---|---|---|
| `clean_base_v2_local/` (ep5,10) | distinct5, 10ep | Phase 4 起点基线：B2 V2 + D2 dwt_haar + D4 extrap/adain，Phase 1 清理前的"原始 baseline" | 03_experiments.md 未直接记录该目录；relu2 notes 引用其 softmax baseline 为 0.7293/0.3203。被 Phase 1D verify + Phase 4 全线取代 | epoch_0005/0010.pt | **删除**（已被取代的起点基线，无对照价值） |
| `clean_base_v2_relu2/` (ep3) | distinct5, 3ep | Phase 1B M9 bug 修复验证：激活 `style_attn_mode=relu2`（原本 bug 默认 softmax） | TDD 修复 PASS，commit 69da87cb0；relu2 vs softmax 对照点 | epoch_0003.pt | **删除**（bug 修复 smoke，结论已记入文档） |
| `task1_endpoint_film_baseline/` (ep1) | wikiarts_5 smoke | Task1：FiLM endpoint heads + GroupNorm 基线 | 03_experiments.md 未记录；smoke 1ep 4% 数据 | epoch_0001.pt | **删除**（smoke test，无性能意义） |
| `task1_endpoint_film_no_norm/` (ep1) | wikiarts_5 smoke | Task1：FiLM 去掉 GroupNorm 对照 | 未记录；smoke | epoch_0001.pt | **删除** |
| `task3_baseline_1ep/` (ep1) | wikiarts_5 smoke | Task3：FiLM×2 + tanh_gate + GN 基线 | 未记录；smoke | epoch_0001.pt | **删除** |
| `task3_combo_a_1ep/` (ep1) | wikiarts_5 smoke | Task3 Combo A：FiLM×2 + fixed_one gate + no GN | 未记录；smoke | epoch_0001.pt | **删除** |
| `task3_combo_b_3ep/` (ep1-3) | wikiarts_5 smoke | Task3 Combo B：Combo A 的 3ep 完整版 | 未记录；smoke | epoch_0001-0003.pt | **删除** |
| `task4_iter/r1a_latent_baseline/` (ep1-2) | wikiarts_5 full | R1-A：no-DINO latent baseline + GN + tanh_gate | 未记录；早期迭代 | epoch_0001/0002.pt | **删除** |
| `task4_iter/r1a_no_dino_minimal/` (ep1-2) | wikiarts_5 full | R1-A：no DINO / no FiLM / no reg 极简基线 | 未记录；早期迭代 | epoch_0001/0002.pt | **删除** |
| `task4_iter/r1b_no_dino_nogn/` (ep1-2) | wikiarts_5 full | R1-B：no GN 对照 | 未记录 | epoch_0001/0002.pt | **删除** |
| `task4_iter/r2a_with_film/` (ep1-2) | wikiarts_5 full | R2-A：加 FiLM | 未记录 | epoch_0001/0002.pt | **删除** |
| `task4_iter/r2b_with_antiwhiten/` (ep1-2) | wikiarts_5 full | R2-B：R2-A + anti-whiten 正则 | 未记录 | epoch_0001/0002.pt | **删除** |
| `task4_iter/r3c_optimal_5ep/` (ep1-5) | wikiarts_5 full | R3-C：R2-B 最优 + 5ep 长训练 | 未记录；pre-DWT 早期最优组合 | epoch_0001-0005.pt | **删除**（5ep 也无意义，被 DWT 路线取代） |
| `task4_iter/r4d1_velmag_high/` (ep1-3) | wikiarts_5 full | R5/R4-D1：velocity direction loss 修 fog | 未记录 | epoch_0001-0003.pt | **删除** |
| `task4_iter/r6_pixel_color_fix/` (ep1-3) | wikiarts_5 full | R6：pixel-space color preservation loss | 未记录 | epoch_0001-0003.pt | **删除** |
| `task4_iter/r7_saturation_loss/` (ep1-5) | wikiarts_5 full | R7：HSV saturation proxy loss | 未记录 | epoch_0001-0005.pt | **删除** |
| `task4_iter/p2_long_10ep/` (ep1-10) | wikiarts_5 full | P2：R4-D1 + 10ep 长训练 + AdaIN 后处理 | 未记录 | epoch_0001-0010.pt | **删除** |
| `task4_style_strength_w05_2ep/` (ep1-2) | wikiarts_5 smoke | Task4：style_strength_reg=0.5 | 未记录；smoke 2ep | epoch_0001/0002.pt | **删除** |
| `task4_style_strength_w10_2ep/` (ep1-2) | wikiarts_5 smoke | Task4：style_strength_reg=1.0 | 未记录；smoke 2ep | epoch_0001/0002.pt | **删除** |
| `628_ablation/` | （无 ckpt） | 628 时代 baseline 的 eval-only 重扫（P8B steps / p8c rescan） | 仅 `p8b_steps_*_summary.json` + `p8c_rescan_results.json`，**无模型 ckpt** | 无 | **保留 JSON**（评估曲线数据），无需删 ckpt |

**Early 小结**：全部早期 ckpt 均无保留价值。`628_ablation/` 仅有评估 JSON 无 ckpt，保留即可。task4_iter 下多个子目录还有嵌套 `src/exp/...` 的污染副本，建议整体清理。

---

## 2. Local T 系列 (`FCSB/local_t/`)

> 全部基于 distinct5 主线、5ep 训练、基于 T11 SOTA (p=0.8, w_ll=0.0) 做消融。

| 实验目录 | why | conclusion（03_experiments.md） | ckpt | 意义判定 |
|---|---|---|---|---|
| `630_local_r1_depth2/` (ep5) | R1 反向消融：depth=2（反 T19a depth=6 NaN） | 未记录（R1-R3 不在 03 文档）；按 notes，验证 depth=4 是否容量甜点 | epoch_0005.pt | **删除**（反向消融，T11 已确认 SOTA） |
| `630_local_r2_dim32/` (ep5) | R2 反向消融：dim=32（反 T19b dim=96 欠拟合） | 未记录；验证 dim=64 甜点 | epoch_0005.pt | **删除** |
| `630_local_r3_gate_init0/` (ep5) | R3 反向消融：gate_init=0.0（去可学习 gate） | 未记录；验证 gate 机制必要性 | epoch_0005.pt | **删除** |
| `630_local_t2_soft_ll_t2a/` (ep5) | T2：Soft LL Route α=0.05，DWT route 早期探索 | 03 文档 §13 未单独记录 T2；后被 T5/T10/T11 取代 | epoch_0005.pt | **删除**（早期 DWT route 探索，已被 T11 取代） |
| `630_local_t5_eval_only_dwt/` (ep5) | T5：训练全空间 query，推理 DWT route | §13.2 FAIL（clip=0.7061, lpips=0.2606）；根因：训练/推理分布失配 | epoch_0005.pt | **删除**（失败方向，结论已记录） |
| `630_local_t14_casi/` (ep5) | T14 CASI：cross-attn 输出统计量 AdaIN | §14.2 MIXED（0.7152/0.2795）；cross-attn 输出仍高频 | epoch_0005.pt | **删除** |
| `630_local_t15_llgqca/` (ep5) | T15 LLGQCA：LL 全局 query cross-attn | §14.3 MIXED（0.7176/0.2764）；clip 渐进提升 | epoch_0005.pt | **删除** |
| `630_local_t18a_wll05/` (ep5) | T18a：w_ll=0.5 恢复 LL velocity 训练 | §15.1 FAIL（0.7174/0.2774）；w_ll>0 是 content-heavy | epoch_0005.pt | **删除** |
| `630_local_t18b_wll10/` (ep5) | T18b：w_ll=1.0 | §15.1 FAIL（0.7180/0.2764） | epoch_0005.pt | **删除** |
| `630_local_t19a_depth6/` (ep5) | T19a：depth=6 增容量 | §15.2 FAIL（NaN，WCT eigh 数值不稳定） | epoch_0005.pt | **删除**（NaN，完全无用） |
| `630_local_t19b_dim96/` (ep5) | T19b：dim=96 增容量 | §15.2 FAIL（0.7207/0.3142，5ep 欠拟合） | epoch_0005.pt | **删除** |
| `630_local_t21_adaln_zero_ll/` (ep5) | T21 Plan C：global AdaLN-Zero 调制 LL mean/std | 未记录（T20-T26 不在 03 文档） | epoch_0005.pt | **删除**（未记录结论，按探索失败处理） |
| `630_local_t22_tone_bias/` (ep5) | T22：tone_bias 色调偏置 | 未记录 | epoch_0005.pt | **删除** |
| `630_local_t23_ll_mean_only/` (ep5) | T23：仅迁移 LL mean（色调不迁移 std） | 未记录 | epoch_0005.pt | **删除** |
| `630_local_t24_ll_std_only/` (ep5) | T24：仅迁移 LL std | 未记录 | epoch_0005.pt | **删除** |
| `630_local_t25_ll_cov_only/` (ep5) | T25：仅迁移 LL cov | 未记录 | epoch_0005.pt | **删除** |
| `630_local_t26_ll_ycbcr/` (ep5) | T26：LL YCbCr 色彩空间迁移 | 未记录 | epoch_0005.pt | **删除** |

**Local T 小结**：除 T11（见 §5）外，本节列出的全部 ckpt 均无意义。T13-T16 + T18-T19 + R1-R3 共同系统性证明"不动 style_mem 无法提取全局风格"+"容量/loss 调优无法破 1:8 trade-off"，结论已记入文档，ckpt 无需保留。T21-T26 系列在 03_experiments.md 中未记录结论，建议补录后再删 ckpt。

---

## 3. Phase 4 系列 (`FCSB/phase4/`)

> 全部 distinct5 主线。

| 实验目录 | why | conclusion（03_experiments.md） | ckpt | 意义判定 |
|---|---|---|---|---|
| `630_phase1d_verify/` (ep2) | Phase 1D：最简 codebase 性能验证 | §2.4 PASS baseline（clip=0.7293, lpips=0.3203），commit 9de1e9e03 | epoch_0002.pt | **删除**（baseline 验证，被后续全线取代） |
| `630_phase1d_verify_v2/` (ep3) | Phase 1D v2：3ep 版本 | 同上 | epoch_0003.pt | **删除** |
| `630_phase2b_mask_random_50/` (ep3) | Phase 2 masking 最佳配置 | §3.2 **最佳**（0.7275/0.3238）；结论：random_50 最佳 | epoch_0003.pt | **删除**（结论已记录，config 可复现） |
| `630_phase2c_mask_random_75/` (ep3) | masking 消融 | §3.2 PASS（0.7268/0.3252） | epoch_0003.pt | **删除** |
| `630_phase2c_mask_shuffle_50/` (ep3) | masking 消融 | §3.2 PASS（0.7259/0.3271） | epoch_0003.pt | **删除** |
| `630_phase2c_mask_shuffle_75/` (ep3) | masking 消融 | §3.2 FLAT（0.7243/0.3284） | epoch_0003.pt | **删除** |
| `630_phase4a2_adain_0/` (ep3) | 4A2 减法消融：endpoint_adain_scale=0 | §5.2 FAIL（0.7082/0.2994）；AdaIN 必要 | epoch_0003.pt | **删除**（关键消融但结论已记录，config 可复现） |
| `630_phase4a2_extrap_0/` (ep3) | 4A2：style_extrap_alpha=0 | §5.2 FAIL（0.7242/0.3333）；extrap 必要 | epoch_0003.pt | **删除** |
| `630_phase4a2_w_ll_0/` (ep3) | 4A2：spectral_w_ll=0 | §5.2 FAIL（0.7117/0.2994，"假阴性"后被 4G.1 澄清） | epoch_0003.pt | **删除** |
| `630_phase4b1_freq_a05/` (ep3) | 4B-1 频域 masking α=0.5 | §6.1 PASS（0.7252/0.3347） | epoch_0003.pt | **删除** |
| `630_phase4b1_freq_a1/` (ep3) | 4B-1 频域 masking α=1.0 | §6.1 PASS（0.7258/0.3357） | epoch_0003.pt | **删除** |
| `630_phase4b1_freq_a1_rand50/` (ep3) | 4B-1 α=1.0 + random=0.5 | §6.1 PASS（0.7264/0.3354） | epoch_0003.pt | **删除** |
| `630_phase4b2_freq_a1_rand30/` (ep3) | 4B-2 ratio 优化 rand=0.3 | §6.2 PASS（0.7250/0.3252, best lpips） | epoch_0003.pt | **删除** |
| `630_phase4b2_freq_a1_rand70/` (ep3) | 4B-2 rand=0.7 | §6.2 PASS（0.7245/0.3284） | epoch_0003.pt | **删除** |
| `630_phase4b3_dwt_a1/` (ep3) | 4B-3 DWT tokenizer α=1.0 | §6.3 PASS（0.7266/0.3402）；正交 Haar 可用 | epoch_0003.pt | **删除** |
| `630_phase4b3_dwt_a1_rand50/` (ep3) | 4B-3 DWT + rand=0.5 | §6.3 PASS（0.7255/0.3297） | epoch_0003.pt | **删除** |
| `630_phase4c_dino_clean_lvl2/` (ep3) | 4C DINO 污染 clean + lvl2 | §7 NEGATIVE（0.7118/0.3038, clip -0.0125）；"Style Is Learned, Not Extracted" | epoch_0003.pt | **删除**（关键负面结论已记录） |
| `630_phase4d_lvl2/` (ep3) | 4D 2-Level DWT 突破 | §8.1 **BREAKTHROUGH**（0.7301/0.3402, +0.0040 clip） | epoch_0003.pt | **边界**（突破时刻，但被 4F.1 lvl3 SOTA 取代；建议删除，文档已留痕） |
| `630_phase4d_lvl2_dwt_rand50/` (ep3) | 4D lvl2 + dwt rand50 | §8.1 PASS（0.7294/0.3394） | epoch_0003.pt | **删除** |
| `630_phase4e_db2_lvl1/` (ep3) | 4E Daubechies db2 lvl1 | §8.2 FLAT（0.7258/0.3288）；基函数非关键 | epoch_0003.pt | **删除** |
| `630_phase4e_db2_lvl2/` (ep3) | 4E db2 lvl2 | §8.2 FLAT（0.7298/0.3398） | epoch_0003.pt | **删除** |
| `630_phase4f_lvl3/` (ep3) | 4F 3-Level DWT NEW SOTA | §8.3 **远程 SOTA**（clip=0.7319, lpips=0.3428）；3-Level 峰值 | epoch_0003.pt | **★保留★**（远程 SOTA，论文 Pareto 关键点） |
| `630_phase4f_lvl4/` (ep3) | 4F 4-Level | §8.3 FAIL（0.7316/0.3461, -0.0003） | epoch_0003.pt | **删除** |
| `630_phase4g1a_lock_ll/` (ep3) | 4G.1a lock=True + w_ll=1.0 | §9.1（0.7178）；LL velocity 应用 +0.0141，训练 -0.0091 lpips | epoch_0003.pt | **删除**（消融已记录） |
| `630_phase4g2_per_subband/` (ep3) | 4G.2 per-subband AdaIN α=1.0 | §9.2 MIXED（0.7361/0.3843, clip NEW SOTA 但 lpips FAIL） | epoch_0003.pt | **删除**（α 失效被 EOTA 推翻） |
| `630_phase4h2h_sota_w_hf_15/` (ep3) | 4H.2h w_hf=1.5 | §10.3 无效（0.7250/0.3330） | epoch_0003.pt | **删除** |
| `630_phase4h4e_sota_depth6/` (ep3) | 4H.4e depth=6 | §10.3 同向权衡（0.7265/0.3366） | epoch_0003.pt | **删除** |
| `630_phase4h4f_sota_dim96/` (ep3) | 4H.4f dim=96 | §10.3 同向权衡（0.7271/0.3368） | epoch_0003.pt | **删除** |
| `630_phase4h4g_sota_dim96_5ep/` (ep5) | 4H.4g dim=96 + 5ep | 未单独记录；4H.4 系列 5ep 延续 | epoch_0005.pt | **删除** |
| `630_phase4h5e_sota_mask25/` (ep3) | 4H.5e mask=0.25 | §10.3 同向权衡（0.7227/0.3172） | epoch_0003.pt | **删除** |
| `630_phase4h5f_sota_mask75/` (ep3) | 4H.5f mask=0.75 | §10.3 同向权衡（0.7237/0.3272） | epoch_0003.pt | **删除** |
| `630_phase4i10b_ept_t01/` (ep5) | 4I.10 Probe 诊断 | §11.8 诊断性（5 大瓶颈） | epoch_0005.pt | **删除**（诊断用，无性能意义） |
| `630_phase4i2a_sota_heun/` (ep3) | 4I.2a SOTA + Heun 3ep | §11.2（0.7260/0.3279, +0.0009 clip） | epoch_0003.pt | **删除**（被 4I.2b 5ep 取代） |
| `630_phase4i2b_sota_heun_5ep/` (ep5) | 4I.2b Heun + 5ep | §11.2 **NEW SOTA 双提升**（0.7266/0.3229） | epoch_0005.pt | **★保留★**（Heun 结构性突破 SOTA，论文 DOF 论据） |
| `630_phase4i6a_sota_rk4_5ep/` (ep5) | 4I.6 RK4 + 5ep | §11.4 饱和（0.7265/0.3235）；Heun→RK4 无收益 | epoch_0005.pt | **删除**（饱和结论已记录） |
| `630_phase4i9_wct_a085_5ep/` (ep5) | 4I.9 WCT α=0.85 | §11.7 STYLE GAIN CONTENT LOSS（0.7319/0.3568） | epoch_0005.pt | **删除** |
| `630_phase4j1_dwt_route/` (ep5) | 4J.1 DWT route cross-attn 起点 | §12.1（0.7226/0.3068）；本地 DWT route 起点 | epoch_0005.pt | **边界→保留**（本地 lineage 起点，T11 的直接前驱；建议保留） |
| `630_phase4j2_wct_aligned/` (ep5) | 4J.2 WCT aligned target | §12.2 未记为成功 | epoch_0005.pt | **删除** |
| `630_phase4j6_fewshot_popart/` (ep5) | 4J.6 v1 few-shot textual inversion | §12.3 FAIL（v1 0.6984）；梯度通路太弱 | epoch_0005.pt | **删除** |

**Phase 4 小结**：仅 `630_phase4f_lvl3/epoch_0003.pt`（4F.1 远程 SOTA）和 `630_phase4i2b_sota_heun_5ep/epoch_0005.pt`（4I.2b Heun SOTA）确认为"保留"。4J.1 起点建议保留。其余全部删除。

---

## 4. ckpt 意义汇总

### 4.1 ★建议保留★（有论文/对照价值）

| 路径 | 角色 | 关键指标 |
|---|---|---|
| `phase4/630_phase4f_lvl3/epoch_0003.pt` | **4F.1 远程 SOTA**（无 DWT route） | clip=0.7319, lpips=0.3428 |
| `phase4/630_phase4i2b_sota_heun_5ep/epoch_0005.pt` | **4I.2b 远程 SOTA**（Euler→Heun 结构性突破） | clip=0.7266, lpips=0.3229 |
| `phase4/630_phase4j1_dwt_route/epoch_0005.pt` | 4J.1 本地 DWT route 起点（T11 前驱） | clip=0.7226, lpips=0.3068 |
| `local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt` | **T11 本地 SOTA**（FC-SB 主报告结果） | clip=0.7213, lpips=0.2868 |
| `phase4/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt` | **4I.7b 远程最终 SOTA**（cosine+Heun+α=0.85） | clip=0.7272, lpips=0.3218 |

> 上表第 4、5 项**不在用户给定的列表中**，但盘盘确实存在且为最高价值 SOTA ckpt。详见 §5。

### 4.2 建议删除清单（无意义 ckpt）

**Early（全部删除）**：
- `clean_base_v2_local/epoch_0005.pt, epoch_0010.pt`
- `clean_base_v2_relu2/epoch_0003.pt`
- `task1_endpoint_film_baseline/epoch_0001.pt`、`task1_endpoint_film_no_norm/epoch_0001.pt`
- `task3_baseline_1ep/epoch_0001.pt`、`task3_combo_a_1ep/epoch_0001.pt`、`task3_combo_b_3ep/epoch_{0001-0003}.pt`
- `task4_iter/{r1a_latent_baseline,r1a_no_dino_minimal,r1b_no_dino_nogn,r2a_with_film,r2b_with_antiwhiten,r3c_optimal_5ep,r4d1_velmag_high,r6_pixel_color_fix,r7_saturation_loss,p2_long_10ep}/epoch_*.pt`（含嵌套 `src/exp/...` 污染副本，建议整体清理）
- `task4_style_strength_{w05_2ep,w10_2ep}/epoch_{0001,0002}.pt`

**Local T（除 T11 外全部删除）**：
- `630_local_{r1_depth2,r2_dim32,r3_gate_init0,t2_soft_ll_t2a,t5_eval_only_dwt,t14_casi,t15_llgqca,t18a_wll05,t18b_wll10,t19a_depth6,t19b_dim96,t21_adaln_zero_ll,t22_tone_bias,t23_ll_mean_only,t24_ll_std_only,t25_ll_cov_only,t26_ll_ycbcr}/epoch_0005.pt`

**Phase 4（除 §4.1 保留项外全部删除）**：
- `630_phase1d_verify/epoch_0002.pt`、`630_phase1d_verify_v2/epoch_0003.pt`
- `630_phase2{b_mask_random_50,c_mask_random_75,c_mask_shuffle_50,c_mask_shuffle_75}/epoch_0003.pt`
- `630_phase4a2_{adain_0,extrap_0,w_ll_0}/epoch_0003.pt`
- `630_phase4b1_{freq_a05,freq_a1,freq_a1_rand50}/epoch_0003.pt`
- `630_phase4b2_freq_a1_{rand30,rand70}/epoch_0003.pt`
- `630_phase4b3_{dwt_a1,dwt_a1_rand50}/epoch_0003.pt`
- `630_phase4c_dino_clean_lvl2/epoch_0003.pt`
- `630_phase4d_{lvl2,lvl2_dwt_rand50}/epoch_0003.pt`
- `630_phase4e_db2_{lvl1,lvl2}/epoch_0003.pt`
- `630_phase4f_lvl4/epoch_0003.pt`
- `630_phase4g1a_lock_ll/epoch_0003.pt`、`630_phase4g2_per_subband/epoch_0003.pt`
- `630_phase4h2h_sota_w_hf_15/epoch_0003.pt`、`630_phase4h4e_sota_depth6/epoch_0003.pt`、`630_phase4h4f_sota_dim96/epoch_0003.pt`、`630_phase4h4g_sota_dim96_5ep/epoch_0005.pt`、`630_phase4h5e_sota_mask25/epoch_0003.pt`、`630_phase4h5f_sota_mask75/epoch_0003.pt`
- `630_phase4i10b_ept_t01/epoch_0005.pt`、`630_phase4i2a_sota_heun/epoch_0003.pt`、`630_phase4i6a_sota_rk4_5ep/epoch_0005.pt`、`630_phase4i9_wct_a085_5ep/epoch_0005.pt`
- `630_phase4j2_wct_aligned/epoch_0005.pt`、`630_phase4j6_fewshot_popart/epoch_0005.pt`

> 删除前流程建议：对每个待删 ckpt 所在目录，先 `git add` 整个目录（含 config.json + src/ + logs/ + full_eval/）→ 在 commit message 中引用本审计报告 §1-3 对应行的 why/conclusion → 再 `git rm` ckpt 文件。这样"详细计入文档后删除"的要求由本报告 + git history 共同满足。

---

## 5. 关键发现：盘盘多出的 SOTA 目录（不在用户列表中）

实际 `ls` 发现，以下高价值目录**未出现在任务给定列表**，但盘盘确实存在且包含 SOTA ckpt。删除任何 ckpt 前必须先核对此清单：

| 目录 | ckpt | 角色 | 处置 |
|---|---|---|---|
| `local_t/630_local_t11_stochastic_dwt_p08/` | epoch_0005.pt | **T11 本地 SOTA**（§13.4, FC-SB 主报告结果 clip=0.7213/lpips=0.2868） | **必须保留** |
| `phase4/630_phase4i7b_cosine_heun_a085_5ep/` | epoch_0005.pt | **4I.7b 远程最终 SOTA**（§11.5, clip=0.7272/lpips=0.3218） | **必须保留** |
| `local_t/630_local_t10_stochastic_dwt/` | epoch_0005.pt | T10 p=0.5（§13.3, lpips BEST 0.2480） | 建议保留作 lpips 极值对照 |
| `local_t/630_local_t13_ll_global_style_inject/` | epoch_0005.pt | T13 LLGSI（§14.1） | 可删 |
| `local_t/630_local_t16{a,b,c}_llgqca_gate{02,03,05}/` | epoch_0005.pt×3 | T16 gate sweep（§14.4 全 FAIL） | 可删 |
| `local_t/630_local_t20_structure_aligned_target/` | epoch_0005.pt | T20 | 未记录，建议补录后删 |
| `phase4/630_phase3_mask_random_50_10ep/` | ep5,10 | Phase 3 完整训练验证（§4, 0.7289/0.3370） | 可删（结论已记录） |
| `phase4/630_phase4c_blockmask_r60_b128_lvl2/` | ep3 | 4C.1 blockmask（§7, 0.7151/0.3177, FAIL） | 可删 |
| `phase4/630_phase4g1b_lock_ll_zero_wll/` | ep3 | 4G.1b（§9.1, 0.7174） | 可删 |
| `phase4/630_phase4g2b_per_subband_a05/` | ep3 | 4G.2b（§9.2, α=0.5≡α=1.0, FAIL） | 可删 |
| `phase4/630_phase4h1{a,b,c,d,e,f,g} + h1g5ep/` | ep3/5 | 4H.1 α sweep + spatial_fiber（§10.1-10.2, 4H.1g NEW SOTA） | **4H.1g 建议保留**（4H.1g=0.7251/0.3281 Pareto 更优），其余可删 |
| `phase4/630_phase4h2i_per_subband_a07_w_ll_05/` | ep3 | 4H.2i | 可删 |
| `phase4/630_phase4h3f_sota_patch_1359_15/` | ep3 | 4H.3f patch+15（§10.3 无效） | 可删 |
| `phase4/630_phase4h7d_sota_terminal_swd_03/` | ep3 | 4H.7d terminal_swd=0.3（§10.3 完全无影响） | 可删 |
| `phase4/630_phase4i1{a,d}_eota_per_subband_*/` | ep3 | 4I.1 多尺度 α（§11.1 FAIL） | 可删 |
| `phase4/630_phase4i5{a,b,c}_sota_heun_*/` | ep3/5 | 4I.5 schedule sweep（§11.3） | 4I.5b 建议保留（内容冠军 0.7262/0.3171），其余可删 |
| `phase4/630_phase4i7a_cosine_heun_a09_5ep/` | ep5 | 4I.7a α=0.9（§11.5） | 可删 |
| `phase4/630_phase4i8{a,b}_*/` | ep5/8 | 4I.8 饱和确认（§11.6） | 可删 |
| `phase4/630_phase4j3_fewshot_stylemem/` | ep5 | 4J.3 | 未记录 |
| `phase4/630_phase4j4_progressive_alpha/` | ep5 | 4J.4 | 未记录 |
| `phase4/630_phase4j5_wct_aligned_progressive/` | ep5 | 4J.5 | 未记录 |
| `phase4/630_phase4j6_fewshot_popart_{v2,v3}/` | ep5/10/15 | 4J.6 v2/v3（§12.3 FAIL） | 可删 |
| `early/phase3_task2_p3{d,e}_contrastive_*/` | ep1-3 | P3-D/E contrastive loss | wikiarts_5 smoke，可删 |
| `early/task4_iter/r1c_no_dino_fixedone/` `r3a_aggressive_antwhiten/` `r3b_endpoint_antiwhiten/` `r4c_velocity_magnitude/` `task4_style_strength_baseline_2ep/` | 各 ep | 早期迭代 | wikiarts_5，可删 |

> **删除操作强约束**：用户列表外的目录，特别是 §5 表中标注"必须保留 / 建议保留"的（T11、4I.7b、4H.1g、4I.5b、T10），**绝对不能删除**。这些是论文 SOTA/Pareto 关键点。

---

## 6. 数据集分类汇总（按实验）

- **distinct5 主线（512, vlen=1.0）**：`clean_base_v2_*` + 全部 `phase4/*` + 全部 `local_t/*`（含 T11）。这是所有有性能意义实验的集合。
- **wikiarts_5 smoke（512, vlen=0.04）**：`task1_endpoint_film_*`、`task3_*`、`task4_style_strength_*`。仅 smoke，无性能意义。
- **wikiarts_5 full（512, vlen=1.0）**：`task4_iter/r1~r7 + p2_long_10ep`。早期迭代，已被 DWT 路线取代。
- **fewshot6（512, vlen=1.0）**：`630_phase4j6_fewshot_popart*`。Few-shot Pop_Art 注入专用。

**结论**：除 `628_ablation/`（仅 JSON）外，所有 wikiarts_5 / fewshot6 实验 ckpt 均建议删除；distinct5 主线仅保留 §4.1 + §5 中标注"保留"的 SOTA/Pareto ckpt。

---

## 7. 审计方法学说明

1. **why 判定**：读取 `config.json::ablation.notes`（每个实验设计时都写明了假设与对照基线）。
2. **conclusion 判定**：以 `docs/72/03_experiments.md` 的 §2-§15 结论表为主；该文档未覆盖的（R1-R3、T20-T26、部分 4J 子项）标注"未记录"。
3. **ckpt 意义判定标准**：
   - ★保留★ = SOTA checkpoint / Pareto 关键点 / 论文引用价值（4F.1, 4I.2b, 4I.7b, T11, 4J.1 起点）
   - 边界 = 突破时刻但被后续取代（4D lvl2）、本地 lineage 关键节点（4J.1）
   - 删除 = 失败实验 / smoke test / smoke 重复 / 被后续 SOTA 取代 / 消融结论已文档化
4. **数据集判定**：`data.data_root` + `data.virtual_length_multiplier` 双字段交叉确认。
5. **未覆盖项**：T20-T26、R1-R3、4J.3-4J.5 在 03_experiments.md 中无结论记录，建议补录 §14/§15 后再删对应 ckpt。

> 本报告仅给出删除建议清单，**未执行任何删除**。实际删除需按 §4.2 末尾的 git 流程逐项 commit，并在 commit message 中引用本报告对应行。
