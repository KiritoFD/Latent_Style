# 本地实验目录清单 (local_exp_inventory.md)

> 生成时间：2026-07-02
> 扫描根目录：`g:\GitHub\Latent_Style\SchrodingerBridge\exp\`
> 方法：PowerShell 递归计算目录大小 + config.json 关键字段解析
> 说明：未进入 src/ 内部读取代码；.pt 文件仅确认存在性；config.json 仅读关键字段

---

## 统计摘要

| 指标 | 数值 |
|------|------|
| 实验子目录总数 | **202** |
| 顶层散落文件数 | 32 (4.6 MB) |
| 总占用空间 | **26,392.7 MB (~25.8 GB)** |

### 各类别数量与空间分布

| 类别 | 目录数 | 占用空间 (MB) | 说明 |
|------|--------|---------------|------|
| ① 重要保留 (keep) | 120 | 26,255.0 | 有 epoch_*.pt 或 full_eval/eval 结果或关键基线资源 |
| ② 仅src无产出 (src_only) | 9 | 1.5 | 只有 src/+config.json，无 ckpt 无 eval |
| ③ 临时脚本集 (temp_scripts) | 4 | 11.0 | 全是 _check_*.sh / debug*.sh 临时脚本 |
| ④ 历史归档 (archive) | 42 | 114.1 | 2026-05 及之前的 probe/诊断 + 旧系列废弃目录 |
| ⑤ smoke/probe 可清理 (smoke_probe) | 27 | 11.1 | _smoke_* / local_wsl_* / probe 实验目录 |
| **合计** | **202** | **26,392.7** | |

---

## 一、重要保留实验 (keep) — 120 个目录

### 1.1 `630_local_*` 系列（当前主线 T2-T26 + R1/R2/R3）— 27 个目录

> 全部含 `config.json` + `src/` + `epoch_0005.pt`（除 eval-only 目录），最后修改 2026-07-01~02。
> config 关键字段映射：dim=`model.base_dim`，depth=`model.num_res_blocks`，alpha=`model.style_extrap_alpha`，w_ll=`bridge.spectral_w_ll`，gate_init=`model.style_cross_attn_gate_init`

| 目录名 | 大小(MB) | 修改日期 | ckpt | full_eval | dim | depth | alpha | w_ll | gate_init | epochs | 实验说明 |
|--------|----------|----------|------|-----------|-----|-------|-------|------|-----------|--------|----------|
| 630_local_r1_depth2 | 13.5 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 2 | 0.4 | 0.3 | 0.05 | 5 | R1 反向实验 depth=2 |
| 630_local_r2_dim32 | 12.2 | 2026-07-02 | epoch_0005.pt | ✓ | 32 | 4 | 0.4 | 0.3 | 0.05 | 5 | R2 反向实验 dim=32 |
| 630_local_r3_gate_init0 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.0 | 5 | R3 反向实验 gate_init=0.0 |
| 630_local_t2_soft_ll_t2a | 15.4 | 2026-07-01 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T2 Soft LL Route α=0.05 |
| 630_local_t5_eval_only_dwt | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T5 Eval-Only DWT Route |
| 630_local_t10_stochastic_dwt | 15.5 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T10 随机DWT p=0.5，有train.log |
| 630_local_t11_stochastic_dwt_p08 | 15.5 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T11 随机DWT p=0.8（SOTA基底），有train.log |
| 630_local_t13_ll_global_style_inject | 15.5 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T13 LLGSI 全局统计注入，有train.log |
| 630_local_t14_casi | 15.5 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T14 CASI 交叉注意力统计注入，有train.log |
| 630_local_t15_llgqca | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T15 LLGQCA LL全局query交叉注意力 |
| 630_local_t16a_llgqca_gate02 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T16a LLGQCA gate=0.2 扫描 |
| 630_local_t16b_llgqca_gate03 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T16b LLGQCA gate=0.3 |
| 630_local_t16c_llgqca_gate05 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T16c LLGQCA gate=0.5 |
| 630_local_t18a_wll05 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.5 | 0.05 | 5 | T18a 恢复LL训练 w_ll=0.5 |
| 630_local_t18b_wll10 | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 1.0 | 0.05 | 5 | T18b 完全恢复LL训练 w_ll=1.0 |
| 630_local_t19a_depth6 | 14.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 6 | 0.4 | 0.3 | 0.05 | 5 | T19a 加深 depth=6 |
| 630_local_t19b_dim96 | 20.4 | 2026-07-02 | epoch_0005.pt | ✓ | 96 | 4 | 0.4 | 0.3 | 0.05 | 5 | T19b 加宽 dim=96 |
| 630_local_t20_structure_aligned_target | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T20 PlanB 结构对齐Flow Matching |
| 630_local_t21_adaln_zero_ll | 15.8 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T21 PlanC AdaLN-Zero on LL |
| 630_local_t22_tone_bias | 15.8 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T22 PlanD 直接Tone Bias注入 |
| 630_local_t23_ll_mean_only | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T23 PlanE 仅迁移LL mean |
| 630_local_t24_ll_std_only | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T24 PlanF 仅迁移LL std |
| 630_local_t25_ll_cov_only | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T25 PlanG 仅迁移LL协方差 |
| 630_local_t26_ll_ycbcr | 15.4 | 2026-07-02 | epoch_0005.pt | ✓ | 64 | 4 | 0.4 | 0.3 | 0.05 | 5 | T26 PlanH YCbCr色彩解耦 |
| 630_local_t3_eval_ll005 | 84.5 | 2026-07-01 | — | ✓ | — | — | — | — | — | — | T3 eval-only 结果目录(无config) |
| 630_local_t12_eval | 421.3 | 2026-07-02 | — | — | — | — | — | — | — | — | T12 批量eval结果(t12a-t12e子目录) |
| 630_local_t4_eval | 424.0 | 2026-07-02 | — | — | — | — | — | — | — | — | T4 批量eval结果(t3b-t4c子目录) |

### 1.2 `630_phase4*` 系列（Phase4 消融实验）— 52 个目录

> 全部含 `config.json` + `src/` + checkpoint + `full_eval/`，修改日期 2026-07-01。
> 默认参数：dim=64, depth=4, gate_init=0.05, style_gate_mode=tanh_gate, body_block_type=global_attn

| 目录名 | 大小(MB) | ckpt文件 | alpha | w_ll | dim | depth | epochs | 阶段说明 |
|--------|----------|----------|-------|------|-----|-------|--------|----------|
| 630_phase4a2_adain_0 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4A-2: endpoint_adain=0 禁用推理AdaIN |
| 630_phase4a2_extrap_0 | 15.4 | epoch_0003.pt | 0.0 | 0.3 | 64 | 4 | 3 | 4A-2: style_extrap_alpha=0 禁用外推 |
| 630_phase4a2_w_ll_0 | 94.4 | epoch_0003.pt | 0.1 | 0.0 | 64 | 4 | 3 | 4A-2: w_ll=0 禁用低频损失 |
| 630_phase4b1_freq_a05 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-1: 频域mask α=0.5 |
| 630_phase4b1_freq_a1 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-1: 频域mask α=1.0 |
| 630_phase4b1_freq_a1_rand50 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-1: 频域mask+随机dropout50% |
| 630_phase4b2_freq_a1_rand30 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-2: 频域+随机30% |
| 630_phase4b2_freq_a1_rand50_10ep | 30.1 | epoch_0005/0010.pt | 0.1 | 0.3 | 64 | 4 | 10 | 4B-2: 最佳配置10ep长训练 |
| 630_phase4b2_freq_a1_rand70 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-2: 频域+随机70% |
| 630_phase4b3_dwt_a1 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-3: Haar DWT频域mask |
| 630_phase4b3_dwt_a1_rand50 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4B-3: DWT+随机50% |
| 630_phase4c_blockmask_r60_b128_lvl2 | 89.1 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4C: RGB块mask r=0.6 b=128 |
| 630_phase4c_dino_clean_lvl2 | 109.2 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4C对照: 真实DINO+2级DWT |
| 630_phase4d_lvl2 | 98.8 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4D: 2级Haar DWT低通(SOTA clip=0.7301) |
| 630_phase4d_lvl2_dwt_rand50 | 99.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4D组合: 2级DWT+频域+随机50% |
| 630_phase4e_db2_lvl1 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4E.1: Daubechies-2单级 |
| 630_phase4e_db2_lvl2 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4E.2: db2+2级级联 |
| 630_phase4f_lvl3 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4F.1: 3级Haar DWT |
| 630_phase4f_lvl4 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4F.2: 4级Haar DWT |
| 630_phase4g1a_lock_ll | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4G.1a: 推理LL锁死 |
| 630_phase4g1b_lock_ll_zero_wll | 15.4 | epoch_0003.pt | 0.1 | 0.0 | 64 | 4 | 3 | 4G.1b: LL锁死+w_ll=0 |
| 630_phase4g2_per_subband | 99.9 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4G.2: 频域per-subband AdaIN(clip SOTA 0.7361) |
| 630_phase4g2b_per_subband_a05 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4G.2b: per-subband α=0.5 |
| 630_phase4h1a_eota_per_subband | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1a: EOTA+per-subband α=1.0 |
| 630_phase4h1b_eota_per_subband_a05 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1b: EOTA+per-subband α=0.5(关键对照) |
| 630_phase4h1c_eota_per_subband_a07 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1c: EOTA+per-subband α=0.7 |
| 630_phase4h1d_eota_per_subband_a08 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1d: EOTA+per-subband α=0.8 |
| 630_phase4h1e_eota_spatial_fiber_a05 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1e: EOTA+spatial_fiber α=0.5 |
| 630_phase4h1f_eota_spatial_fiber_a07 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1f: EOTA+spatial_fiber α=0.7(关键) |
| 630_phase4h1g_eota_spatial_fiber_a08 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.1g: EOTA+spatial_fiber α=0.8 |
| 630_phase4h1g5ep_eota_spatial_fiber_a08 | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4H.1g 5ep(新SOTA) |
| 630_phase4h2h_sota_w_hf_15 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.2h: SOTA+w_lh=w_hl=1.5 |
| 630_phase4h2i_per_subband_a07_w_ll_05 | 15.4 | epoch_0003.pt | 0.1 | 0.5 | 64 | 4 | 3 | 4H.2i: per-subband α=0.7+w_ll=0.5 |
| 630_phase4h3f_sota_patch_1359_15 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.3f: SOTA+swd_patch增加15 |
| 630_phase4h4e_sota_depth6 | 17.2 | epoch_0003.pt | 0.1 | 0.3 | 64 | 6 | 3 | 4H.4e: SOTA+depth=6 |
| 630_phase4h4f_sota_dim96 | 20.4 | epoch_0003.pt | 0.1 | 0.3 | 96 | 4 | 3 | 4H.4f: SOTA+dim=96 |
| 630_phase4h4g_sota_dim96_5ep | 20.4 | epoch_0005.pt | 0.1 | 0.3 | 96 | 4 | 5 | 4H.4g: dim=96+5ep |
| 630_phase4h5e_sota_mask25 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.5e: SOTA+mask=0.25 |
| 630_phase4h5f_sota_mask75 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.5f: SOTA+mask=0.75 |
| 630_phase4h7d_sota_terminal_swd_03 | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4H.7d: SOTA+terminal_swd=0.3 |
| 630_phase4i10b_ept_t01 | 112.0 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.10b: EPT端点预测训练t=[0,0.1] |
| 630_phase4i1a_eota_per_subband_multi_alpha | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4I.1a: 多尺度alpha(LH=HL=0.5,HH=0.9) |
| 630_phase4i1d_eota_per_subband_hh_only | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4I.1d: HH=1.0对照 |
| 630_phase4i2a_sota_heun | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4I.2a: Heun求解器 |
| 630_phase4i2b_sota_heun_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.2b: Heun+5ep |
| 630_phase4i5a_sota_heun_cosine | 15.4 | epoch_0003.pt | 0.1 | 0.3 | 64 | 4 | 3 | 4I.5a: Heun+余弦时间调度 |
| 630_phase4i5b_sota_heun_cosine_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.5b: Heun+余弦+5ep |
| 630_phase4i5c_sota_heun_rquad_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.5c: Heun+rquad+5ep |
| 630_phase4i6a_sota_rk4_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.6a: RK4求解器+5ep |
| 630_phase4i7a_cosine_heun_a09_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.7a: 余弦+Heun+α=0.9+5ep |
| 630_phase4i7b_cosine_heun_a085_5ep | 1112.3 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.7b: 余弦+Heun+α=0.85+5ep(体积异常大) |
| 630_phase4i8a_cosine_heun_a085_8ep | 30.2 | epoch_0005/0008.pt | 0.1 | 0.3 | 64 | 4 | 8 | 4I.8a: α=0.85+8ep |
| 630_phase4i8b_warpcos_p08_a085_5ep | 15.4 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.8b: warp_cos(p=0.8)+α=0.85 |
| 630_phase4i9_wct_a085_5ep | 94.9 | epoch_0005.pt | 0.1 | 0.3 | 64 | 4 | 5 | 4I.9: WCT替代AdaIN |
| 630_phase4j1_dwt_route | 15.5 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.1: DWT路由交叉注意力(方案B)，有run.log |
| 630_phase4j2_wct_aligned | 15.4 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.2: WCT对齐目标(方案A) |
| 630_phase4j3_fewshot_stylemem | 12.6 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.3: Few-shot style_mem优化 |
| 630_phase4j4_progressive_alpha | 15.4 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.4: 渐进Alpha调度(方案C) |
| 630_phase4j5_wct_aligned_progressive | 98.8 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.5: 方案A+C综合(最优候选) |
| 630_phase4j6_fewshot_popart | 14.8 | epoch_0005.pt | 0.4 | 0.3 | 64 | 4 | 5 | 4J.6: Few-shot Pop_Art学习 |
| 630_phase4j6_fewshot_popart_v2 | 43.1 | epoch_0005/0010/0015.pt | 0.4 | 0.3 | 64 | 4 | 15 | 4J.6 v2: 高LR+15ep |
| 630_phase4j6_fewshot_popart_v3 | 43.0 | epoch_0005/0010/0015.pt | 0.4 | 0.3 | 64 | 4 | 15 | 4J.6 v3: 高LR+15ep |

### 1.3 `630_phase1d/2b/2c/3` 系列 — 7 个目录

| 目录名 | 大小(MB) | 修改日期 | ckpt文件 | full_eval | 说明 |
|--------|----------|----------|----------|-----------|------|
| 630_phase1d_verify | 94.8 | 2026-06-30 | epoch_0002.pt | ✓ | Phase1d 验证 |
| 630_phase1d_verify_v2 | 94.4 | 2026-06-30 | epoch_0003.pt | ✓ | Phase1d 验证v2 |
| 630_phase2b_mask_random_50 | 95.1 | 2026-06-30 | epoch_0003.pt | ✓ | Phase2b 随机mask50% |
| 630_phase2c_mask_random_75 | 96.4 | 2026-06-30 | epoch_0003.pt | ✓ | Phase2c 随机mask75% |
| 630_phase2c_mask_shuffle_50 | 94.6 | 2026-06-30 | epoch_0003.pt | ✓ | Phase2c shuffle mask50% |
| 630_phase2c_mask_shuffle_75 | 95.4 | 2026-06-30 | epoch_0003.pt | ✓ | Phase2c shuffle mask75% |
| 630_phase3_mask_random_50_10ep | 114.5 | 2026-06-30 | epoch_0005/0010.pt | ✓ | Phase3 10ep长训练 |

### 1.4 `task1/3/4` 系列（早期 phase3 探索）— 9 个目录

| 目录名 | 大小(MB) | 修改日期 | ckpt文件 | full_eval | dim | depth | gate_mode | epochs | 说明 |
|--------|----------|----------|----------|-----------|-----|-------|-----------|--------|------|
| task1_endpoint_film_baseline | 15.8 | 2026-06-23 | epoch_0001.pt | ✓ | 64 | 4 | — | 1 | Task1 FiLM基线(含GroupNorm) |
| task1_endpoint_film_no_norm | 15.8 | 2026-06-23 | epoch_0001.pt | ✓ | 64 | 4 | — | 1 | Task1 FiLM无GroupNorm |
| task3_baseline_1ep | 17.4 | 2026-06-23 | epoch_0001.pt | ✓ | 64 | 4 | tanh_gate | 1 | Task3 基线 |
| task3_combo_a_1ep | 17.4 | 2026-06-23 | epoch_0001.pt | ✓ | 64 | 4 | fixed_one | 1 | Task3 组合A |
| task3_combo_b_3ep | 50.1 | 2026-06-23 | epoch_0001/0002/0003.pt | ✓ | 64 | 4 | fixed_one | 3 | Task3 组合B 3ep |
| task4_style_strength_baseline_2ep | 33.8 | 2026-06-23 | epoch_0001/0002.pt | ✓ | 64 | 4 | fixed_one | 2 | Task4 基线 |
| task4_style_strength_w05_2ep | 33.8 | 2026-06-23 | epoch_0001/0002.pt | ✓ | 64 | 4 | fixed_one | 2 | Task4 w=0.5 |
| task4_style_strength_w10_2ep | 33.8 | 2026-06-23 | epoch_0001/0002.pt | ✓ | 64 | 4 | fixed_one | 2 | Task4 w=1.0 |
| task4_iter | 2686.0 | 2026-06-24 | — | — | — | — | — | — | Task4迭代实验集(r1a-r7共15子实验) |

### 1.5 `phase3_task2` / `clean_base` 系列 — 4 个目录

| 目录名 | 大小(MB) | 修改日期 | ckpt文件 | full_eval | 说明 |
|--------|----------|----------|----------|-----------|------|
| phase3_task2_p3d_contrastive_w01_margin01 | 45.4 | 2026-06-24 | epoch_0001/0002/0003.pt | ✓ | P3-D 对比损失 w=0.1 margin=0.1 |
| phase3_task2_p3e_contrastive_w05_margin005 | 45.4 | 2026-06-24 | epoch_0001/0002/0003.pt | ✓ | P3-E 对比损失 w=0.5 margin=0.05 |
| clean_base_v2_local | 105.8 | 2026-06-30 | epoch_0005/0010.pt | ✓ | P4 T5 干净基线(dim=64,depth=4,α=0.1,w_ll=0.3,10ep) |
| clean_base_v2_relu2 | 94.9 | 2026-06-30 | epoch_0003.pt | ✓ | M9修复: relu2注意力(3ep) |

### 1.6 基线 / 资源 / 评估目录 — 12 个目录

| 目录名 | 大小(MB) | 修改日期 | 说明 |
|--------|----------|----------|------|
| baseline_reeval | 5372.4 | 2026-06-30 | 基线重评估结果(adain/identity/samam/sdedit等13子目录) |
| baseline_images | 4132.0 | 2026-06-30 | 基线图像库(adain/samam/sdedit/styleid等21子目录) |
| baseline_v2 | 1030.1 | 2026-07-01 | 基线v2(含eval/和images/子目录) |
| eval_cache | 874.2 | 2026-07-01 | 评估缓存(CLIP特征/ref_feats/src_feats，full_eval依赖) |
| phase616_live_dashboard | 1220.8 | 2026-06-20 | 616系列实时仪表盘+eval归档(14个.tgz) |
| 620_spatial_bridge | 5183.9 | 2026-06-23 | 620系列65个smoke/ablation子实验(旧系列，可考虑归档) |
| 628_ablation | 6.7 | 2026-06-29 | 628系列消融(p8b_steps_fine) |
| adain_checkpoints | 100.9 | 2026-06-30 | AdaIN预训练权重(vgg19/decoder_v32k/decoder_vgg19) |
| clean_base | 0.0 | 2026-06-29 | 空目录(有full_eval/但无内容) |
| 630_planA_zero_step_wct | 250.5 | 2026-07-02 | PlanA零步WCT评估结果(3个子目录+results.json) |
| _smoke_distinct5_512_ema_baseline_vlen004 | 0.5 | 2026-06-05 | smoke基线(有full_eval) |
| 630_local_t3_eval_ll005 | 84.5 | 2026-07-01 | T3 eval结果(无config，有full_eval) |

---

## 二、仅src无产出 (src_only) — 9 个目录

> 只有 src/ + config.json，无 ckpt 无 eval，可归档/删除。

| 目录名 | 大小(MB) | 修改日期 | config | src | 说明 |
|--------|----------|----------|--------|-----|------|
| 630_local_t11_long30ep | 0.6 | 2026-07-02 | ✓ | ✓ | T11 30ep长训练(无产出，dim=64,depth=4,30ep) |
| 630_local_t3_adain_ll_t3a | 0.6 | 2026-07-01 | ✓ | ✓ | T3 AdaIN LL=0.05(无产出) |
| task5_baseline_2ep | 0.0 | 2026-06-23 | ✓ | ✓ | Task5 基线(有train.log无ckpt) |
| task5_endpoint_a_2ep | 0.0 | 2026-06-23 | ✓ | ✓ | Task5 Endpoint A(有train.log无ckpt) |
| task5_endpoint_b_2ep | 0.0 | 2026-06-23 | ✓ | ✓ | Task5 Endpoint B(有train.log无ckpt) |
| task5_endpoint_c_2ep | 0.0 | 2026-06-23 | ✓ | ✓ | Task5 Endpoint C(有train.log无ckpt) |
| task6_baseline_5ep | 0.1 | 2026-06-23 | ✓ | ✗ | Task6 基线(有train.log无ckpt无src) |
| task6_exp_a_optimal_5ep | 0.1 | 2026-06-23 | ✓ | ✗ | Task6 Exp A(有train.log无ckpt无src) |
| task6_exp_b_two_stage_5ep | 0.1 | 2026-06-23 | ✓ | ✗ | Task6 Exp B两阶段(有train.log无ckpt无src) |

---

## 三、临时脚本集 (temp_scripts) — 4 个目录

> 全是临时诊断/检查脚本，可清理。

| 目录名 | 大小(MB) | 修改日期 | 文件数 | 说明 |
|--------|----------|----------|--------|------|
| 625_fc_sb | 1.3 | 2026-06-26 | ~150 | 全是 _check_*.sh / _launch_*.sh / _diag_*.sh 临时脚本 |
| p3_remote_10h | 8.5 | 2026-06-25 | ~130 | 全是 check_*.sh / debug*.sh / run_*.sh 远程调试脚本 |
| tuning_deepdive | 0.4 | 2026-06-25 | ~15 | 全是 sh 调优诊断脚本 |
| phase4j_batch_logs | 0.8 | 2026-07-01 | 5 | 4J批量评估日志(4J.2/4J.4/4J.5 + eval日志) |

---

## 四、历史归档 (archive) — 42 个目录

### 4.1 2026-05 及之前的 probe/诊断/校准实验 — 36 个目录

> 全部为 2026-05-27~30 的早期 probe/calibration 实验，无 ckpt 无 eval，总计仅 5.2 MB。

| 目录名 | 大小(MB) | 修改日期 | 类型 |
|--------|----------|----------|------|
| armored_breakthrough_proper | 0.0 | 2026-05-27 | 空 |
| decision_tree_clip_style | 0.0 | 2026-05-27 | 空 |
| diffeomorphic_tangent_head_sweep | 0.0 | 2026-05-27 | 空 |
| diffeomorphic_tangent_sweep | 0.3 | 2026-05-27 | sweep |
| fisher_operator_consumer_probe | 0.3 | 2026-05-28 | probe |
| fisher_operator_tokenizer_probe | 0.3 | 2026-05-28 | probe |
| fisher_style_backbone_probe | 0.1 | 2026-05-28 | probe |
| fisher_style_memory_adapter_probe | 0.0 | 2026-05-28 | 空 |
| local_repro_sadd_38f_8ep_20260528_224707 | 0.0 | 2026-05-28 | 空(有log) |
| manual_k1_k2_8epoch | 0.4 | 2026-05-27 | 手动实验 |
| phase1_diagnostic_probes | 0.1 | 2026-05-27 | 诊断 |
| physical_loss_tree | 0.1 | 2026-05-27 | 物理损失树 |
| reference_memory_generation_probe_full | 0.2 | 2026-05-28 | probe |
| remote_factorized_tokenizer_pull | 0.1 | 2026-05-27 | 远程拉取 |
| router_aware_backbone_probe | 0.1 | 2026-05-28 | probe |
| scripts | 0.1 | 2026-05-27 | 旧脚本 |
| style_memory_bank_adapter_probe | 0.1 | 2026-05-28 | probe |
| style_memory_bank_adapter_route_probe | 0.1 | 2026-05-28 | probe |
| style_memory_bank_probe | 0.1 | 2026-05-28 | probe |
| style_memory_typed_adapter_probe | 0.0 | 2026-05-28 | 空 |
| style_representation_adapter_probe | 0.1 | 2026-05-29 | probe |
| style_representation_safe_projection_probe | 0.2 | 2026-05-29 | probe |
| style_representation_style_aware_router_probe | 0.2 | 2026-05-29 | probe |
| t01_local_base | 0.0 | 2026-05-29 | 空 |
| temp_anneal_proper | 0.0 | 2026-05-27 | 空 |
| tokenizer_adain_gate_calibration | 0.7 | 2026-05-28 | 校准 |
| tokenizer_adain_texture_gate_calibration_rerun | 0.2 | 2026-05-28 | 校准 |
| tokenizer_bandgate_calibration | 0.1 | 2026-05-27 | 校准 |
| tokenizer_prototype_carrier_calibration | 0.1 | 2026-05-28 | 校准(有log) |
| tokenizer_stat_reader_probe | 0.3 | 2026-05-28 | probe(有log) |
| tokenizer_stat_vocab_probe | 0.1 | 2026-05-28 | probe(有log) |
| tokenizer_texton_carrier_calibration | 0.1 | 2026-05-28 | 校准(有log) |
| vae_backend_256_mse_controls | 0.0 | 2026-05-28 | 空 |
| vae_backend_256_status | 0.0 | 2026-05-27 | 空 |
| wikiart_512_encode_logs | 0.7 | 2026-05-30 | 编码日志 |
| wikiart_512_transfer_logs | 0.0 | 2026-05-30 | 传输日志(有log) |

### 4.2 旧系列废弃目录 — 6 个目录

| 目录名 | 大小(MB) | 修改日期 | 说明 |
|--------|----------|----------|------|
| phase3_task1 | 108.7 | 2026-06-24 | Phase3 Task1(p3a/p3b子目录+json配置) |
| fc_sb_r2 | 0.2 | 2026-06-25 | 旧fc_sb实验(g0/g5子目录+gen脚本) |
| 20250618_lite_ot_vertical_auto | 0.0 | 2026-06-18 | 空 |
| p4_fusion_breakout | 0.0 | 2026-06-29 | 空 |
| task4_no_dino | 0.0 | 2026-06-23 | 空(仅TASK4_REPORT.md) |
| wikiart_stress1_..._variant_f_b44_remote | 0.0 | 2026-06-06 | 空 |

---

## 五、smoke/probe 可清理 (smoke_probe) — 27 个目录

> _smoke_* / local_wsl_* / probe 实验目录，无 ckpt 无 eval（除1个有full_eval），总计 11.1 MB。

### 5.1 `_smoke_distinct5_*` 系列 — 12 个目录

| 目录名 | 大小(MB) | 修改日期 | config | src | 说明 |
|--------|----------|----------|--------|-----|------|
| _smoke_distinct5_512_ema_variant_a_class_prototypes_b8_vlen001 | 0.4 | 2026-06-05 | ✓ | ✓ | variant_a |
| _smoke_distinct5_512_ema_variant_b_global_vq_b8_vlen001 | 0.4 | 2026-06-05 | ✓ | ✓ | variant_b |
| _smoke_distinct5_512_ema_variant_c_content_guided_spatial_b8_vlen001 | 0.4 | 2026-06-05 | ✓ | ✓ | variant_c |
| _smoke_distinct5_512_ema_variant_d_vq_content_guided_b8_vlen001 | 0.4 | 2026-06-05 | ✓ | ✓ | variant_d |
| _smoke_distinct5_512_ema_variant_e_latent_prototype_ot_queue_b8_vlen001 | 0.4 | 2026-06-05 | ✓ | ✓ | variant_e |
| _smoke_distinct5_512_ema_variant_i_dual_mix_local | 0.4 | 2026-06-05 | ✓ | ✓ | variant_i |
| _smoke_distinct5_512_ema_variant_j_aux_hard_swd_local | 0.4 | 2026-06-05 | ✓ | ✓ | variant_j |
| _smoke_distinct5_512_ema_variant_k_content_adaptive_local | 0.4 | 2026-06-05 | ✓ | ✓ | variant_k |
| _smoke_distinct5_512_ema_variant_m_style_gated_local_windows | 0.5 | 2026-06-05 | ✓ | ✓ | variant_m |
| _smoke_distinct5_profile_probe_b8_vlen001 | 0.4 | 2026-06-02 | ✓ | ✓ | profile probe |
| _smoke_distinct5_variant_a_latent_init | 0.4 | 2026-06-05 | ✓ | ✓ | latent init a |
| _smoke_distinct5_variant_b_latent_init | 0.4 | 2026-06-05 | ✓ | ✓ | latent init b |

### 5.2 `local_wsl_*` 系列 — 10 个目录

| 目录名 | 大小(MB) | 修改日期 | config | src | 说明 |
|--------|----------|----------|--------|-----|------|
| local_wsl_distinct5_512_ema_k_b16_step2min | 0.5 | 2026-06-05 | ✓ | ✓ | b16 step2min |
| local_wsl_distinct5_512_ema_k_b16_step2min_v160 | 0.5 | 2026-06-05 | ✓ | ✓ | b16 step2min v160 |
| local_wsl_distinct5_512_ema_k_b16_stepcalib | 0.5 | 2026-06-05 | ✓ | ✓ | step校准 |
| local_wsl_distinct5_512_ema_k_b32_e8 | 0.5 | 2026-06-05 | ✓ | ✓ | b32 e8 |
| local_wsl_wikiart512_carrier_gate_from_hist_e3 | 0.4 | 2026-06-05 | ✓ | ✓ | carrier_gate e3 |
| local_wsl_wikiart512_execution_budget_from_hist_e1 | 0.4 | 2026-06-05 | ✓ | ✓ | execution_budget e1 |
| local_wsl_wikiart512_full_b32_e8 | 0.4 | 2026-06-01 | ✓ | ✓ | full b32 e8(有log) |
| local_wsl_wikiart512_style_injection_delta_div_from_hist_e3 | 0.4 | 2026-06-05 | ✓ | ✓ | delta_div e3 |
| local_wsl_wikiart512_style_injection_delta_div_w05_from_hist_e3 | 0.4 | 2026-06-05 | ✓ | ✓ | delta_div w05 e3 |
| local_wsl_wikiart512_style_injection_from_hist_e1 | 0.4 | 2026-06-05 | ✓ | ✓ | style_injection e1 |

### 5.3 其他 smoke/probe — 5 个目录

| 目录名 | 大小(MB) | 修改日期 | 说明 |
|--------|----------|----------|------|
| local_wsl_wikiart512_style_injection_from_hist_e3 | 0.4 | 2026-06-05 | style_injection e3 |
| probes_20260601 | 0.0 | 2026-06-01 | 空 |
| smoke_blockmask | 0.0 | 2026-07-01 | 空(仅original.png) |
| style_representation_residual_scale_sweep | 0.8 | 2026-06-03 | 残差尺度扫描 |
| tmp_output_appearance_resume_smoke | 0.6 | 2026-06-13 | 临时外观恢复smoke |

---

## 六、顶层散落文件 — 32 个文件 (4.6 MB)

> exp/ 根目录下的散落日志/脚本/zip，非实验子目录。

| 文件名 | 类型 | 说明 |
|--------|------|------|
| 630_local_t15_llgqca_train.log | 日志 | T15训练日志 |
| 630_local_t19b_dim96_train.log | 日志 | T19b训练日志 |
| 630_phase3_train.log | 日志 | Phase3训练日志 |
| 630_phase4a2_adain_0_train.log | 日志 | 4A2训练日志 |
| 630_phase4a2_extrap_0_train.log | 日志 | 4A2训练日志 |
| 630_phase4b1_freq_a05_train.log | 日志 | 4B1训练日志 |
| 630_phase4b1_freq_a1_train.log | 日志 | 4B1训练日志 |
| _eval_samam_unified.bat | 脚本 | SaMam统一评估 |
| _remote_scan_v2.py | 脚本 | 远程扫描 |
| ablation_log.md | 文档 | 消融日志 |
| analyze_task6_results.py | 脚本 | Task6结果分析 |
| archive_manifest.json | 清单 | 归档清单 |
| baseline_err.log / baseline_train.log | 日志 | 基线日志 |
| clean_base_v2_relu2_train.log | 日志 | clean_base训练日志 |
| gen_task6_configs.py / verify_task6_configs.py | 脚本 | Task6配置 |
| t5_err.log / t5_train.log | 日志 | T5日志 |
| tuning_deepdive.zip | 压缩包 | 调优诊断打包 |

---

## 建议清理清单

### A. 可立即删除（安全，无产出）— 82 个目录，约 137.7 MB

| 清理类别 | 目录数 | 释放空间 | 目录列表 |
|----------|--------|----------|----------|
| ② 仅src无产出 | 9 | 1.5 MB | 630_local_t11_long30ep, 630_local_t3_adain_ll_t3a, task5_baseline_2ep, task5_endpoint_a_2ep, task5_endpoint_b_2ep, task5_endpoint_c_2ep, task6_baseline_5ep, task6_exp_a_optimal_5ep, task6_exp_b_two_stage_5ep |
| ③ 临时脚本集 | 4 | 11.0 MB | 625_fc_sb, p3_remote_10h, tuning_deepdive, phase4j_batch_logs |
| ④ 历史归档(5月) | 36 | 5.2 MB | armored_breakthrough_proper, decision_tree_clip_style, diffeomorphic_tangent_head_sweep, diffeomorphic_tangent_sweep, fisher_operator_consumer_probe, fisher_operator_tokenizer_probe, fisher_style_backbone_probe, fisher_style_memory_adapter_probe, local_repro_sadd_38f_8ep_20260528_224707, manual_k1_k2_8epoch, phase1_diagnostic_probes, physical_loss_tree, reference_memory_generation_probe_full, remote_factorized_tokenizer_pull, router_aware_backbone_probe, scripts, style_memory_bank_adapter_probe, style_memory_bank_adapter_route_probe, style_memory_bank_probe, style_memory_typed_adapter_probe, style_representation_adapter_probe, style_representation_safe_projection_probe, style_representation_style_aware_router_probe, t01_local_base, temp_anneal_proper, tokenizer_adain_gate_calibration, tokenizer_adain_texture_gate_calibration_rerun, tokenizer_bandgate_calibration, tokenizer_prototype_carrier_calibration, tokenizer_stat_reader_probe, tokenizer_stat_vocab_probe, tokenizer_texton_carrier_calibration, vae_backend_256_mse_controls, vae_backend_256_status, wikiart_512_encode_logs, wikiart_512_transfer_logs |
| ④ 历史归档(旧系列) | 6 | 108.9 MB | phase3_task1, fc_sb_r2, 20250618_lite_ot_vertical_auto, p4_fusion_breakout, task4_no_dino, wikiart_stress1_..._b44_remote |
| ⑤ smoke/probe | 27 | 11.1 MB | _smoke_distinct5_*(12个), local_wsl_*(10个), probes_20260601, smoke_blockmask, style_representation_residual_scale_sweep, tmp_output_appearance_resume_smoke, local_wsl_wikiart512_style_injection_from_hist_e3 |
| **合计** | **82** | **~137.7 MB** | |

### B. 可考虑归档后删除（大体积旧系列，需人工确认）— 2 个目录，约 6.4 GB

| 目录名 | 大小 | 修改日期 | 说明 | 建议 |
|--------|------|----------|------|------|
| 620_spatial_bridge | 5183.9 MB | 2026-06-23 | 620系列65个smoke/ablation子实验，已被630系列取代 | 归档后删除，释放5.1GB |
| phase616_live_dashboard | 1220.8 MB | 2026-06-20 | 616系列仪表盘+14个eval.tgz归档，旧系列 | 归档.tgz后删除，释放1.2GB |

### C. 顶层散落文件清理 — 约 4.6 MB

可清理的顶层文件：`tuning_deepdive.zip`（已有解压目录）、各类 `_train.log`/`_err.log`（实验目录内已有）、临时脚本 `_eval_samam_unified.bat`/`_remote_scan_v2.py`。

### 清理效果预估

| 操作 | 释放空间 |
|------|----------|
| A. 立即删除82个无产出目录 | ~137.7 MB |
| B. 归档后删除2个大体积旧系列 | ~6.4 GB |
| C. 清理顶层散落文件 | ~4.6 MB |
| **总计** | **~6.5 GB** |

> 注：重要保留的 120 个目录（含 630_phase4* / 630_local* 主线实验、基线评估、task1/3/4 系列、eval_cache 等）共计 ~25.6 GB，不建议清理。

---

## 附录：扫描方法说明

- 目录大小：`Get-ChildItem -Recurse -File | Measure-Object Length -Sum`，单位 MB 保留1位小数
- config.json 关键字段映射：
  - `dim` ← `model.base_dim`（默认 64）
  - `depth` ← `model.num_res_blocks`（默认 4）
  - `alpha` ← `model.style_extrap_alpha`（默认 0.1，630_local/4J系列为 0.4）
  - `w_ll` ← `bridge.spectral_w_ll`（默认 0.3）
  - `gate_init` ← `model.style_cross_attn_gate_init`（默认 0.05）
  - `gate_mode` ← `model.style_gate_mode`（tanh_gate / fixed_one）
- 临时脚本目录文件计数：625_fc_sb ≈150个, p3_remote_10h ≈130个, tuning_deepdive ≈15个
- 分类规则：
  - keep = 有 epoch_*.pt OR 有 full_eval/ OR 有 eval/ OR 关键基线/资源目录
  - src_only = 有 src/+config.json 但无 ckpt 无 eval
  - temp_scripts = 全是临时脚本（_check_*.sh 等）
  - archive = 2026-05及之前 OR 旧系列废弃目录
  - smoke_probe = _smoke_* / local_wsl_* / probe 实验目录
