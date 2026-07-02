# 本地实验清单 (Local Experiments on G Drive)

**本地路径**: `g:\GitHub\Latent_Style\SchrodingerBridge\exp\`
**整理日期**: 2026-07-02 ~ 2026-07-03
**整理执行**: Deli_AutoResearch cleanup task (M4+M5+M6 已完成, M7 本文)
**数据源**: `local_exp_inventory.md` (202 个目录扫描) + `local_timing.csv` (122 个实验指标采集)

---

## 0. 整理历史

### M4 本地探查（2026-07-02）
- 扫描 `exp/` 子目录：**202 个**，总占用 **25.8 GB**
- 顶层散落文件：32 个 (4.6 MB)
- 分类结果：
  - ① 重要保留 (keep)：120 个目录 / 26.3 GB
  - ② 仅 src 无产出 (src_only)：9 个 / 1.5 MB
  - ③ 临时脚本集 (temp_scripts)：4 个 / 11.0 MB
  - ④ 历史归档 (archive)：42 个 / 114.1 MB
  - ⑤ smoke/probe 可清理：27 个 / 11.1 MB
- 完整清单：`.trae/autoresearch/cleanup/local_exp_inventory.md`

### M5 本地清理 + 重组（2026-07-02 ~ 2026-07-03）
- **删除约 25.7 GB**，明细：
  - `scale/` (18.9 GB) — 5 月旧 wikiart_1024 数据集
  - `_codex_tmp/` (99 MB) — 临时监控文件
  - `aaai_submission_snapshot_*` (30 MB) + 30 个 `tmp_*` 渲染目录
  - `exp/620_spatial_bridge/` (5.2 GB) — 65 个旧 smoke 子实验
  - `exp/phase616_live_dashboard/` (1.2 GB) — 旧仪表盘归档
  - 9 个 src_only + 4 个 temp_scripts + 36 个 May2026 probes + 6 个 old_series
  - 28 个 smoke/probe 目录 + 20 个 exp/ 顶层散落文件
  - 3 个 logs/ 误命名 eval 目录 (125 MB)
- **重组 113 个目录** 到 5 个分组结构（同文件系统内 Move-Item，瞬间完成）
- 完整日志：
  - `.trae/autoresearch/cleanup/logs/m5_cleanup.log`
  - `.trae/autoresearch/cleanup/logs/m5_reorg.log`

### M6 本地数据集检查（2026-07-03）
- `datasets/` 为空，无本地数据集重复
- 训练 / 评估数据集统一从 `I:\` 盘读取（远程）

### 当前 `exp/` 重组后结构 (2026-07-03 exp reorg, 按实验脉络+数据集物理分离)

```
exp/
├── FCSB/              (93 个 distinct5 主线: early 3 + phase4 66 + local_t 24)
│   ├── early/         (3 个: clean_base_v2_local, clean_base_v2_relu2, 628_ablation)
│   ├── phase4/        (66 个: 630_phase1d-4j6 系统消融实验)
│   └── local_t/       (24 个: 630_local T/R 系列实验)
├── baseline/          (3 个: reeval, images, v2)
├── 256/               (256 分辨率历史实验占位, 非主线)
├── wiki5/             (11 个 wikiarts5 非主线: smoke 10 + full/task4_iter 16 子目录)
├── fewshot6/          (3 个: 630_phase4j6_fewshot_popart{,_v2,_v3}, 非主线)
├── legacy/            (shared 7 + logs 12 散落文件)
└── README.md
```

---

## 1. Baseline 评估实验 (`baseline/`)

数据源：`baseline/reeval/`（重评估目录）+ `baseline/v2/`
评估协议：`run_evaluation.py`，HF transformers CLIP (ViT-B/32)，LPIPS (Alex)，750 pairs
test_dir：`I:\wikiart_distinct5_samam_512_classview\test`

### 1.1 `baseline_reeval/` — 论文用 baseline 重评估结果

| # | 目录 | 模型 | mtime | n_pairs | CLIP-S | LPIPS | 说明 |
|---|------|------|-------|---------|--------|-------|------|
| 1 | `identity/` | Identity | 2026-06-30 | 750 | 0.6933 | 0.0000 | 基线（无变换） |
| 2 | `identity_vgg/` | Identity (VGG19) | 2026-06-30 | - | - | - | VGG19 评估对照（仅 metrics.csv） |
| 3 | `adain_bad/` | AdaIN (退化) | 2026-06-30 | 750 | 0.6654 | 0.7448 | 早期配置，已弃用 |
| 4 | `adain_v32k/` | AdaIN (v32k) | 2026-06-30 | 750 | 0.6679 | 0.7425 | 主 AdaIN 结果 |
| 5 | `adain_vgg19/` | AdaIN (VGG19) | 2026-06-30 | 750 | - | - | VGG19 评估对照 |
| 6 | `sdturbo/` | SD-Turbo | 2026-06-30 | 750 | 0.6933 | 0.0033 | 几乎无变换 |
| 7 | `sdedit_010/` | SDEdit s=0.10 | 2026-06-24 | - | - | - | 按用户要求不再用于论文 |
| 8 | `sdedit_020/` | SDEdit s=0.20 | 2026-06-24 | - | - | - | 按用户要求不再用于论文 |
| 9 | `sdedit_035/` | SDEdit s=0.35 | 2026-06-30 | 750 | 0.7797 | 0.4508 | 论文保留点 |
| 10 | `sdedit_040/` | SDEdit s=0.40 | 2026-06-30 | 750 | 0.7934 | 0.4826 | 论文保留点 |

**说明**：
- 完整 12 baseline 对照表（含 StyleID/CUT/SaMST/SeeDream 等）见 `docs/exp/remote_experiments.md` 第 1.1 节，本目录仅存放重评估子集
- SaMam 收敛曲线实验存于远程 `I:\exp_samam\training\`，见 remote_experiments.md 第 2 节

### 1.2 `baseline_v2/` — 基线 v2 评估归档

| 子目录 / 文件 | 内容 |
|---------------|------|
| `eval/unified_results.json` | 统一结果汇总 |
| `eval/wct_a06_summary.json` | WCT α=0.6 评估 |
| `eval/wct_summary.json` / `wct_vgg19_summary.json` | WCT 评估（两个后端） |
| `baseline_summary_table.csv` | 基线汇总表 |
| `baseline_conclusions.md` | 基线结论文档 |

---

## 2. 早期实验 (`FCSB/early/` + `wiki5/`) — 14 个目录

> 2026-06-23 ~ 2026-06-30 的早期探索实验，包含 task1/3/4 系列、clean_base_v2 基线、628 消融。

### 2.1 task1 / task3 / task4 系列（2026-06-23）

数据集：`wikiart_distinct5_samam_512_classview` (5 风格: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
评估：HF transformers CLIP, LPIPS (Alex)

| 目录 | 模型配置 | mtime | epochs | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|-------|--------|--------------|----------|--------|-------|------|
| `task1_endpoint_film_baseline/` | FiLM + GroupNorm | 2026-06-23 | 1 | epoch_0001 | 59.02 | 0.7035 | 0.2610 | Task1 FiLM 基线 |
| `task1_endpoint_film_no_norm/` | FiLM 无 GN | 2026-06-23 | 1 | epoch_0001 | 58.05 | 0.7021 | 0.2691 | Task1 无归一化对照 |
| `task3_baseline_1ep/` | dim=64, depth=4, tanh_gate | 2026-06-23 | 1 | epoch_0001 | 63.88 | 0.7037 | 0.2567 | Task3 基线（gate_mode=tanh_gate） |
| `task3_combo_a_1ep/` | fixed_one | 2026-06-23 | 1 | epoch_0001 | 64.07 | 0.7034 | 0.2606 | Task3 组合 A |
| `task3_combo_b_3ep/` | fixed_one | 2026-06-23 | 3 | epoch_0003 | 63.44 | 0.7072 | 0.2648 | Task3 组合 B 3ep |
| `task4_style_strength_baseline_2ep/` | fixed_one | 2026-06-23 | 2 | epoch_0002 | 62.71 | 0.7060 | 0.2757 | Task4 风格强度基线 |
| `task4_style_strength_w05_2ep/` | fixed_one, w=0.5 | 2026-06-23 | 2 | epoch_0002 | 63.35 | 0.7064 | 0.2753 | Task4 w=0.5 |
| `task4_style_strength_w10_2ep/` | fixed_one, w=1.0 | 2026-06-23 | 2 | epoch_0002 | 62.97 | 0.7062 | 0.2750 | Task4 w=1.0 |

### 2.2 task4_iter/ — Task4 迭代实验集（2026-06-24，16 个子实验）

> r1a-r7 系列迭代实验，含 src/ 和 config.json，部分有 ckpt。
> 总占用：2686 MB（最大子实验 r3c_optimal_5ep 含 epoch_0003.pt）

| 子目录 | epochs | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|--------|--------|--------------|----------|--------|-------|------|
| `p2_long_10ep/` | 10 | epoch_0010 | 100.77 | 0.6720 | 0.6042 | p2 长训练 10ep（LPIPS 退化严重） |
| `r1a_no_dino_minimal/` | 2 | epoch_0002 | 92.89 | 0.6966 | 0.4737 | 无 DINO 最小配置 |
| `r1b_no_dino_nogn/` | 2 | epoch_0002 | 118.51 | 0.6963 | 0.4736 | 无 DINO 无 GN |
| `r1c_no_dino_fixedone/` | 2 | epoch_0002 | 132.94 | 0.7032 | 0.4598 | 无 DINO fixed_one |
| `r2a_with_film/` | 2 | epoch_0002 | 100.55 | 0.7037 | 0.4538 | 含 FiLM |
| `r2b_with_antiwhiten/` | 2 | epoch_0002 | 93.34 | 0.7065 | 0.4403 | 含 antiwhiten |
| `r3a_aggressive_antwhiten/` | 3 | epoch_0003 | 77.12 | 0.7060 | 0.4696 | 激进 antiwhiten |
| `r3b_endpoint_antiwhiten/` | 3 | epoch_0003 | 126.65 | 0.7061 | 0.5243 | 端点 antiwhiten |
| `r3c_optimal_5ep/` | 5 | epoch_0005 | 75.56 | 0.7053 | 0.5045 | 最优候选 5ep |
| `r4c_velocity_magnitude/` | 3 | epoch_0003 | 98.63 | 0.7062 | 0.4738 | velocity magnitude |
| `r4d1_velmag_high/` | 3 | epoch_0003 | 98.71 | 0.7030 | 0.4939 | 高 velmag |
| `r6_pixel_color_fix/` | 3 | epoch_0003 | 102.69 | 0.7079 | 0.4766 | 像素颜色修复 |
| `r7_saturation_loss/` | 5 | epoch_0005 | 103.97 | 0.7024 | 0.5109 | 饱和度损失 |

**未评估子目录**：`r1a_latent_baseline`, `r4a_velocity_scaling`, `r5_diagnosis` — 无 full_eval 产出

### 2.3 phase3_task2 系列（2026-06-24）

| 目录 | 配置 | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|--------------|----------|--------|-------|------|
| `phase3_task2_p3d_contrastive_w01_margin01/` | 对比损失 w=0.1, margin=0.1 | epoch_0003 | 78.78 | 0.7019 | 0.2804 | P3-D |
| `phase3_task2_p3e_contrastive_w05_margin005/` | 对比损失 w=0.5, margin=0.05 | epoch_0003 | 73.43 | 0.7022 | 0.2811 | P3-E |

### 2.4 clean_base_v2 系列（2026-06-30，Phase4 干净基线）

| 目录 | 配置 | mtime | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|-------|------|--------------|----------|--------|-------|------|
| `clean_base_v2_local/` | dim=64, depth=4, α=0.1, w_ll=0.3, 10ep | 2026-06-30 | epoch_0005/0010 | epoch_0010_full | 5.58 | 0.7292 | 0.3239 | Phase4 干净基线（10ep） |
| `clean_base_v2_relu2/` | relu2 注意力 | 2026-06-30 | epoch_0003 | epoch_0003_full | 5.95 | 0.7269 | 0.3370 | M9 修复: relu2 注意力 |

### 2.5 628_ablation（2026-06-29）

| 目录 | 内容 | 说明 |
|------|------|------|
| `628_ablation/` | `p8c_rescan_results.json` | 628 系列消融重扫描结果（仅 JSON） |

---

## 3. Phase4 消融实验 (`FCSB/phase4/`) — 66 个目录

> 2026-06-30 ~ 2026-07-01 的 Phase4 系统消融实验。
> 默认配置：dim=64, depth=4, gate_init=0.05, style_gate_mode=tanh_gate, body_block_type=global_attn, α=0.1, w_ll=0.3
> 数据集：`wikiart_distinct5_samam_512_classview`
> 评估：HF transformers CLIP, LPIPS (Alex), 5 风格 × 150 pairs = 750 pairs

### 3.1 Phase1d/2b/2c/3 验证系列（2026-06-30，7 个）

| 目录 | mtime | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|-------|------|--------------|----------|--------|-------|------|
| `630_phase1d_verify/` | 2026-06-30 | epoch_0002 | epoch_0002_full | 6.42 | 0.7224 | 0.3231 | Phase1d 验证 |
| `630_phase1d_verify_v2/` | 2026-06-30 | epoch_0003 | epoch_0003_full | 82.73 | 0.7251 | 0.3373 | Phase1d 验证 v2 |
| `630_phase2b_mask_random_50/` | 2026-06-30 | epoch_0003 | epoch_0003_full | 83.16 | 0.7261 | 0.3296 | 随机 mask 50% |
| `630_phase2c_mask_random_75/` | 2026-06-30 | epoch_0003 | epoch_0003_full | 83.78 | 0.7250 | 0.3278 | 随机 mask 75% |
| `630_phase2c_mask_shuffle_50/` | 2026-06-30 | epoch_0003 | epoch_0003_full | 85.72 | 0.7234 | 0.3205 | shuffle mask 50% |
| `630_phase2c_mask_shuffle_75/` | 2026-06-30 | epoch_0003 | epoch_0003_full | 85.90 | 0.7232 | 0.3177 | shuffle mask 75% |
| `630_phase3_mask_random_50_10ep/` | 2026-06-30 | epoch_0010 | epoch_0010 | 86.56 | 0.7289 | 0.3370 | 10ep 长训练 |

### 3.2 Phase4A-2 消融（2026-07-01，3 个）

| 目录 | 配置变更 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|------|--------------|----------|--------|-------|------|
| `630_phase4a2_adain_0/` | endpoint_adain=0 | epoch_0003 | epoch_0003 | 75.95 | 0.7082 | 0.2994 | 禁用推理 AdaIN |
| `630_phase4a2_extrap_0/` | style_extrap_alpha=0 | epoch_0003 | epoch_0003 | 77.38 | 0.7242 | 0.3333 | 禁用外推 |
| `630_phase4a2_w_ll_0/` | w_ll=0 | epoch_0003 | epoch_0003 | 76.12 | 0.7117 | 0.3120 | 禁用低频损失 |

### 3.3 Phase4B 频域 mask 系列（2026-07-01，9 个）

| 目录 | 配置 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|--------------|----------|--------|-------|------|
| `630_phase4b1_freq_a05/` | α=0.5 | epoch_0003 | epoch_0003 | 74.89 | 0.7252 | 0.3347 | 频域 α=0.5 |
| `630_phase4b1_freq_a1/` | α=1.0 | epoch_0003 | epoch_0003 | 74.50 | 0.7258 | 0.3357 | 频域 α=1.0 |
| `630_phase4b1_freq_a1_rand50/` | α=1.0 + 随机 50% | epoch_0003 | epoch_0003 | 75.79 | 0.7264 | 0.3354 | 频域 + 随机 dropout |
| `630_phase4b2_freq_a1_rand30/` | α=1.0 + 随机 30% | epoch_0003 | epoch_0003 | 75.45 | 0.7250 | 0.3252 | 频域 + 30% |
| `630_phase4b2_freq_a1_rand50_10ep/` | α=1.0 + 50%, 10ep | epoch_0010 | epoch_0010 | 74.98 | 0.7277 | 0.3394 | 最佳配置长训练 |
| `630_phase4b2_freq_a1_rand70/` | α=1.0 + 随机 70% | epoch_0003 | epoch_0003 | 75.53 | 0.7245 | 0.3284 | 频域 + 70% |
| `630_phase4b3_dwt_a1/` | Haar DWT α=1.0 | epoch_0003 | epoch_0003 | 75.73 | 0.7266 | 0.3402 | Haar DWT |
| `630_phase4b3_dwt_a1_rand50/` | DWT + 随机 50% | epoch_0003 | epoch_0003 | 90.85 | 0.7255 | 0.3297 | DWT + dropout |
| `630_phase4c_blockmask_r60_b128_lvl2/` | RGB 块 mask r=0.6 b=128 | epoch_0003 | epoch_0003 | 108.60 | 0.7151 | 0.3177 | RGB 块 mask |

### 3.4 Phase4C-D DWT 低通系列（2026-07-01，5 个）

| 目录 | 配置 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|--------------|----------|--------|-------|------|
| `630_phase4c_dino_clean_lvl2/` | 真实 DINO + 2 级 DWT | epoch_0003 | epoch_0003 | 95.31 | 0.7118 | 0.3038 | DINO 对照 |
| `630_phase4d_lvl2/` | 2 级 Haar DWT 低通 | epoch_0003 | epoch_0003 | 75.44 | **0.7301** | 0.3402 | **4D SOTA clip=0.7301** |
| `630_phase4d_lvl2_dwt_rand50/` | 2 级 DWT + 随机 50% | epoch_0003 | epoch_0003 | 76.56 | 0.7294 | 0.3394 | 4D 组合 |
| `630_phase4e_db2_lvl1/` | Daubechies-2 单级 | epoch_0003 | epoch_0003 | 79.21 | 0.7258 | 0.3288 | db2 单级 |
| `630_phase4e_db2_lvl2/` | db2 + 2 级级联 | epoch_0003 | epoch_0003 | 86.47 | 0.7298 | 0.3398 | db2 2 级 |

### 3.5 Phase4F-G 多级 DWT + per-subband 系列（2026-07-01，11 个）

| 目录 | 配置 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|--------------|----------|--------|-------|------|
| `630_phase4f_lvl3/` | 3 级 Haar DWT | epoch_0003 | epoch_0003 | 77.57 | 0.7319 | 0.3428 | 3 级 DWT |
| `630_phase4f_lvl4/` | 4 级 Haar DWT | epoch_0003 | epoch_0003 | 76.05 | 0.7316 | 0.3461 | 4 级 DWT |
| `630_phase4g1a_lock_ll/` | 推理 LL 锁死 | epoch_0003 | epoch_0003 | 76.90 | 0.7178 | 0.3281 | LL 锁死 |
| `630_phase4g1b_lock_ll_zero_wll/` | LL 锁死 + w_ll=0 | epoch_0003 | epoch_0003 | 75.20 | 0.7174 | 0.3372 | LL 锁死 + 无 LL loss |
| `630_phase4g2_per_subband/` | per-subband AdaIN | epoch_0003 | epoch_0003 | 75.18 | **0.7361** | 0.3843 | **4G.2 SOTA clip=0.7361** |
| `630_phase4g2b_per_subband_a05/` | per-subband α=0.5 | epoch_0003 | epoch_0003 | 74.95 | 0.7362 | 0.3845 | per-subband α=0.5 |

### 3.6 Phase4H SOTA 优化系列（2026-07-01，14 个）

| 目录 | 配置变更 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|------|--------------|----------|--------|-------|------|
| `630_phase4h1a_eota_per_subband/` | EOTA + per-subband α=1.0 | epoch_0003 | epoch_0003 | 70.90 | 0.7359 | 0.3853 | EOTA per-subband |
| `630_phase4h1b_eota_per_subband_a05/` | α=0.5 | epoch_0003 | epoch_0003 | 72.13 | 0.7219 | 0.3226 | 关键对照 |
| `630_phase4h1c_eota_per_subband_a07/` | α=0.7 | epoch_0003 | epoch_0003 | 71.54 | 0.7280 | 0.3442 | EOTA α=0.7 |
| `630_phase4h1d_eota_per_subband_a08/` | α=0.8 | epoch_0003 | epoch_0003 | 71.93 | 0.7309 | 0.3572 | EOTA α=0.8 |
| `630_phase4h1e_eota_spatial_fiber_a05/` | spatial_fiber α=0.5 | epoch_0003 | epoch_0003 | 71.68 | 0.7185 | 0.3095 | spatial fiber |
| `630_phase4h1f_eota_spatial_fiber_a07/` | α=0.7 | epoch_0003 | epoch_0003 | 71.98 | 0.7231 | 0.3208 | 关键 |
| `630_phase4h1g_eota_spatial_fiber_a08/` | α=0.8 | epoch_0003 | epoch_0003 | 72.56 | 0.7251 | 0.3281 | spatial α=0.8 |
| `630_phase4h1g5ep_eota_spatial_fiber_a08/` | α=0.8, 5ep | epoch_0005 | epoch_0005 | 72.48 | 0.7261 | 0.3279 | 新 SOTA |
| `630_phase4h2h_sota_w_hf_15/` | w_lh=w_hl=1.5 | epoch_0003 | epoch_0003 | 71.71 | 0.7250 | 0.3330 | 高频权重 1.5 |
| `630_phase4h2i_per_subband_a07_w_ll_05/` | α=0.7 + w_ll=0.5 | epoch_0003 | epoch_0003 | 71.99 | 0.7265 | 0.3389 | per-subband + w_ll |
| `630_phase4h3f_sota_patch_1359_15/` | swd_patch +15 | epoch_0003 | epoch_0003 | 71.76 | 0.7252 | 0.3280 | patch SWD |
| `630_phase4h4e_sota_depth6/` | depth=6 | epoch_0003 | epoch_0003 | 77.71 | 0.7265 | 0.3366 | depth=6 |
| `630_phase4h4f_sota_dim96/` | dim=96 | epoch_0003 | epoch_0003 | 71.78 | 0.7271 | 0.3368 | dim=96 |
| `630_phase4h4g_sota_dim96_5ep/` | dim=96, 5ep | epoch_0005 | epoch_0005 | 71.72 | 0.7268 | 0.3313 | dim=96 5ep |
| `630_phase4h5e_sota_mask25/` | mask=0.25 | epoch_0003 | epoch_0003 | 71.46 | 0.7227 | 0.3172 | mask 25% |
| `630_phase4h5f_sota_mask75/` | mask=0.75 | epoch_0003 | epoch_0003 | 73.33 | 0.7237 | 0.3272 | mask 75% |
| `630_phase4h7d_sota_terminal_swd_03/` | terminal_swd=0.3 | epoch_0003 | epoch_0003 | 71.94 | 0.7251 | 0.3281 | terminal SWD |

### 3.7 Phase4I 求解器/调度系列（2026-07-01，14 个）

| 目录 | 配置变更 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|------|--------------|----------|--------|-------|------|
| `630_phase4i1a_eota_per_subband_multi_alpha/` | LH=HL=0.5, HH=0.9 | epoch_0003 | epoch_0003 | 71.64 | 0.7263 | 0.3383 | 多尺度 alpha |
| `630_phase4i1d_eota_per_subband_hh_only/` | HH=1.0 | epoch_0003 | epoch_0003 | 71.73 | 0.7310 | 0.3576 | HH-only 对照 |
| `630_phase4i2a_sota_heun/` | Heun 求解器 | epoch_0003 | epoch_0003 | 85.92 | 0.7260 | 0.3279 | Heun |
| `630_phase4i2b_sota_heun_5ep/` | Heun, 5ep | epoch_0005 | epoch_0005 | 86.57 | 0.7266 | 0.3229 | Heun 5ep |
| `630_phase4i5a_sota_heun_cosine/` | Heun + 余弦 | epoch_0003 | epoch_0003 | 86.16 | 0.7256 | 0.3238 | Heun 余弦 |
| `630_phase4i5b_sota_heun_cosine_5ep/` | + 5ep | epoch_0005 | epoch_0005 | 91.37 | 0.7262 | 0.3171 | Heun 余弦 5ep |
| `630_phase4i5c_sota_heun_rquad_5ep/` | + rquad | epoch_0005 | epoch_0005 | 90.91 | 0.7293 | 0.3429 | Heun rquad 5ep |
| `630_phase4i6a_sota_rk4_5ep/` | RK4, 5ep | epoch_0005 | epoch_0005 | 140.73 | 0.7265 | 0.3235 | RK4 5ep |
| `630_phase4i7a_cosine_heun_a09_5ep/` | 余弦+Heun+α=0.9 | epoch_0005 | epoch_0005 | 112.12 | 0.7283 | 0.3255 | α=0.9 5ep |
| `630_phase4i7b_cosine_heun_a085_5ep/` | α=0.85 | epoch_0005 | epoch_0005 | 111.25 | 0.7272 | 0.3218 | α=0.85 5ep |
| `630_phase4i8a_cosine_heun_a085_8ep/` | α=0.85, 8ep | epoch_0008 | epoch_0008 | 104.16 | 0.7284 | 0.3283 | α=0.85 8ep |
| `630_phase4i8b_warpcos_p08_a085_5ep/` | warp_cos p=0.8 | epoch_0005 | epoch_0005 | 95.02 | 0.7282 | 0.3271 | warp cos |
| `630_phase4i9_wct_a085_5ep/` | WCT 替代 AdaIN | epoch_0005 | epoch_0005 | 112.40 | 0.7319 | 0.3568 | WCT 5ep |
| `630_phase4i10b_ept_t01/` | EPT t=[0,0.1] | epoch_0005 | epoch_0005 | 92.26 | 0.7153 | 0.4474 | EPT 端点预测 |

### 3.8 Phase4J 路由/Few-shot 系列（2026-07-01，9 个）

> α=0.4 系列，含 train.log（630_phase4j1_dwt_route 有 run.log）

| 目录 | 配置 | ckpt | train_min | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|-----------|--------------|----------|--------|-------|------|
| `630_phase4j1_dwt_route/` | DWT 路由 (方案 B) | epoch_0005 | 3.03 | epoch_0005 | 93.55 | 0.7226 | 0.3068 | DWT 路由 |
| `630_phase4j2_wct_aligned/` | WCT 对齐 (方案 A) | epoch_0005 | - | epoch_0005 | 135.72 | 0.7153 | 0.3003 | WCT 对齐目标 |
| `630_phase4j3_fewshot_stylemem/` | Few-shot style_mem | epoch_0005 | - | epoch_0005 | 111.69 | 0.7218 | 0.3020 | Few-shot 优化 |
| `630_phase4j4_progressive_alpha/` | 渐进 Alpha (方案 C) | epoch_0005 | - | epoch_0005 | 132.89 | 0.7326 | 0.4126 | 渐进调度 |
| `630_phase4j5_wct_aligned_progressive/` | 方案 A+C 综合 | epoch_0005 | - | epoch_0005 | 150.41 | 0.7274 | 0.3858 | 最优候选 |
| `630_phase4j6_fewshot_popart/` | Few-shot Pop_Art | epoch_0005 | - | epoch_0005 | 120.62 | 0.7210 | 0.3069 | Pop_Art |
| `630_phase4j6_fewshot_popart_v2/` | 高 LR + 15ep | epoch_0015 | - | epoch_0015 | 113.24 | 0.7213 | 0.3117 | v2 高 LR |
| `630_phase4j6_fewshot_popart_v3/` | 高 LR + 15ep | epoch_0015 | - | epoch_0015 | 141.80 | 0.7214 | 0.3106 | v3 高 LR |

---

## 4. 630_local T/R 系列 (`FCSB/local_t/`) — 24 个目录

> 2026-07-01 ~ 2026-07-02 的当前主线实验。
> 默认配置：dim=64, depth=4, α=0.4, w_ll=0.3, gate_init=0.05
> 数据集：`wikiart_distinct5_samam_512_classview`
> 部分实验有 train.log，训练时长见下表

### 4.1 R1/R2/R3 反向实验（2026-07-02）

| 目录 | 配置变更 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|------|--------------|----------|--------|-------|------|
| `630_local_r1_depth2/` | depth=2 | epoch_0005 | epoch_0005 | 84.65 | 0.7173 | 0.2705 | R1 反向 depth=2 |
| `630_local_r2_dim32/` | dim=32 | epoch_0005 | epoch_0005 | 101.63 | 0.7153 | 0.2705 | R2 反向 dim=32 |
| `630_local_r3_gate_init0/` | gate_init=0.0 | epoch_0005 | epoch_0005 | 101.71 | 0.7080 | 0.2641 | R3 反向 gate=0 |

### 4.2 T2/T5 早期对照（2026-07-01~02）

| 目录 | 配置 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|--------------|----------|--------|-------|------|
| `630_local_t2_soft_ll_t2a/` | Soft LL α=0.05 | epoch_0005 | epoch_0005 | 100.09 | 0.7277 | 0.3379 | T2 Soft LL |
| `630_local_t5_eval_only_dwt/` | Eval-Only DWT | epoch_0005 | epoch_0005 | 96.85 | 0.7061 | 0.2606 | T5 Eval-Only DWT |

### 4.3 T10-T16 随机 DWT + LLGQCA 系列（2026-07-02）

> 含 train.log，训练时长约 3 分钟（5 ep）

| 目录 | 配置 | ckpt | train_min | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|-----------|--------------|----------|--------|-------|------|
| `630_local_t10_stochastic_dwt/` | 随机 DWT p=0.5 | epoch_0005 | 3.08 | epoch_0005 | 95.22 | 0.7083 | 0.2480 | T10 p=0.5 |
| `630_local_t11_stochastic_dwt_p08/` | 随机 DWT p=0.8 | epoch_0005 | 3.08 | epoch_0005 | 97.25 | **0.7213** | 0.2868 | **T11 p=0.8 局部 SOTA 基底** |
| `630_local_t13_ll_global_style_inject/` | LLGSI 全局统计注入 | epoch_0005 | 2.97 | epoch_0005 | 93.68 | 0.7128 | 0.2706 | T13 LLGSI |
| `630_local_t14_casi/` | CASI 交叉注意力 | epoch_0005 | 3.03 | epoch_0005 | 95.18 | 0.7152 | 0.2795 | T14 CASI |
| `630_local_t15_llgqca/` | LLGQCA LL 全局 query | epoch_0005 | - | epoch_0005 | 94.14 | 0.7176 | 0.2764 | T15 LLGQCA |
| `630_local_t16a_llgqca_gate02/` | gate=0.2 | epoch_0005 | - | epoch_0005 | 93.74 | 0.7145 | 0.2706 | T16a gate=0.2 |
| `630_local_t16b_llgqca_gate03/` | gate=0.3 | epoch_0005 | - | epoch_0005 | 93.60 | 0.7101 | 0.2681 | T16b gate=0.3 |
| `630_local_t16c_llgqca_gate05/` | gate=0.5 | epoch_0005 | - | epoch_0005 | 94.06 | 0.7108 | 0.2688 | T16c gate=0.5 |

### 4.4 T18/T19 损失/容量调整（2026-07-02）

| 目录 | 配置变更 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|----------|------|--------------|----------|--------|-------|------|
| `630_local_t18a_wll05/` | w_ll=0.5 | epoch_0005 | epoch_0005 | 95.35 | 0.7174 | 0.2774 | T18a 恢复 LL 训练 |
| `630_local_t18b_wll10/` | w_ll=1.0 | epoch_0005 | epoch_0005 | 94.67 | 0.7180 | 0.2764 | T18b 完全恢复 LL |
| `630_local_t19a_depth6/` | depth=6 | epoch_0005 | epoch_0005 | 108.11 | NaN | NaN | T19a 数值不稳定 (WCT NaN) |
| `630_local_t19b_dim96/` | dim=96 | epoch_0005 | epoch_0005 | 95.72 | 0.7207 | 0.3142 | T19b 欠拟合 |

### 4.5 T20-T26 PlanB-H 结构变体（2026-07-02）

| 目录 | 方案 | ckpt | eval_summary | eval_sec | CLIP-S | LPIPS | 说明 |
|------|------|------|--------------|----------|--------|-------|------|
| `630_local_t20_structure_aligned_target/` | PlanB 结构对齐 Flow Matching | epoch_0005 | epoch_0005 | 94.43 | 0.7037 | 0.2722 | T20 PlanB |
| `630_local_t21_adaln_zero_ll/` | PlanC AdaLN-Zero on LL | epoch_0005 | epoch_0005 | 100.96 | 0.7174 | 0.2790 | T21 PlanC |
| `630_local_t22_tone_bias/` | PlanD 直接 Tone Bias | epoch_0005 | epoch_0005 | 100.49 | 0.7150 | 0.2597 | T22 PlanD |
| `630_local_t23_ll_mean_only/` | PlanE 仅迁移 LL mean | epoch_0005 | epoch_0005 | 99.06 | 0.7215 | 0.3856 | T23 PlanE |
| `630_local_t24_ll_std_only/` | PlanF 仅迁移 LL std | epoch_0005 | epoch_0005 | 99.24 | 0.7172 | 0.2889 | T24 PlanF |
| `630_local_t25_ll_cov_only/` | PlanG 仅迁移 LL 协方差 | epoch_0005 | epoch_0005 | 98.63 | 0.7205 | 0.3069 | T25 PlanG |
| `630_local_t26_ll_ycbcr/` | PlanH YCbCr 色彩解耦 | epoch_0005 | epoch_0005 | 169.52 | 0.7200 | 0.3085 | T26 PlanH |

---

## 5. 共享资源 (`legacy/shared/`) — 7 个目录

| 目录 | 大小 | mtime | 内容 | 说明 |
|------|------|-------|------|------|
| `adain_checkpoints/` | 100.9 MB | 2026-06-30 | `decoder_v32k.pth` | AdaIN 预训练权重 |
| `eval_cache/` | 874.2 MB | 2026-07-01 | CLIP 特征 (ref_feats/src_feats), offline_pairing | full_eval 依赖缓存 |
| `clean_base/` | 0 MB | 2026-06-29 | 空 (有 full_eval/ 但无内容) | 占位目录 |
| `630_local_t12_eval/` | 421.3 MB | 2026-07-02 | t12a-t12e 子目录 + batch_results.json | T12 批量 eval 结果 |
| `630_local_t4_eval/` | 424.0 MB | 2026-07-02 | t3b-t4c 子目录 + batch_results.json | T4 批量 eval 结果 |
| `630_planA_zero_step_wct/` | 250.5 MB | 2026-07-02 | 3 子目录 + results.json | PlanA 零步 WCT 评估 |

---

## 6. 顶层散落日志文件 (`exp/`)

> 这些是训练日志的顶层副本（实验目录内通常已有 train.log）

| 文件 | 关联实验 | 说明 |
|------|----------|------|
| `630_local_t23_ll_mean_only_train.log` | T23 | PlanE 训练日志 |
| `630_local_t24_ll_std_only_train.log` | T24 | PlanF 训练日志 |
| `630_local_t25_ll_cov_only_train.log` | T25 | PlanG 训练日志 |
| `630_phase4b1_freq_a1_rand50_train.log` | 4B1 | 频域训练日志 |
| `630_phase4b2_freq_a1_rand50_10ep_train.log` | 4B2 10ep | 长训练日志 |
| `630_phase4j3_fewshot_stylemem_run.log` / `_run2.log` | 4J.3 | Few-shot 运行日志 |
| `630_phase4j6_fewshot_popart_run.log` / `_v2_run.log` | 4J.6 | Pop_Art 运行日志 |

---

## 7. 关键发现汇总 (Key Findings)

### 7.1 SOTA 实验排行（CLIP-S 降序）

| 排名 | 实验 | CLIP-S | LPIPS | 1-LPIPS | Pareto 评价 | 位置 |
|------|------|--------|-------|---------|-------------|------|
| 1 | 630_phase4g2b_per_subband_a05 | 0.7362 | 0.3845 | 0.6155 | 高 CLIP / 中 LPIPS | phase4 |
| 2 | 630_phase4g2_per_subband | 0.7361 | 0.3843 | 0.6157 | 4G.2 SOTA | phase4 |
| 3 | 630_phase4h1a_eota_per_subband | 0.7359 | 0.3853 | 0.6147 | EOTA per-subband | phase4 |
| 4 | 630_phase4j4_progressive_alpha | 0.7326 | 0.4126 | 0.5874 | 渐进 Alpha（高 CLIP 高 LPIPS） | phase4 |
| 5 | 630_phase4f_lvl3 | 0.7319 | 0.3428 | 0.6572 | 3 级 DWT（平衡点） | phase4 |
| 5 | 630_phase4i9_wct_a085_5ep | 0.7319 | 0.3568 | 0.6432 | WCT 5ep | phase4 |
| 7 | 630_phase4f_lvl4 | 0.7316 | 0.3461 | 0.6539 | 4 级 DWT | phase4 |
| 8 | 630_phase4h1d_eota_per_subband_a08 | 0.7309 | 0.3572 | 0.6428 | EOTA α=0.8 | phase4 |
| 9 | 630_phase4i1d_eota_per_subband_hh_only | 0.7310 | 0.3576 | 0.6424 | HH-only | phase4 |
| 10 | 630_phase4d_lvl2 | 0.7301 | 0.3402 | 0.6598 | 4D（低 LPIPS 优良点） | phase4 |

### 7.2 T11 局部 SOTA（630_local 系列，α=0.4 路线）

| 实验 | CLIP-S | LPIPS | 1-LPIPS | 说明 |
|------|--------|-------|---------|------|
| 630_local_t11_stochastic_dwt_p08 | 0.7213 | 0.2868 | 0.7132 | **T11 局部 SOTA**（最低 LPIPS） |
| 630_local_t2_soft_ll_t2a | 0.7277 | 0.3379 | 0.6621 | Soft LL（高 CLIP） |
| 630_local_t23_ll_mean_only | 0.7215 | 0.3856 | 0.6144 | PlanE |
| 630_local_t19b_dim96 | 0.7207 | 0.3142 | 0.6858 | dim=96 |

### 7.3 训练时长统计

| 来源 | 实验数 | 平均训练时长 | 说明 |
|------|--------|--------------|------|
| train.log（630_local T10-T14 + 4J.1） | 5 | ~3.0 min | 5 ep, dim=64, depth=4 本地训练 |
| 无 train.log（其余 117 个） | 117 | - | 训练日志未保留或远程训练 |

### 7.4 评估时长统计

| 分位数 | eval_sec |
|--------|----------|
| min | 5.58 (clean_base_v2_local, full 评估) |
| 25% | ~72 |
| median | ~85 |
| 75% | ~100 |
| max | 169.52 (T26 YCbCr) |

**说明**：eval_sec = `timings_sec.wall_total`，包含 VAE 编码/解码、CLIP 特征提取、LPIPS 计算、750 pairs 全评估

---

## 8. 已删除目录备忘（M5 清理记录）

> 以下目录已在 M5 阶段删除，记录在此供追溯。详细日志见 `logs/m5_cleanup.log`。

### 8.1 大体积删除（>100 MB）

| 目录 | 大小 | mtime | 删除原因 |
|------|------|-------|----------|
| `scale/` | 18.9 GB | 2026-05 | 旧 wikiart_1024 数据集（已被 I 盘数据集取代） |
| `exp/620_spatial_bridge/` | 5.2 GB | 2026-06-23 | 65 个旧 smoke 子实验，已被 630 系列取代 |
| `exp/phase616_live_dashboard/` | 1.2 GB | 2026-06-20 | 616 系列仪表盘 + 14 个 eval.tgz 归档 |
| `_codex_tmp/` | 99 MB | 2026-06 | 临时监控/打包文件 |
| 3 个 logs/ 误命名 eval 目录 | 125 MB | - | logs/ 下误命名的 eval 输出 |

### 8.2 批量删除（小体积）

| 类别 | 数量 | 总大小 | 删除原因 |
|------|------|--------|----------|
| src_only（仅 src 无产出） | 9 | 1.5 MB | 无 ckpt 无 eval，仅代码副本 |
| temp_scripts（临时脚本） | 4 | 11.0 MB | 全是 _check_*.sh / debug*.sh |
| May2026 probes | 36 | 5.2 MB | 2026-05 早期 probe/calibration，无产出 |
| old_series（旧系列废弃） | 6 | 108.9 MB | phase3_task1, fc_sb_r2 等已废弃 |
| smoke/probe | 28 | 11.1 MB | _smoke_* / local_wsl_* 系列 |
| exp/ 顶层散落文件 | 20 | 4.6 MB | 各类 _train.log / _err.log / 临时脚本 |
| aaai_submission tmp_* | 30 | 30 MB | 临时 PDF 渲染目录 |

---

## 附录 A: 数据采集方法

### A.1 目录扫描
- 工具：PowerShell `Get-ChildItem -Recurse -File | Measure-Object Length -Sum`
- 单位：MB 保留 1 位小数
- 脚本：`.trae/autoresearch/cleanup/scan_local_sizes.ps1`

### A.2 config.json 关键字段映射
- `dim` ← `model.base_dim`（默认 64）
- `depth` ← `model.num_res_blocks`（默认 4）
- `alpha` ← `model.style_extrap_alpha`（默认 0.1，630_local/4J 系列为 0.4）
- `w_ll` ← `bridge.spectral_w_ll`（默认 0.3）
- `gate_init` ← `model.style_cross_attn_gate_init`（默认 0.05）
- `gate_mode` ← `model.style_gate_mode`（tanh_gate / fixed_one）

### A.3 评估指标采集
- 数据源：各实验 `full_eval/<epoch>/summary.json`
- 字段映射：
  - `eval_duration_sec` ← `timings_sec.wall_total`
  - `clip_style` ← `analysis.all_pairs_overview.clip_style`
  - `lpips` ← `analysis.all_pairs_overview.content_lpips`
  - `clip_content` ← `analysis.all_pairs_overview.clip_content`
- 脚本：`.trae/autoresearch/cleanup/gen_local_timing.py`
- 输出：`.trae/autoresearch/cleanup/local_timing.csv` (122 行)

### A.4 训练时长采集
- 数据源：实验目录内 `train.log` / `run.log`
- 仅 5 个 630_local/4J 实验保留 train.log，训练时长约 3.0 min (5 ep, dim=64, depth=4)
- 其余 117 个实验训练日志未保留（多为远程训练或日志清理）
