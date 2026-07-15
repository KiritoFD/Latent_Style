# 远程 I 盘实验清单 (Remote Experiments on I Drive)

**远程服务器**: `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62` (Windows + WSL2 Ubuntu-22.04)
**根路径**: `/mnt/i/Github/Latent_Style/`
**整理日期**: 2026-07-02
**整理执行**: Deli_AutoResearch cleanup task (M1+M2 已完成)

---

## 0. 整理历史

### M1 清理（2026-07-02 16:35）
- 删除 86 个废弃目录
- 释放 10.02 GiB
- 删除类别：
  - 40 个 img2img_turbo smoke 测试（10.10G）
  - 28 个 SaMam/SaMST 失败 probe（4K-200K，均无 ckpt 无图片）
  - 14 个 aaai2027 invalid/空目录（无 ckpt）
  - 4 个 cyclegan/lbm smoke
- 完整日志：`.trae/autoresearch/cleanup/logs/m1_cleanup.log`

### M2 重组（2026-07-02 16:50）
- 移动 71 个目录到新分组结构
- 创建独立目录：
  - `exp_baselines/` — 12 baseline + SaMST 训练 + 元数据
  - `exp_samam/training/` — 14 个 SaMam 训练实验
  - `exp_ours/phase2/` — 23 个 aaai2027_phase2_* 实验
  - `exp_ours/recent/` — 620_spatial_bridge、inmortal-exp、highres 等
- `experiments/` → `experiments_historical/`（重命名，269 个历史实验保留）
- `exp/` 已清空删除
- 完整日志：`.trae/autoresearch/cleanup/logs/m2_reorg.log`

### 当前总磁盘占用
| 根目录 | 大小 | 说明 |
|--------|------|------|
| `exp_samam/training/` | 56G | 含 44G 主训练 |
| `exp_ours/` | 27G | phase2 + recent |
| `experiments_historical/` | ~9.0G | 269 个历史实验 |
| `exp_baselines/` | ~6G | 12 baseline + SaMST 训练 |
| `final_works/` | 31M | 7 个最终展示作品 |
| `Related_Works/runs/` | 4.9G | hf_snapshots CLIP cache |
| `Related_Works/repos/` | - | baseline 源码（不动） |

---

## 1. Baseline 评估实验 (`exp_baselines/`)

### 1.1 论文用 12 个 baseline（用于 AAAI 2027 对比）

数据源：`exp_baselines/`，全部用 `run_evaluation.py` 评估，HF transformers CLIP (ViT-B/32, openai/clip-vit-base-patch32)，LPIPS (Alex)，750 pairs。

| # | 目录 | 模型 | 类别 | mtime | 大小 | n_pairs | CLIP-S | LPIPS | Δ_idt | Finding ID |
|---|------|------|------|-------|------|---------|--------|-------|-------|------------|
| 1 | `identity/` | Identity | baseline | 2026-06-30 20:50 | - | 750 | 0.6933 | 0.0000 | 0.0000 | F001 |
| 2 | `adain/` | AdaIN | classical-inf | 2026-06-30 20:50 | - | 750 | 0.6679 | 0.7425 | -0.0254 | F002 |
| 3 | `wct_vgg19/` | WCT (VGG19) | classical-inf | 2026-07-01 17:20 | - | 750 | 0.7063 | 0.6348 | +0.0130 | F019 |
| 4 | `sdturbo/` | SD-Turbo | diffusion-inf | 2026-06-30 20:50 | - | 750 | 0.6933 | 0.0033 | 0.0000 | F007 |
| 5 | `sdedit_str_0p35/` | SDEdit s=0.35 | diffusion-sweep | 2026-06-30 20:50 | - | 750 | 0.7797 | 0.4508 | +0.0864 | F005 |
| 6 | `sdedit_str_0p40/` | SDEdit s=0.40 | diffusion-sweep | 2026-06-30 20:50 | - | 750 | 0.7934 | 0.4826 | +0.1001 | F006 |
| 7 | `styleid/` | StyleID | diffusion-inf | 2026-06-30 20:50 | 33M | 750 | 0.8223 | 0.5523 | +0.1290 | F008 |
| 8 | `cut/` | CUT | gan-train | 2026-06-30 20:50 | 7.0M | 745 | 0.7137 | 0.3743 | +0.0204 | F014 |
| 9 | `samst/` | SaMST | mamba-train | 2026-06-30 20:50 | 8.3M | 750 | 0.6183 | 0.7490 | -0.0750 | F011 |
| 10 | `seedream45_api/` | SeeDream | diffusion-inf | 2026-06-24 02:05 | 1006M | 750 | 0.7198 | 0.4767 | +0.0266 | F021 |
| 11 | `s2wat/` | S2WAT | other | 2026-06-24 02:05 | 3.5M | - | - | - | - | (待评估) |
| 12 | `sdedit_str_0p10/`、`sdedit_str_0p20/` | SDEdit s=0.10/0.20 | diffusion-sweep | 2026-06-24 02:05 | 3.4-3.5M | - | - | - | - | (按用户要求不再用于论文) |

**说明**：
- SaMam 不在此处，单独存于 `exp_samam/`（见第 2 节）
- Δ_idt = CLIP-S − 0.6933 (Identity 基线)
- 评估协议：`run_evaluation.py`，test_dir = `I:\wikiart_distinct5_samam_512_classview\test`
- 5 风格：Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e

### 1.2 SaMST 训练实验（与 baseline 一起存放）

| 目录 | mtime | 大小 | 训练时长 | ckpt | 图片 | 数据集 | 备注 |
|------|-------|------|---------|------|------|--------|------|
| `samst_distinct5_512_wsl_stepalign40_remote_20260605_r1/` | 2026-06-05 19:21 | 2.4G | 17.8min | 0 | 5007 | distinct5 | 主 SaMST 训练 |
| `samst_latent_distinct5_512_convergence_20260606_180529/` | 2026-06-06 18:40 | 624M | 42.6s | 0 | 1502 | distinct5 | 极短收敛 |
| `samst_latent_distinct5_512_convergence_20260606_214051/` | 2026-06-06 22:31 | 342M | 34.0min | 0 | 751 | distinct5 | |
| `samst_latent_distinct5_512_samecost_20260606_034941/` | 2026-06-06 03:50 | 896K | 9.7s | 0 | 5 | distinct5 | 失败 probe |
| `samst_latent_distinct5_512_samecost_20260606_041227/` | 2026-06-06 04:34 | 259M | 18.3min | 0 | 1502 | distinct5 | |
| `samst_latent_distinct5_512_samecost_20260606_145824/` | 2026-06-06 14:58 | 2.3M | 52.8min | 0 | 0 | distinct5 | 训练时长异常 |
| `samst_latent_distinct5_512_samecost_20260606_172021/` | 2026-06-06 17:27 | 537M | 58.3min | 0 | 1502 | distinct5 | |

**SaMST 训练总时长**: ~3.03h

### 1.3 辅助目录
- `_auxiliary_runs/` — cut_5x5、sdedit_multi、sdturbo_5x5、s2wat_bs1_safe_e2000_full_eval 等 5x5 评估
- `_metadata/` — 协议评估 CSV/JSON 汇总（protocol_a_800 等）

---

## 2. SaMam 训练实验 (`exp_samam/training/`)

### 2.1 ★ 主训练（最终用于论文对比）

| 目录 | mtime | 大小 | 训练时长 | ckpt | 图片 | 数据集 | 备注 |
|------|-------|------|---------|------|------|--------|------|
| `samam_distinct5_512_scratch_7k_250eval_remote/` | 2026-07-02 07:50 | **44G** | **3.87h** | 83 | 65928 | distinct5_512 | **★ 论文用主训练** |

**关键数据**（来自 progress.log）：
- 训练步数: 7000（resume 从 7K → 20K，但目录名仍叫 7k_250eval）
- 实际训练步数: 20000（每 250 步存 checkpoint，共 80 + last）
- 训练时长: 13948.61 秒 = 3.87 小时
- Checkpoints: 83 个 .ckpt 文件
- HF CLIP 评估: 81 个 checkpoint 完整评估完成（step 250-20000 + last）
- 最终值: CLIP-S=0.5816, LPIPS=0.2434 (step=20000)
- 评估耗时: Phase 1 推理 2.8h + Phase 2 评估 32.2min = 3.4h

### 2.2 历史 SaMam 训练（保留作参考）

| 目录 | mtime | 大小 | 训练时长 | ckpt | 图片 | 数据集 | 备注 |
|------|-------|------|---------|------|------|--------|------|
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/` | 2026-06-03 20:37 | 3.9G | 52.6min | 0 | 12003 | distinct5 | segmented.log |
| `samam_latent_distinct5_512_convergence_20260606_222608/` | 2026-06-07 00:13 | 834M | 1.43h | 0 | 2253 | distinct5 | |
| `samam_latent_distinct5_512_convergence_20260607_002420/` | 2026-06-07 01:01 | 626M | 27.3min | 0 | 1502 | distinct5 | |
| `samam_latent_distinct5_512_convergence_20260607_011328/` | 2026-06-07 01:34 | 318M | 27.5min | 0 | 751 | distinct5 | |
| `samam_latent_distinct5_512_samecost_20260606_133730/` | 2026-06-06 15:23 | 151M | 39.4min | 0 | 1502 | distinct5 | |
| `samam_latent_distinct5_512_samecost_20260606_155105/` | 2026-06-06 16:04 | 514M | 未记录 | 0 | 1502 | distinct5 | 中断无 END |
| `samam_latent_distinct5_512_samecost_20260606_162933/` | 2026-06-06 16:53 | 602M | 11.3min | 0 | 1502 | distinct5 | |
| `samam_latent_legacy256_probe4/` | 2026-06-06 02:11 | 183M | 1.88h | 0 | 1507 | legacy256_overfit50 | 旧 256 分辨率 |
| `samam_256_faithful_p8_remote/` | 2026-05-22 05:05 | 20M | 未记录 | 0 | 6 | 256 | 早期 256 实验 |
| `samam_distinct5_512_mamba_b6_20k_remote_wsl_20260601_1900/` | 2026-06-01 18:52 | 388K | 未记录 | 0 | 0 | distinct5 | 中断 |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_1910/` | 2026-06-01 19:06 | 428K | 未记录 | 0 | 0 | distinct5 | 中断 |
| `samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601_1935/` | 2026-06-01 19:22 | 664K | 未记录 | 0 | 0 | distinct5 | 中断 |
| `samam_distinct5_remote_wsl_batch_probe_20260601_1840/` | 2026-06-01 18:41 | 460K | 未记录 | 0 | 0 | distinct5 | 中断 |

**SaMam 有效训练总时长**: ~9.37h（8 个有效记录）

### 2.3 SaMam HF CLIP 评估曲线

完整 81 个 checkpoint 评估，本地路径：`tools/samam_distinct5_scratch/curve_metrics_hf.csv`

关键数据点：
| Step | CLIP-S | LPIPS | 备注 |
|------|--------|-------|------|
| 250 | 0.5208 | 0.8441 | 起始 |
| 3000 | 0.5868 | 0.3803 | CLIP-S 收敛 |
| 6500 | 0.5925 | 0.3518 | **Best CLIP-S** |
| 19500 | 0.5787 | 0.2223 | Best LPIPS |
| 20000 | 0.5816 | 0.2434 | **最终值** |

收敛曲线图：
- `tools/samam_distinct5_scratch/samam_hf_curve.png` — 三联图（CLIP-S / LPIPS / CLIP-content）
- `tools/samam_distinct5_scratch/samam_clip_style_vs_baselines.png` — 与所有 baseline 对比

---

## 3. 我们模型 - aaai2027 Phase 2 实验 (`exp_ours/phase2/`)

23 个 aaai2027_phase2_* 实验，全部 ours 模型，2026-06-13 至 2026-06-17 期间完成。

### 3.1 完整实验表

| # | 目录名（简写） | mtime | 大小 | 训练时长 | ckpt | 备注 |
|---|------|-------|------|---------|------|------|
| 1 | `i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 10:19 | 1.5G | 1.58h | 28 | train.log×5 |
| 2 | `i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 14:36 | 311M | 13.5min | 6 | eval_ts×6 |
| 3 | `i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-17 12:21 | 1.3G | 1.25h | 24 | train.log×4 |
| 4 | `i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 14:02 | 621M | 29.6min | 12 | eval_ts×12 |
| 5 | `i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 13:19 | 569M | 32.4min | 11 | train.log×2 |
| 6 | `i2sb_pattn_topo_anchor_sigma0p02_residual_warm_vel2_seed42_b22a1` | 2026-06-13 07:19 | 36M | 28.2min | 1 | train.log×1 |
| 7 | `i2sb_pattn_topo_anchor_sigma0p02_warm_vel2_seed42_b22a1` | 2026-06-13 06:49 | 36M | 30.1min | 1 | train.log×1 |
| 8 | `i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1` | 2026-06-13 06:25 | 36M | 22.7min | 1 | train.log×1 |
| 9 | `i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-17 11:02 | 1.3G | 2.23h | 24 | train.log×3 |
| 10 | `i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005_seed42_b20a1` | 2026-06-13 22:41 | 43M | 40.4min | 1 | train.log×2 |
| 11 | `i2sb_topo_anchor_sigma0p10_warm_vel2_seed42_b30a1` | 2026-06-13 05:56 | 36M | 21.3min | 1 | train.log×1 |
| 12 | `i2sb_topo_anchor_sigma0p25_seed42_b30a1` | 2026-06-13 05:29 | 36M | 26.5min | 1 | train.log×2 |
| 13 | `smoe_translator_k070_e3_seed42_b12a1` | 2026-06-14 13:18 | 1.1G | 49.9min | 15 | train.log×3 |
| 14 | `vel_pattn_enhanced_tok_seed42_b22a1` ⚠️ | 2026-06-14 03:17 | 351M | **25.15h** | 10 | 异常时长（跨天 gap） |
| 15 | `vel_pattn_enhanced_tok_seed42_b8a2` | 2026-06-13 01:22 | 2.5M | 12.9min | 0 | |
| 16 | `vel_pattn_topo_anchor_k075_seed42_b22a1` | 2026-06-13 05:04 | 71M | 45.9min | 2 | train.log×2 |
| 17 | `vel_tok32_pos_refresh_seed42_b20a1` | 2026-06-13 10:13 | 245M | 19.9min | 6 | train.log×2 |
| 18 | `vel_tok32_safe_rescan_r1_seed42_b20a1` | 2026-06-13 11:19 | 83M | 1.12h | 2 | train.log×3 |
| 19 | `vel_tok32_safe_rescan_r2_seed42_b20a1` | 2026-06-13 17:27 | 326M | 2.86h | 8 | eval_ts×8 |
| 20 | `vel_tok32_safe_semantic_topogate_k070_kin070_vlen010_seed42_b12a1` | 2026-06-14 14:16 | 124M | 21.5min | 3 | train.log×1 |
| 21 | `vel_tok32_safe_semantic_topogate_k070_seed42_b12a1` ⚠️ | 2026-06-14 01:23 | 206M | **5.0s** | 5 | 极短疑似失败 |
| 22 | `vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1` | 2026-06-13 22:07 | 165M | 2.00h | 4 | train.log×1 |
| 23 | `vel_tok32_safe_semantic_topogate_k085_seed42_b16a1` | 2026-06-13 19:27 | 164M | 1.88h | 4 | train.log×1 |

**Phase 2 训练总时长**: ~40.11h（含异常 25.15h）/ ~14.96h（剔除异常）

### 3.2 异常说明
- **#14 `vel_pattn_enhanced_tok_seed42_b22a1`**: 90527s (25.15h) 是 eval_ts 跨天 gap 导致数值虚高，实际训练时长远低于此
- **#21 `vel_tok32_safe_semantic_topogate_k070_seed42_b12a1`**: 仅 5 秒，启动即失败

### 3.3 命名约定
- `i2sb` = Image-to-Schrodinger Bridge（图像到薛定谔桥）
- `vel` = velocity formulation（速度形式）
- `pattn` = positional attention
- `topogate` = topological gate
- `tok32` = 32 tokens
- `k070` / `k085` = coupling coefficient 0.70 / 0.85
- `sigma0p02` = noise sigma 0.02
- `bXXaY` = batch XX, accumulate Y

---

## 4. 我们模型 - 近期实验 (`exp_ours/recent/`)

| # | 目录 | mtime | 大小 | 训练时长 | ckpt | 图片 | 备注 |
|---|------|-------|------|---------|------|------|------|
| 1 | `620_spatial_bridge/` | 2026-06-21 08:37 | 3.3G | 1.45h | 213 | 5 | 10 个 train.log 汇总 |
| 2 | `inmortal-exp/` | 2026-06-16 18:41 | **9.8G** | **61.92h** | 137 | 14 | round1+round2 子实验汇总（76/89 logs 有效） |
| 3 | `highres/` | 2026-05-20 13:43 | 5.5G | 41.22h | 0 | 5195 | 含 samst_pipeline+s2wat_pipeline 重叠运行 |
| 4 | `phase2_eval_rgbcal/` | 2026-06-14 14:46 | 564K | - | 0 | 0 | 纯 eval 目录，无 train.log |
| 5 | `ours_pareto_probe_4_epoch_0001/` | 2026-06-24 02:05 | 112K | - | 0 | 0 | 小 probe |
| 6 | `all620.json` | - | - | - | - | - | 620 系列汇总元数据 |
| 7 | `620_t5_base_multimodal_train.log` | - | - | - | - | - | T5 multimodal 训练日志 |

**近期实验训练总时长**: ~104.59h（含 inmortal-exp 61.92h 和 highres 41.22h 重叠）

### 4.1 inmortal-exp 详细结构
- 9.8G 大型实验，含 round1/round2 子目录
- 137 个 ckpt 文件
- 89 个 train.log，其中 76 个有完整 START/END 时间戳
- 训练总时长 61.92h（多次运行累计）

### 4.2 highres 详细结构
- 5.5G 高分辨率实验
- 5195 张图片
- 含 samst_pipeline 和 s2wat_pipeline 两个独立 log
- 41.22h 是两个 pipeline 时间跨度总和（可能重叠）

---

## 5. 历史实验 (`experiments_historical/`)

269 个历史 ours 实验（2026-02 至 2026-06），保留作参考。

### 5.1 按月份分布
| 月份 | 实验数 | 总大小 |
|------|--------|--------|
| 2026-06 | 7 | <1G |
| 2026-04 | 148 | 7.92G |
| 2026-03 | 107 | 1.30G |
| 2026-02 | 7 | <1G |

### 5.2 Top 10 最大历史实验
| 目录 | mtime | 大小 | 备注 |
|------|-------|------|------|
| `eval_cache/` | 2026-04-09 05:27 | 3.5G | 评估缓存 |
| `42/` | 2026-04-02 10:30 | 506M | |
| `freq/` | 2026-04-05 18:16 | 441M | |
| `Aline120/` | 2026-04-04 03:15 | 400M | |
| `src-no-hf/` | 2026-03-26 15:18 | 381M | |
| `Ablate43/` | 2026-04-03 12:09 | 380M | |
| `micro/` | 2026-04-02 19:08 | 207M | |
| `Color120/` | 2026-04-04 00:31 | 200M | |
| `45/` | 2026-04-06 00:11 | 158M | |
| `nce-gate_norm/` | 2026-04-09 05:27 | 144M | |

### 5.3 主要实验类别
- **42 系列**: 42, 42_A01-A10（架构消融）
- **Ablate 系列**: ablate_A0-A6, ablate_E1-E2, ablate_M1-M3（参数消融）
- **Decoder 系列**: decoder-A 到 decoder-H（解码器变体）
- **Master sweep 系列**: master_sweep_01-20（参数扫描）
- **arch 系列**: arch_1-8, arch_ablate_A1-D2, arch_ablate_E1-E2（架构搜索）
- **Inject 系列**: inject_I0-I7（注入机制实验）
- **Cross attn 系列**: cross_attn_Run_0-2, cross_attn_v3_v3_0-3（交叉注意力）
- **NCE 系列**: nce, nce-gate_*, nce-swd_*（噪声对比估计）

---

## 6. 最终展示作品 (`final_works/`)

7 个目录 + 4 个 CSV 文件，全部 mtime=2026-06-24 02:07。

| 名称 | 类型 | 大小 | 模型 |
|------|------|------|------|
| `CUT/` | DIR | 3.7M | CUT |
| `SaMST-epoch_0100/` | DIR | 3.5M | SaMST |
| `Star-GAN-epoch_100000/` | DIR | 3.6M | StarGAN |
| `str_0.40/` | DIR | 3.8M | SDEdit |
| `trial_0016/` | DIR | 5.5M | ours-final |
| `trial_0019/` | DIR | 5.7M | ours-final |
| `trial_0044/` | DIR | 5.1M | ours-final |
| `cross_attn_reeval.csv` | FILE | 12K | 交叉注意力重评估 |
| `final_works_metrics.csv` | FILE | 5.6K | 最终指标 |
| `final_works_metrics_overfit50_fair_partial.csv` | FILE | 4.9K | overfit50 公平部分评估 |
| `final_works_style_classifier_texture_e120_bs160.csv` | FILE | 3.0K | 风格分类器纹理 |

---

## 7. 数据集（I 盘根目录）

| 路径 | 用途 |
|------|------|
| `/mnt/i/wikiart_distinct5_samam_512_classview/` | **★ 主测试集**（5 风格，512×512，用于所有 baseline 评估） |
| `/mnt/i/wikiart_distinct5_samam_512_classview_real/` | 真实图像变体 |
| `/mnt/i/wikiart_distinct5_samam_512_flat/` | 扁平结构 |
| `/mnt/i/wikiart_distinct5_samam_512_latents_ema/` | EMA latent 编码 |
| `/mnt/i/wikiart_distinct5_latents_512_ema/` | EMA latent |
| `/mnt/i/wikiart_distinct5_latents_512_ema_test/` | EMA latent 测试集 |
| `/mnt/i/wikiart_faraday_splits/` | Faraday 分割 |
| `/mnt/i/wikiart_images_512_ema_test/` | 图像 EMA 测试集 |
| `/mnt/i/wikiart_latents_512_ema/` | EMA latent（全量） |
| `/mnt/i/wikiart_latents_512_ema_test/` | EMA latent 测试集 |
| `/mnt/i/wikiarts_5_full_notest/` | 5 风格全量无测试集 |
| `/mnt/i/wikiarts_5_full_notest_latents_ema/` | 上述的 EMA latent |

---

## 8. 相关参考文档

- `tools/samam_distinct5_scratch/all_related_works_summary.md` — 12 baseline 完整数据汇总（含 SaMam 81 ckpt 收敛曲线）
- `tools/samam_distinct5_scratch/curve_metrics_hf.csv` — SaMam HF CLIP 81 checkpoint 完整数据
- `tools/samam_distinct5_scratch/samam_hf_curve.png` — 收敛曲线图
- `.trae/autoresearch/cleanup/logs/m1_cleanup.log` — M1 删除日志
- `.trae/autoresearch/cleanup/logs/m2_reorg.log` — M2 重组日志
- `scan_report.md` — 整理前完整扫描报告（470 行）

---

## 9. 已知问题与待办

1. **CUT 训练时长未记录**：用户记忆 322.6min，但远程目录无 train.log，仅 summary.json（eval 指标）
2. **SaMam 6 个中断实验无时长**：均为 b6/b8 系列的 20K 训练中断，无 END 时间戳
3. **phase2 #14 时长异常**：vel_pattn_enhanced_tok_b22a1 的 25.15h 是 eval_ts 跨天导致虚高
4. **phase2 #21 极短**：vel_tok32_safe_semantic_topogate_k070_b12a1 仅 5 秒，启动即失败
5. **highres 时长可能高估**：含 samst_pipeline 和 s2wat_pipeline 重叠运行
6. **experiments_historical/ 的 269 个实验**：未来如需进一步清理，可按月份/类别批量归档
