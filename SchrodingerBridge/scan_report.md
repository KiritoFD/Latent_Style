# 远程实验数据整理探查报告

**扫描时间**: 2026-07-02
**远程服务器**: 100.115.18.62 (Windows + WSL2)
**根路径**: /mnt/i/Github/Latent_Style/
**总实验数**: 430

## 总磁盘占用统计

| 根目录 | 总大小 |
|--------|--------|
| `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results` | 56G |
| `/mnt/i/Github/Latent_Style/Related_Works/runs` | 16G |
| `/mnt/i/Github/Latent_Style/exp` | 27G |
| `/mnt/i/Github/Latent_Style/experiments` | ~9.0G (按子目录逐项求和) |
| `/mnt/i/Github/Latent_Style/final_works` | 31M |
| **已知合计** | **~108G** |
| (其中 samam_distinct5_512_scratch_7k 占 44G) | - |
| (其中 inmortal-exp 占 9.8G) | - |
| (其中 exp/highres 占 5.5G) | - |
| (其中 experiments/eval_cache 占 3.5G) | - |

## A. Baseline评估目录 (Related_Works/baseline_pipeline/results + runs)

### A.1 baseline_pipeline/results/ (62 个实验)

| 目录名 | mtime | 大小 | 模型 | 数据集 | 训练时长 | ckpt | img | metrics.csv | 备注 |
|--------|-------|------|------|--------|----------|------|-----|-------------|------|
| `samam_distinct5_512_scratch_7k_250eval_remote` | 2026-07-02 07:50 | 44G | SaMam | distinct5 | 3.87h | 83 | 0 | ✓ | **★Tier1重点** 7000步 |
| `cut` | 2026-06-24 02:05 | 7.0M | CUT | - | - | 0 | 2 | ✓ |  |
| `ours_pareto_probe_4_epoch_0001` | 2026-06-24 02:05 | 112K | other | - | - | 0 | 0 | ✓ |  |
| `s2wat` | 2026-06-24 02:05 | 3.5M | S2WAT | - | - | 0 | 2 | ✓ |  |
| `samst` | 2026-06-24 02:05 | 8.3M | SaMST | - | - | 0 | 6 | ✓ |  |
| `sdedit_str_0p10` | 2026-06-24 02:05 | 3.4M | SDEdit | - | - | 0 | 1 | ✓ |  |
| `sdedit_str_0p20` | 2026-06-24 02:05 | 3.5M | SDEdit | - | - | 0 | 1 | ✓ |  |
| `sdedit_str_0p35` | 2026-06-24 02:05 | 3.6M | SDEdit | - | - | 0 | 1 | ✓ |  |
| `sdedit_str_0p40` | 2026-06-24 02:05 | 3.7M | SDEdit | - | - | 0 | 1 | ✓ |  |
| `sdturbo` | 2026-06-24 02:05 | 3.1M | SDTurbo | - | - | 0 | 1 | ✓ |  |
| `seedream45_api` | 2026-06-24 02:05 | 1006M | other | - | - | 0 | 750 | ✓ |  |
| `styleid` | 2026-06-24 02:05 | 33M | StyleID | - | - | 0 | 7 | ✓ |  |
| `samam_latent_distinct5_512_samecost_20260606_162544` | 2026-06-07 16:26 | 24K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_512_samecost_20260606_133519` | 2026-06-07 13:47 | 56K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_035344` | 2026-06-07 03:54 | 8.0K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_035136` | 2026-06-07 03:51 | 12K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_512_convergence_20260607_011328` | 2026-06-07 01:34 | 318M | SaMam | distinct5 | - | 0 | 1 | ✓ |  |
| `samam_latent_distinct5_512_convergence_20260607_002420` | 2026-06-07 01:01 | 626M | SaMam | distinct5 | - | 0 | 2 | ✓ |  |
| `samam_latent_distinct5_512_convergence_20260606_222608` | 2026-06-07 00:13 | 834M | SaMam | distinct5 | - | 0 | 3 | ✓ |  |
| `samst_latent_distinct5_512_convergence_20260606_214051` | 2026-06-06 22:31 | 342M | SaMST | distinct5 | - | 0 | 1 | ✓ |  |
| `samst_latent_distinct5_512_convergence_20260606_180529` | 2026-06-06 18:40 | 624M | SaMST | distinct5 | - | 0 | 2 | ✓ |  |
| `samst_latent_distinct5_512_samecost_20260606_172021` | 2026-06-06 17:27 | 537M | SaMST | distinct5 | - | 0 | 2 | ✓ |  |
| `samam_latent_distinct5_512_samecost_20260606_162933` | 2026-06-06 16:53 | 602M | SaMam | distinct5 | - | 0 | 2 | ✓ |  |
| `samam_latent_distinct5_512_samecost_20260606_155105` | 2026-06-06 16:04 | 514M | SaMam | distinct5 | - | 0 | 2 | ✓ |  |
| `samam_latent_distinct5_512_samecost_20260606_133730` | 2026-06-06 15:23 | 151M | SaMam | distinct5 | - | 0 | 2 | ✓ |  |
| `samst_latent_distinct5_512_samecost_20260606_145824` | 2026-06-06 14:58 | 2.3M | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_041227` | 2026-06-06 04:34 | 259M | SaMST | distinct5 | - | 0 | 2 | ✓ |  |
| `samst_latent_distinct5_512_samecost_20260606_040419` | 2026-06-06 04:04 | 4.0K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_035731` | 2026-06-06 03:57 | 4.0K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_035540` | 2026-06-06 03:56 | 8.0K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samst_latent_distinct5_512_samecost_20260606_034941` | 2026-06-06 03:50 | 896K | SaMST | distinct5 | - | 0 | 5 |  |  |
| `samam_latent_distinct5_512_samecost_20260606_034632` | 2026-06-06 03:47 | 20K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_512_samecost_20260606_034359` | 2026-06-06 03:44 | 52K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_512_samecost_20260606_034059` | 2026-06-06 03:41 | 28K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_samecost_probe3` | 2026-06-06 02:37 | 16K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_samecost_probe2` | 2026-06-06 02:34 | 8.0K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_distinct5_samecost_probe1` | 2026-06-06 02:32 | 8.0K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_latent_legacy256_probe4` | 2026-06-06 02:11 | 183M | SaMam | legacy256 | - | 0 | 2 | ✓ |  |
| `samst_latent_legacy256_probe3` | 2026-06-05 23:44 | 120K | SaMST | legacy256 | - | 0 | 5 |  |  |
| `samst_latent_legacy256_probe1` | 2026-06-05 23:42 | 116K | SaMST | legacy256 | - | 0 | 5 |  |  |
| `samst_latent_legacy256_probe2` | 2026-06-05 23:42 | 120K | SaMST | legacy256 | - | 0 | 5 |  |  |
| `samam_latent_legacy256_probe3` | 2026-06-05 23:34 | 24K | SaMam | legacy256 | - | 0 | 0 |  |  |
| `samam_latent_legacy256_probe2` | 2026-06-05 23:29 | 48K | SaMam | legacy256 | - | 0 | 0 |  |  |
| `samam_latent_legacy256_probe` | 2026-06-05 23:22 | 20K | SaMam | legacy256 | - | 0 | 0 |  |  |
| `samam_latent_legacy256_remote` | 2026-06-05 23:05 | 28K | SaMam | legacy256 | - | 0 | 0 |  |  |
| `samst_distinct5_512_wsl_stepalign40_remote_20260605_r1` | 2026-06-05 19:21 | 2.4G | SaMST | distinct5 | - | 0 | 5005 |  |  |
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag` | 2026-06-03 20:37 | 3.9G | SaMam | distinct5 | - | 0 | 3 | ✓ |  |
| `samst_distinct5_512_prepared` | 2026-06-01 21:43 | 8.0K | SaMST | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130` | 2026-06-01 19:52 | 16K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601_1935` | 2026-06-01 19:22 | 664K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_1910` | 2026-06-01 19:06 | 428K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b6_20k_remote_wsl_20260601_1900` | 2026-06-01 18:52 | 388K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_1850` | 2026-06-01 18:46 | 16K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_remote_wsl_batch_probe_20260601_1840` | 2026-06-01 18:41 | 460K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_1830b` | 2026-06-01 18:27 | 12K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_1830` | 2026-06-01 18:26 | 8.0K | SaMam | distinct5 | - | 0 | 0 |  |  |
| `flux2_klein` | 2026-05-23 15:25 | 4.0K | Flux2 | - | - | 0 | 0 |  |  |
| `zimage_turbo` | 2026-05-23 15:03 | 0 | ZImageTurbo | - | - | 0 | 0 |  |  |
| `samam_256_faithful_p8_remote` | 2026-05-22 05:05 | 20M | SaMam | 256 | - | 0 | 0 |  |  |
| `samam_256_faithful_p8_remote_debug_foreground` | 2026-05-22 05:00 | 104K | SaMam | 256 | - | 0 | 0 |  |  |
| `samam_256_faithful_p8_remote_debug_idckpt` | 2026-05-22 04:45 | 108K | SaMam | 256 | - | 0 | 0 |  |  |
| `samam_256_faithful_p8_remote_debug` | 2026-05-22 04:43 | 12K | SaMam | 256 | - | 0 | 0 |  |  |

### A.2 Related_Works/runs/ (51 个实验)

| 目录名 | mtime | 大小 | 模型 | 备注 |
|--------|-------|------|------|------|
| `hf_snapshots` | 2026-06-30 08:26 | 4.9G | other |  |
| `img2img_turbo_distinct5_remote_smoke_20260606_202920` | 2026-06-06 20:29 | 1014M | Img2ImgTurbo | smoke测试 |
| `cut_5x5` | 2026-06-24 02:05 | 567M | CUT |  |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_050156` | 2026-06-07 05:02 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_050239` | 2026-06-07 05:03 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_050448` | 2026-06-07 05:05 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_050640` | 2026-06-07 05:07 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_051058` | 2026-06-07 05:11 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_051309` | 2026-06-07 05:13 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_051439` | 2026-06-07 05:15 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_052126` | 2026-06-07 05:22 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_052546` | 2026-06-07 05:26 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_055754` | 2026-06-07 05:58 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_060315` | 2026-06-07 06:04 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_060823` | 2026-06-07 06:09 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_061146` | 2026-06-07 06:13 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_061608` | 2026-06-07 06:16 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_062547` | 2026-06-07 06:26 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_185423` | 2026-06-07 18:55 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_190514` | 2026-06-07 19:06 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_192636` | 2026-06-07 19:27 | 460M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_20260606_201322` | 2026-06-06 20:13 | 192M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_052126` | 2026-06-06 05:21 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_052546` | 2026-06-06 05:26 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_055754` | 2026-06-06 05:58 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_060315` | 2026-06-06 06:03 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_060823` | 2026-06-06 06:08 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_061146` | 2026-06-06 06:12 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_061608` | 2026-06-06 06:16 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_062547` | 2026-06-06 06:26 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_190514` | 2026-06-06 19:05 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_192636` | 2026-06-06 19:26 | 75M | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_201322` | 2026-06-07 20:14 | 54M | Img2ImgTurbo | smoke测试重复 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_202920` | 2026-06-07 20:30 | 54M | Img2ImgTurbo | smoke测试重复 |
| `sdedit_multi` | 2026-06-24 02:06 | 13M | SDEdit |  |
| `sdturbo_5x5` | 2026-06-24 02:06 | 3.0M | SDTurbo |  |
| `s2wat_bs1_safe_e2000_full_eval` | 2026-06-24 02:06 | 2.6M | S2WAT |  |
| `img2img_turbo_distinct5_remote_smoke_20260606_185423` | 2026-06-06 18:54 | 44K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050156` | 2026-06-06 05:02 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050239` | 2026-06-06 05:02 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050448` | 2026-06-06 05:05 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050640` | 2026-06-06 05:06 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_051058` | 2026-06-06 05:11 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_051309` | 2026-06-06 05:13 | 24K | Img2ImgTurbo | smoke测试 |
| `img2img_turbo_distinct5_remote_smoke_20260606_051439` | 2026-06-06 05:14 | 24K | Img2ImgTurbo | smoke测试 |
| `benchmark_logs` | 2026-06-24 02:05 | 20K | other |  |
| `cyclegan_5x5_smoke` | 2026-06-24 02:06 | 16K | CycleGAN |  |
| `cyclegan_5x5` | 2026-06-24 02:06 | 4.0K | CycleGAN |  |
| `lbm_train_smoke_run` | 2026-06-24 02:06 | 4.0K | LBM |  |
| `server_new_baselines` | 2026-06-24 02:06 | 4.0K | other |  |
| `lbm_train_wds_smoke_photo_to_monet` | 2026-06-24 02:06 | 0 | LBM |  |

## B. SaMam训练目录 (所有samam_*)

共 33 个SaMam实验

| 目录名 | 路径 | mtime | 大小 | 数据集 | ckpt | img | 备注 |
|--------|------|-------|------|--------|------|-----|------|
| `samam_distinct5_512_scratch_7k_250eval_remote` | results/ | 2026-07-02 07:50 | 44G | distinct5 | 83 | 0 | **★Tier1重点训练** 7000步/3.87h |
| `samam_latent_distinct5_512_samecost_20260606_162544` | results/ | 2026-06-07 16:26 | 24K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_512_samecost_20260606_133519` | results/ | 2026-06-07 13:47 | 56K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_512_convergence_20260607_011328` | results/ | 2026-06-07 01:34 | 318M | distinct5 | 0 | 1 |  |
| `samam_latent_distinct5_512_convergence_20260607_002420` | results/ | 2026-06-07 01:01 | 626M | distinct5 | 0 | 2 |  |
| `samam_latent_distinct5_512_convergence_20260606_222608` | results/ | 2026-06-07 00:13 | 834M | distinct5 | 0 | 3 |  |
| `samam_latent_distinct5_512_samecost_20260606_162933` | results/ | 2026-06-06 16:53 | 602M | distinct5 | 0 | 2 |  |
| `samam_latent_distinct5_512_samecost_20260606_155105` | results/ | 2026-06-06 16:04 | 514M | distinct5 | 0 | 2 |  |
| `samam_latent_distinct5_512_samecost_20260606_133730` | results/ | 2026-06-06 15:23 | 151M | distinct5 | 0 | 2 |  |
| `samam_latent_distinct5_512_samecost_20260606_034632` | results/ | 2026-06-06 03:47 | 20K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_512_samecost_20260606_034359` | results/ | 2026-06-06 03:44 | 52K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_512_samecost_20260606_034059` | results/ | 2026-06-06 03:41 | 28K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_samecost_probe3` | results/ | 2026-06-06 02:37 | 16K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_samecost_probe2` | results/ | 2026-06-06 02:34 | 8.0K | distinct5 | 0 | 0 |  |
| `samam_latent_distinct5_samecost_probe1` | results/ | 2026-06-06 02:32 | 8.0K | distinct5 | 0 | 0 |  |
| `samam_latent_legacy256_probe4` | results/ | 2026-06-06 02:11 | 183M | legacy256 | 0 | 2 |  |
| `samam_latent_legacy256_probe3` | results/ | 2026-06-05 23:34 | 24K | legacy256 | 0 | 0 |  |
| `samam_latent_legacy256_probe2` | results/ | 2026-06-05 23:29 | 48K | legacy256 | 0 | 0 |  |
| `samam_latent_legacy256_probe` | results/ | 2026-06-05 23:22 | 20K | legacy256 | 0 | 0 |  |
| `samam_latent_legacy256_remote` | results/ | 2026-06-05 23:05 | 28K | legacy256 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601` | results/ | 2026-06-03 20:37 | 3.9G | distinct5 | 0 | 3 | 大型训练 |
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601` | results/ | 2026-06-01 19:52 | 16K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601` | results/ | 2026-06-01 19:22 | 664K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_19` | results/ | 2026-06-01 19:06 | 428K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b6_20k_remote_wsl_20260601_19` | results/ | 2026-06-01 18:52 | 388K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_18` | results/ | 2026-06-01 18:46 | 16K | distinct5 | 0 | 0 |  |
| `samam_distinct5_remote_wsl_batch_probe_20260601_1840` | results/ | 2026-06-01 18:41 | 460K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_18` | results/ | 2026-06-01 18:27 | 12K | distinct5 | 0 | 0 |  |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_18` | results/ | 2026-06-01 18:26 | 8.0K | distinct5 | 0 | 0 |  |
| `samam_256_faithful_p8_remote` | results/ | 2026-05-22 05:05 | 20M | 256 | 0 | 0 |  |
| `samam_256_faithful_p8_remote_debug_foreground` | results/ | 2026-05-22 05:00 | 104K | 256 | 0 | 0 |  |
| `samam_256_faithful_p8_remote_debug_idckpt` | results/ | 2026-05-22 04:45 | 108K | 256 | 0 | 0 |  |
| `samam_256_faithful_p8_remote_debug` | results/ | 2026-05-22 04:43 | 12K | 256 | 0 | 0 |  |

## C. 我们模型 - aaai2027系列 (exp/aaai2027_* + exp/inmortal-exp + exp/phase2_eval_rgbcal)

共 41 个实验

| 目录名 | mtime | 大小 | ckpt | img | 备注 |
|--------|-------|------|------|-----|------|
| `620_spatial_bridge` | 2026-06-21 08:37 | 3.3G | 213 | 1 |  |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-17 12:21 | 1.3G | 24 | 0 |  |
| `aaai2027_phase2_i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen01` | 2026-06-17 11:02 | 1.3G | 24 | 0 |  |
| `inmortal-exp` | 2026-06-16 18:41 | 9.8G | 137 | 0 | ★含round1/round2大量子实验+训练日志 |
| `aaai2027_eval_fiber_project_slerpe2_sigma0p0` | 2026-06-16 15:08 | 80K | 0 | 0 | 空/失败 |
| `aaai2027_eval_fiber_project_slerpe2_sigma0p5` | 2026-06-16 15:08 | 80K | 0 | 0 | 空/失败 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma0p0` | 2026-06-16 14:56 | 80K | 0 | 0 | 空/失败 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma1p0` | 2026-06-16 14:52 | 80K | 0 | 0 | 空/失败 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma1p5` | 2026-06-16 14:52 | 80K | 0 | 0 | 空/失败 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma0p5` | 2026-06-16 14:51 | 108K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_i2sb_orthogonal_chmeanlow_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 14:36 | 311M | 6 | 0 |  |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 14:02 | 621M | 12 | 0 |  |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 13:19 | 569M | 11 | 0 |  |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 11:54 | 940K | 0 | 0 | **失效** 空/失败 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 11:48 | 908K | 0 | 0 | **失效** 空/失败 |
| `aaai2027_phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 10:19 | 1.5G | 28 | 0 |  |
| `phase2_eval_rgbcal` | 2026-06-14 14:46 | 564K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_kin070_vlen010_seed42` | 2026-06-14 14:16 | 124M | 3 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_kin070_seed42_b12a1` | 2026-06-14 13:44 | 820K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1` | 2026-06-14 13:18 | 1.1G | 15 | 0 |  |
| `aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1` | 2026-06-14 03:17 | 351M | 10 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1` | 2026-06-14 01:23 | 206M | 5 | 0 |  |
| `aaai2027_phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor` | 2026-06-13 22:41 | 43M | 1 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1` | 2026-06-13 22:07 | 165M | 4 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1` | 2026-06-13 19:48 | 984K | 0 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1` | 2026-06-13 19:27 | 164M | 4 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1` | 2026-06-13 17:33 | 668K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1` | 2026-06-13 17:27 | 326M | 8 | 0 |  |
| `aaai2027_phase2_vel_tok32_safe_rescan_r1_seed42_b20a1` | 2026-06-13 11:19 | 83M | 2 | 0 |  |
| `aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1` | 2026-06-13 10:13 | 245M | 6 | 0 |  |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p02_residual_warm_vel2_seed42_` | 2026-06-13 07:19 | 36M | 1 | 0 |  |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p02_warm_vel2_seed42_b22a1` | 2026-06-13 06:49 | 36M | 1 | 0 |  |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1` | 2026-06-13 06:25 | 36M | 1 | 0 |  |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b24a1` | 2026-06-13 06:05 | 664K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b26a1` | 2026-06-13 06:03 | 664K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_i2sb_topo_anchor_sigma0p10_warm_vel2_seed42_b30a1` | 2026-06-13 05:56 | 36M | 1 | 0 |  |
| `aaai2027_phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1` | 2026-06-13 05:29 | 36M | 1 | 0 |  |
| `aaai2027_phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1` | 2026-06-13 05:10 | 744K | 0 | 0 | 空/失败 |
| `aaai2027_phase2_vel_pattn_topo_anchor_k075_seed42_b22a1` | 2026-06-13 05:04 | 71M | 2 | 0 |  |
| `aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b8a2` | 2026-06-13 01:22 | 2.5M | 0 | 0 |  |
| `highres` | 2026-05-20 13:43 | 5.5G | 0 | 0 |  |

## D. 我们模型 - 历史实验 (experiments/)

共 269 个实验（其中 232 个有内容，37 个为空目录或4K占位）

### D.1 Top 30 最大历史实验

| 目录名 | mtime | 大小 | summary | ckpt | img |
|--------|-------|------|---------|------|-----|
| `eval_cache` | 2026-04-09 05:27 | 3.5G | 0 | 0 | 0 |
| `42` | 2026-04-02 10:30 | 506M | 0 | 0 | 0 |
| `freq` | 2026-04-05 18:16 | 441M | 0 | 0 | 0 |
| `Aline120` | 2026-04-04 03:15 | 400M | 0 | 0 | 0 |
| `src-no-hf` | 2026-03-26 15:18 | 381M | 0 | 0 | 0 |
| `Ablate43` | 2026-04-03 12:09 | 380M | 0 | 0 | 0 |
| `micro` | 2026-04-02 19:08 | 207M | 0 | 0 | 0 |
| `Color120` | 2026-04-04 00:31 | 200M | 0 | 0 | 0 |
| `45` | 2026-04-06 00:11 | 158M | 0 | 0 | 0 |
| `nce-gate_norm` | 2026-04-09 05:27 | 144M | 0 | 0 | 0 |
| `FinalMicro_2` | 2026-04-02 21:21 | 88M | 0 | 0 | 0 |
| `nce` | 2026-04-09 05:27 | 59M | 0 | 0 | 0 |
| `Aline120_aline_03_ghost_wireframe` | 2026-04-09 05:27 | 58M | 8 | 0 | 4 |
| `nstyle-proj` | 2026-04-09 05:27 | 54M | 0 | 0 | 0 |
| `ablate_M1-Aggressive-Fine` | 2026-03-08 01:23 | 50M | 6 | 0 | 3 |
| `ablate_M2-Smooth-Impasto` | 2026-03-08 01:23 | 49M | 6 | 0 | 3 |
| `G0_Balanced_Base` | 2026-04-09 05:27 | 45M | 6 | 0 | 3 |
| `G1_High_HF_Test` | 2026-04-09 05:27 | 45M | 6 | 0 | 3 |
| `decoder-H-MSCTM-no_clamp_mult-tv-2` | 2026-04-09 05:27 | 42M | 6 | 0 | 3 |
| `nce-gate_norm-swd_0.45-cl_0.01` | 2026-04-09 05:27 | 38M | 0 | 0 | 0 |
| `nce-swd_0.25-cl_0.01` | 2026-04-09 05:27 | 37M | 0 | 0 | 0 |
| `final_demodulation` | 2026-03-08 01:23 | 36M | 4 | 0 | 2 |
| `decoder-1` | 2026-03-08 01:23 | 33M | 4 | 0 | 2 |
| `decoder-D-sweetspot` | 2026-03-08 01:23 | 33M | 4 | 0 | 2 |
| `decoder-H-MSCTM` | 2026-04-09 05:27 | 33M | 4 | 0 | 2 |
| `G0-Base-Gain0.5` | 2026-03-08 01:23 | 32M | 4 | 0 | 2 |
| `decoder-B-hf-strict-id` | 2026-03-08 01:23 | 32M | 4 | 0 | 2 |
| `decoder-E-extreme-brush` | 2026-03-08 01:23 | 32M | 4 | 0 | 2 |
| `cross_attn_v3_v3_2_skip_naive` | 2026-04-09 05:27 | 31M | 3 | 0 | 0 |
| `decoder-C-relaxed-id-nohf` | 2026-03-08 01:23 | 31M | 4 | 0 | 2 |

### D.2 历史实验按mtime分组

| 月份 | 实验数 | 总大小(估算) |
|------|--------|-------------|
| 2026-06 | 7 | 0.00G |
| 2026-04 | 148 | 7.92G |
| 2026-03 | 107 | 1.30G |
| 2026-02 | 7 | 0.00G |

## E. 其他重要目录

### E.1 final_works/ (最终展示用)

| 目录名 | mtime | 大小 | 模型 |
|--------|-------|------|------|
| `CUT` | 2026-06-24 02:07 | 3.7M | CUT |
| `SaMST-epoch_0100` | 2026-06-24 02:07 | 3.5M | SaMST |
| `Star-GAN-epoch_100000` | 2026-06-24 02:07 | 3.6M | StarGAN |
| `str_0.40` | 2026-06-24 02:07 | 3.8M | SDEdit |
| `trial_0016` | 2026-06-24 02:07 | 5.5M | ours-final |
| `trial_0019` | 2026-06-24 02:07 | 5.7M | ours-final |
| `trial_0044` | 2026-06-24 02:07 | 5.1M | ours-final |

## 决策清单

### 可删除候选清单

#### 1. 显式smoke/probe/debug测试 (重复且无价值)

共 40 个小型smoke/probe/debug目录

| 目录名 | 路径 | 大小 | mtime |
|--------|------|------|-------|
| `img2img_turbo_distinct5_remote_smoke_20260606_052126` | runs/ | 75M | 2026-06-06 05:21 |
| `img2img_turbo_distinct5_remote_smoke_20260606_052546` | runs/ | 75M | 2026-06-06 05:26 |
| `img2img_turbo_distinct5_remote_smoke_20260606_055754` | runs/ | 75M | 2026-06-06 05:58 |
| `img2img_turbo_distinct5_remote_smoke_20260606_060315` | runs/ | 75M | 2026-06-06 06:03 |
| `img2img_turbo_distinct5_remote_smoke_20260606_060823` | runs/ | 75M | 2026-06-06 06:08 |
| `img2img_turbo_distinct5_remote_smoke_20260606_061146` | runs/ | 75M | 2026-06-06 06:12 |
| `img2img_turbo_distinct5_remote_smoke_20260606_061608` | runs/ | 75M | 2026-06-06 06:16 |
| `img2img_turbo_distinct5_remote_smoke_20260606_062547` | runs/ | 75M | 2026-06-06 06:26 |
| `img2img_turbo_distinct5_remote_smoke_20260606_190514` | runs/ | 75M | 2026-06-06 19:05 |
| `img2img_turbo_distinct5_remote_smoke_20260606_192636` | runs/ | 75M | 2026-06-06 19:26 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_20132` | runs/ | 54M | 2026-06-07 20:14 |
| `img2img_turbo_distinct5_remote_smoke_datasets_20260606_20292` | runs/ | 54M | 2026-06-07 20:30 |
| `samam_distinct5_remote_wsl_batch_probe_20260601_1840` | results/ | 460K | 2026-06-01 18:41 |
| `SOTA_Probe_Exp00_Baseline` | experiments/ | 180K | 2026-03-26 19:01 |
| `perf_probe_A1_HF_Soft` | experiments/ | 176K | 2026-03-26 18:48 |
| `samst_latent_legacy256_probe2` | results/ | 120K | 2026-06-05 23:42 |
| `samst_latent_legacy256_probe3` | results/ | 120K | 2026-06-05 23:44 |
| `samst_latent_legacy256_probe1` | results/ | 116K | 2026-06-05 23:42 |
| `ours_pareto_probe_4_epoch_0001` | results/ | 112K | 2026-06-24 02:05 |
| `samam_256_faithful_p8_remote_debug_idckpt` | results/ | 108K | 2026-05-22 04:45 |
| `samam_256_faithful_p8_remote_debug_foreground` | results/ | 104K | 2026-05-22 05:00 |
| `samam_latent_legacy256_probe2` | results/ | 48K | 2026-06-05 23:29 |
| `img2img_turbo_distinct5_remote_smoke_20260606_185423` | runs/ | 44K | 2026-06-06 18:54 |
| `samam_latent_legacy256_probe3` | results/ | 24K | 2026-06-05 23:34 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050156` | runs/ | 24K | 2026-06-06 05:02 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050239` | runs/ | 24K | 2026-06-06 05:02 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050448` | runs/ | 24K | 2026-06-06 05:05 |
| `img2img_turbo_distinct5_remote_smoke_20260606_050640` | runs/ | 24K | 2026-06-06 05:06 |
| `img2img_turbo_distinct5_remote_smoke_20260606_051058` | runs/ | 24K | 2026-06-06 05:11 |
| `img2img_turbo_distinct5_remote_smoke_20260606_051309` | runs/ | 24K | 2026-06-06 05:13 |
| ... 还有 10 个 | | | |

#### 2. img2img_turbo重复smoke测试 (Related_Works/runs/)

共 40 个，总大小 10.10G。其中 `_datasets_` 后缀的是带数据集副本，可考虑删除数据集副本

#### 3. SaMam/SaMST 失败probe (4K-50K空目录)

共 28 个小型SaMam/SaMST目录（<200K，多为失败probe）

| 目录名 | 路径 | 大小 | mtime |
|--------|------|------|-------|
| `samam_latent_distinct5_512_samecost_20260606_162544` | results/ | 24K | 2026-06-07 16:26 |
| `samam_latent_distinct5_512_samecost_20260606_133519` | results/ | 56K | 2026-06-07 13:47 |
| `samst_latent_distinct5_512_samecost_20260606_035344` | results/ | 8.0K | 2026-06-07 03:54 |
| `samst_latent_distinct5_512_samecost_20260606_035136` | results/ | 12K | 2026-06-07 03:51 |
| `samst_latent_distinct5_512_samecost_20260606_040419` | results/ | 4.0K | 2026-06-06 04:04 |
| `samst_latent_distinct5_512_samecost_20260606_035731` | results/ | 4.0K | 2026-06-06 03:57 |
| `samst_latent_distinct5_512_samecost_20260606_035540` | results/ | 8.0K | 2026-06-06 03:56 |
| `samam_latent_distinct5_512_samecost_20260606_034632` | results/ | 20K | 2026-06-06 03:47 |
| `samam_latent_distinct5_512_samecost_20260606_034359` | results/ | 52K | 2026-06-06 03:44 |
| `samam_latent_distinct5_512_samecost_20260606_034059` | results/ | 28K | 2026-06-06 03:41 |
| `samam_latent_distinct5_samecost_probe3` | results/ | 16K | 2026-06-06 02:37 |
| `samam_latent_distinct5_samecost_probe2` | results/ | 8.0K | 2026-06-06 02:34 |
| `samam_latent_distinct5_samecost_probe1` | results/ | 8.0K | 2026-06-06 02:32 |
| `samst_latent_legacy256_probe3` | results/ | 120K | 2026-06-05 23:44 |
| `samst_latent_legacy256_probe1` | results/ | 116K | 2026-06-05 23:42 |
| `samst_latent_legacy256_probe2` | results/ | 120K | 2026-06-05 23:42 |
| `samam_latent_legacy256_probe3` | results/ | 24K | 2026-06-05 23:34 |
| `samam_latent_legacy256_probe2` | results/ | 48K | 2026-06-05 23:29 |
| `samam_latent_legacy256_probe` | results/ | 20K | 2026-06-05 23:22 |
| `samam_latent_legacy256_remote` | results/ | 28K | 2026-06-05 23:05 |
| `samst_distinct5_512_prepared` | results/ | 8.0K | 2026-06-01 21:43 |
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130` | results/ | 16K | 2026-06-01 19:52 |
| `samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_1850` | results/ | 16K | 2026-06-01 18:46 |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_1830b` | results/ | 12K | 2026-06-01 18:27 |
| `samam_distinct5_512_mamba_b1_20k_remote_wsl_20260601_1830` | results/ | 8.0K | 2026-06-01 18:26 |
| `samam_256_faithful_p8_remote_debug_foreground` | results/ | 104K | 2026-05-22 05:00 |
| `samam_256_faithful_p8_remote_debug_idckpt` | results/ | 108K | 2026-05-22 04:45 |
| `samam_256_faithful_p8_remote_debug` | results/ | 12K | 2026-05-22 04:43 |

#### 4. aaai2027系列中的invalid/空目录

共 14 个

| 目录名 | mtime | 大小 | ckpt |
|--------|-------|------|------|
| `aaai2027_eval_fiber_project_slerpe2_sigma0p0` | 2026-06-16 15:08 | 80K | 0 |
| `aaai2027_eval_fiber_project_slerpe2_sigma0p5` | 2026-06-16 15:08 | 80K | 0 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma0p0` | 2026-06-16 14:56 | 80K | 0 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma1p0` | 2026-06-16 14:52 | 80K | 0 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma1p5` | 2026-06-16 14:52 | 80K | 0 |
| `aaai2027_eval_fiber_project_lowanchor050e9_sigma0p5` | 2026-06-16 14:51 | 108K | 0 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 11:54 | 940K | 0 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 11:48 | 908K | 0 |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_kin070_seed42_b12a1` | 2026-06-14 13:44 | 820K | 0 |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1` | 2026-06-13 19:48 | 984K | 0 |
| `aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1` | 2026-06-13 17:33 | 668K | 0 |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b24a1` | 2026-06-13 06:05 | 664K | 0 |
| `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b26a1` | 2026-06-13 06:03 | 664K | 0 |
| `aaai2027_phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1` | 2026-06-13 05:10 | 744K | 0 |

### 必须保留清单

#### A. Baseline关键实验（用于论文对比）

| 目录名 | 模型 | 大小 | mtime |
|--------|------|------|-------|
| `cut` | CUT | 7.0M | 2026-06-24 02:05 |
| `s2wat` | S2WAT | 3.5M | 2026-06-24 02:05 |
| `sdedit_str_0p40` | SDEdit | 3.7M | 2026-06-24 02:05 |
| `sdedit_str_0p35` | SDEdit | 3.6M | 2026-06-24 02:05 |
| `sdedit_str_0p20` | SDEdit | 3.5M | 2026-06-24 02:05 |
| `sdedit_str_0p10` | SDEdit | 3.4M | 2026-06-24 02:05 |
| `sdturbo` | SDTurbo | 3.1M | 2026-06-24 02:05 |
| `samst_distinct5_512_wsl_stepalign40_remote_20260605_r1` | SaMST | 2.4G | 2026-06-05 19:21 |
| `samst_latent_distinct5_512_convergence_20260606_180529` | SaMST | 624M | 2026-06-06 18:40 |
| `samst_latent_distinct5_512_samecost_20260606_172021` | SaMST | 537M | 2026-06-06 17:27 |
| `samst_latent_distinct5_512_convergence_20260606_214051` | SaMST | 342M | 2026-06-06 22:31 |
| `samst_latent_distinct5_512_samecost_20260606_041227` | SaMST | 259M | 2026-06-06 04:34 |
| `samst` | SaMST | 8.3M | 2026-06-24 02:05 |
| `samst_latent_distinct5_512_samecost_20260606_145824` | SaMST | 2.3M | 2026-06-06 14:58 |
| `styleid` | StyleID | 33M | 2026-06-24 02:05 |

#### B. SaMam关键训练

| 目录名 | 路径 | 大小 | mtime | 备注 |
|--------|------|------|-------|------|
| `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601` | results/ | 3.9G | 2026-06-03 20:37 | 大训练 |
| `samam_latent_distinct5_512_convergence_20260606_222608` | results/ | 834M | 2026-06-07 00:13 | 大训练 |
| `samam_latent_distinct5_512_convergence_20260607_002420` | results/ | 626M | 2026-06-07 01:01 | 大训练 |
| `samam_latent_distinct5_512_samecost_20260606_162933` | results/ | 602M | 2026-06-06 16:53 | 大训练 |
| `samam_latent_distinct5_512_samecost_20260606_155105` | results/ | 514M | 2026-06-06 16:04 | 大训练 |
| `samam_distinct5_512_scratch_7k_250eval_remote` | results/ | 44G | 2026-07-02 07:50 | **★Tier1重点** 7000步/3.87h |

#### C. aaai2027关键阶段（有大量ckpt的）

| 目录名 | mtime | 大小 | ckpt数 |
|--------|-------|------|--------|
| `620_spatial_bridge` | 2026-06-21 08:37 | 3.3G | 213 |
| `inmortal-exp` | 2026-06-16 18:41 | 9.8G | 137 |
| `aaai2027_phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 10:19 | 1.5G | 28 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-17 12:21 | 1.3G | 24 |
| `aaai2027_phase2_i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen01` | 2026-06-17 11:02 | 1.3G | 24 |
| `aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1` | 2026-06-14 13:18 | 1.1G | 15 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor055_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 14:02 | 621M | 12 |
| `aaai2027_phase2_i2sb_orthogonal_lowanchor065_k070_e3_sigma0p02_b8a2_vlen010` | 2026-06-16 13:19 | 569M | 11 |
| `aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1` | 2026-06-14 03:17 | 351M | 10 |

#### D. final_works/ 全部保留
final_works/ 是最终展示用，全部保留。

#### E. 历史实验保留建议
experiments/ 共 269 个，其中：
- 有summary/checkpoint/images的: 232 个 → 保留
- 空/4K占位: 37 个 → 可删除（节省极少空间）

## 总结

- **总扫描实验数**: 430
- **已知根目录总大小**: ~108G
  - results/: 56G (其中 samam_distinct5_512_scratch_7k 占 44G)
  - runs/: 16G (其中 img2img_turbo smoke 占 ~10G)
  - exp/: 27G (其中 inmortal-exp 9.8G + highres 5.5G + aaai2027系列 ~12G)
  - experiments/: ~9.0G (其中 eval_cache 3.5G)
  - final_works/: 31M
- **可删除候选**: ~122 个目录，预计可释放 ~11-12G
  - smoke/probe/debug: 40
  - img2img_turbo smoke: 40 (10.10G)
  - SaMam/SaMST 失败probe: 28
  - aaai2027 invalid/空: 14
- **必须保留**: baseline关键实验 + SaMam大训练 + aaai2027有ckpt的 + final_works全部 + experiments有内容的
- **最大空间占用者**: samam_distinct5_512_scratch_7k (44G) > inmortal-exp (9.8G) > highres (5.5G) > eval_cache (3.5G) > 620_spatial_bridge (3.3G)