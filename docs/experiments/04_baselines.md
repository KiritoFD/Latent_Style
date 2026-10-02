# 04 对比方法（Baselines）

**关键结论**：`Related_Works/` **不是** Table I 的数据源——它保存的是旧协议（D5 CLIP-B/32 + LPIPS，或 `protocol_a_800` 的 photo/monet/vangogh/cezanne/Hayao 板），没有 DINO、没有 P2A/R5。
Table I 的 DINO 数字来自 `WEAVE/state/dino/`；CLIP/LPIPS 只有部分有原始文件（见 [02](02_results_of_record.md)）。

## Table I 中的方法

| 方法 | 来源 | 运行方式 | CLIP/LPIPS 出处 | 时间出处 |
|---|---|---|---|---|
| SD-Turbo（**ICME 版已移除**，见 06 A19） | stabilityai/sd-turbo | D5：`Related_Works/runs/sdturbo_5x5`；三板：`WEAVE/tools/bench_all_3060.py` / `_run_sdturbo_remote.py`（后者**未入库**） | D5：`WEAVE/docs/baseline/README.md`（0.6933/0.0033） | 0.404 s/张 × 750 ≈ 5 min（`make_radar*.py` 注释） |
| StyleAligned | google/style-aligned 的 SD1.5 移植 `WEAVE/tools/style_aligned/` | D5：`WEAVE/tools/run_stylealigned_distinct5.py`（20 步，cfg 7.5，inversion cfg 3.5，seed 42，512） | paper-only；**冲突**：`WEAVE/BASELINE_PROGRESS.md` 记 D5 CLIP 0.8739 / LPIPS 0.7825，论文 0.780 / 0.869 | 6.18 s/张 → 77 min |
| Z-STAR | HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement | `WEAVE/tools/_zstar_launch.bat` → `_run_zstar_remote.py`（未入库） | paper-only | **估计** ~3 h（512² 在 12 GB OOM，按步数外推） |
| StyleShot | open-mmlab/StyleShot | `_styleshot_launch.bat` → `_run_styleshot_remote.py`（未入库），`--preprocessor Contour --prompt "a painting"` | paper-only | 24.63 s/张 → 5.1 h |
| StyleID | jiwoogit/StyleID | `Related_Works/baseline_pipeline/scripts/run_styleid.py` | D5：docs/baseline/README（0.8223/0.5523） | 论文 63 min；仓库内只有 51.4 min / 603 s 两个不同口径，63 min 出处未找到 |
| CUT | taesungp/CUT（`Related_Works/repos/external/CUT`） | `copy_cut_results.py`、`collect_cut_results.py` | D5：0.7137/0.3743（README 记 745 对） | 训练 322.6 min、推理 5 min，标注 "prior value, not re-measured on 3060" |
| SaMST | `Related_Works/repos/SaMST-main` | `run_samst*.py`、`run_511/outputs/samst_750_strict` | D5：0.6183/0.7490 | 训练 39.5 min、推理 10 min（prior value；run_511 记 39.8 s/750） |
| SaMam | SaMam（Mamba） | `run_samam_latent_baseline.py`、`eval_samam_checkpoint_curve.py`（81 个 ckpt 曲线） | D5：0.5816/0.2434 | 训练 436 min、推理 17.6 min（prior value） |
| StyTR-2 | diyiiyiii/StyTR-2 | 输出/评测：`WEAVE/exp_baselines/stytr2/*`（运行脚本未入库） | `clip_lpips_summary.json`（三板一致） | 训练 ~1440 min 为作者报告；推理 38 min 出处未找到 |
| AesPA-Net | 官方实现 | `WEAVE/exp_baselines/aespa/*` | `clip_lpips_summary.json`（三板一致） | 推理 23 min 出处未找到 |
| Seedream 4.5 | 字节 API | `Related_Works/baseline_pipeline/scripts/run_seedream_wikiart512.py` | D5：0.7198/0.4767 | API，无 |
| Latent-WCT | 自建解析 baseline | 归档：`WEAVE/archives/legacy-scripts-20260715/_run_latent_wct_*.ps1` 等 | paper-only（DINO 也 paper-only） | 18 s 出处未找到 |

评测管线：CLIP-S/LPIPS/ArtFID → `WEAVE/utils/run_evaluation.py`；DINO → `WEAVE/utils/compute_dino_metrics.py`；
汇总 `WEAVE/state/dino/*.json` → `dino_main.json` 的聚合脚本**未找到**。无显式 pair-list 文件，配对由目录结构与文件名 `{src_style}__{src_stem}__to__{tgt_style}.png` 隐式定义。

## 已跑但未入表

| 方法 | 结果（旧协议） | 位置 |
|---|---|---|
| AdaIN、WCT(VGG19) | D5 CLIP/LPIPS .6679/.7425、.7063/.6348；DINO D5 AdaIN 0.336/0.214、WCT 0.136/0.025 | `WEAVE/docs/baseline/README.md`、`WEAVE/state/dino/` |
| SDEdit（s=0.10–0.40） | protocol_a 见表 | `Related_Works/runs/sdedit_multi` |
| S2WAT | protocol_a .7138/.7464/.5263 | `Related_Works/run_511/complete_750/s2wat_strict` |
| CycleGAN(-Turbo)、ArtBank、AesFA、LBM、Dreambooth-SD | smoke / 无官方权重 | `Related_Works/repos/`、`BASELINE_CKPT_STATUS.md` |
| SCSA、CSGO、StyleGallery、StyleShot（Related_Works 下） | 空 submodule gitlink（无 `.gitmodules`），只有失败的 smoke | `Related_Works/{CSGO,SCSA,...}`、根目录 `StyleShot/` |

`LANCET` 不是 baseline，是本项目 Cycle-NCE 时期的一个配置（`Cycle-NCE/freq/freq_09_lancet`）。

## 待补（按优先级）

1. 把 `_run_{sdturbo,stylealigned,zstar,styleshot}_remote.py`、DINO 聚合脚本、每个 (方法, 板) 的 CLIP/LPIPS `summary.json` 入库。
2. 在 RTX 3060 上重测 CUT/SaMST/SaMam/StyleID/StyTR-2/AesPA 推理时间，或在表注中说明来源。
3. 核对 StyleAligned D5 CLIP/LPIPS 两组数字；核对 CUT R5 DINO-C 与 D5 完全相同（疑似缓存复用）；SD-Turbo R5 n=1123。
