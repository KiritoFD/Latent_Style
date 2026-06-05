# 本地/远程实验考古总归纳 - 2026-06-05

这是一份给后续继续执行用的总入口，不是任务完成声明。它把当前已经逐目录打开、索引、清理、保留、仍缺口的结论合并到一个地方，避免只看零散 CSV 或旧乱码报告。

写入范围：仅 `EXPERIMENT_ARCHAEOLOGY`。本轮未改论文 TeX/PDF，未改源代码，未回滚或暂存用户已有脏文件。

## 1. 总结论

`G:\GitHub\Latent_Style` 不是单一实验目录，而是合并证据工作区。当前必须按三条主线理解：

- 本地 `G:\GitHub\Latent_Style`：数据、latent/cache、SchrodingerBridge 主实验、Related_Works baseline、Cycle-NCE 历史、root archive/tmp/paper scratch 混合在一起。
- 远程主树 `I:\Github\Latent_Style`：SchrodingerBridge 远程正式/探索 runs、SaMAM baseline、data/cache/archive、Cycle-NCE RAR 历史和 expanded evidence。
- 远程 TokenizerClean `I:\Github\Latent_Style_TokenizerClean`：AAAI2027/tokenizer 关闭证据面，和远程主树不是同一条 evidence surface。

当前已经形成了数据层：

- `final_master_experiments.csv`：22629 行。
- `final_timeline.csv`：7829 个事件。
- `conclusions_by_dataset.csv`：25 个 dataset/setting 行。
- `final_by_dataset/`：按 dataset/setting 分开的 CSV。
- `timing_quality_master_20260605.csv`：1093 行 timing overlay。
- `manual_top_level_directory_index_20260605.csv`：67 行顶层目录索引。

当前缺的不是“有没有数据”，而是继续逐目录把剩余 owner choice、tracked-file policy、timing promotion、archive/tmp provenance 和最终一致性审计做完。

## 2. 现场事实

本地 live check：

| path | dirs | files | 结论 |
| --- | ---: | ---: | --- |
| `G:\GitHub\Latent_Style` | 35 | 39 | 根目录是混合证据工作区 |
| `G:\GitHub\Latent_Style\SchrodingerBridge\exp` | 110 | 4 | 当前 LANCET/LBM 输出面 |
| `G:\GitHub\Latent_Style\Related_Works` | 9 | 1 | baseline evidence root |
| `G:\GitHub\Latent_Style\Cycle-NCE` | 39 | 58 | 历史 Cycle-NCE evidence root |
| `G:\GitHub\Latent_Style\EXPERIMENT_ARCHAEOLOGY` | 3 | 224 | 当前考古输出目录 |

远程 live check：

| path | dirs | files | 结论 |
| --- | ---: | ---: | --- |
| `I:\Github` | 3 | 2 | 非空；包含主树和 TokenizerClean |
| `I:\Github\Latent_Style` | 23 | 53 | 远程主树存在 |
| `I:\Github\Latent_Style\SchrodingerBridge\exp` | 123 | 1 | 远程主树 exp 存在 |
| `I:\Github\Latent_Style\Cycle-NCE` | 26 | 78 | Cycle-NCE 历史面仍存在 |
| `I:\Github\Latent_Style_TokenizerClean` | 17 | 37 | TokenizerClean 独立树存在 |
| `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp` | 142 | 23 | TokenizerClean exp 清理后仍大规模存在 |

远程关键验证：

- `I:\Github\Latent_Style\Cycle-NCE\45.rar`：absent。
- `I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605`：present。
- `I:\Github\Latent_Style\experiments.rar`：absent。

## 3. 本地结论

本地目录不能按扩展名或大小扫删：

- `Dataset`、`style_data`、`latent-256`、`clip-feats-vitb32`、`horse2zebra` 是数据/latent/feature cache，不是训练 checkpoint。
- `eval_cache` 是 CLIP/VAE/ArtFID/DINO 等 eval dependency，不是实验垃圾。
- `SchrodingerBridge/exp` 是当前 LANCET/LBM evidence surface，包含正式 eval、timing、compact-anchor、diagnostic 和历史 probe。
- `Related_Works` 是 baseline evidence surface，包含 SaMAM/SaMST/CUT/CycleGAN/Seedream 等。
- `Cycle-NCE` 是历史 evidence surface；视频、summary、metrics、archive lineage 都有意义。
- `archive`、`tmp`、paper scratch 是混合 provenance surface，必须单独查，不可碰 TeX/PDF。

本地已经完成的清理：

- 本地 checkpoint-like 清理 ledger：`cleanup/manual_deleted_checkpoints_20260605.csv`。
- 无效 eval cache residue：55.994 MB。
- root duplicate archive/stale launcher residue：1503.203 MB 加小残留。
- failed dataset download cache residue：63.948 MB。
- local remaining-surface whitelist cleanup：237.860 MB。
- CUT video work-frame dirs：3068.463 MB。
- generated-media pass4 duplicate cleanup：101.913 MB。

本地 generated media 当前结论：

- formal eval、paper bundle、no-op/IDT control、timing benchmark、protocol baseline、compact-anchor、diagnostic、generation-only calibration、inference preview、tracked CUT raw 都不能按媒体数量删除。
- `seedream_gap` 已有 owner-decision manifest：7 个 input set、5250 JPG、全部 `delete_whitelist=no`。
- `inference_param_sweep_t01e8_quick` 已有 owner-decision manifest：14 个参数点、3500 JPG、全部 `delete_whitelist=no`。
- `inference_param_sweep_t01e8_fine` 已有 owner-decision manifest：8 个参数点、2000 JPG、全部 `delete_whitelist=no`。
- CUT raw web outputs 是 tracked/mixed boundary，不能在 generated-media cleanup 里直接删。

本地仍缺：

- owner 对 `seedream_gap` 和 inference sweeps 选择 keep/migrate/delete。
- CUT raw outputs 的 tracked-file migration/untracking policy。
- root `archive/tmp/paper scratch` provenance pass。
- dataset-by-dataset 中文结论重写，替代旧乱码/粗糙字段。

## 4. 远程主树结论

远程主树不是空盘，也不是单个 experiments 文件夹。它包含 current runs、baseline、cache、RAR history 和 expanded evidence。

已完成远程主树清理：

- SchrodingerBridge epoch thinning：84 个 checkpoint，4961.604 MB。
- SaMAM alias cleanup：7 个 alias，1931.291 MB。
- data/cache/archive residue cleanup：11 个 exact targets，381.807 MB。
- duplicate/stale archive cleanup：3 个 archive，3290.714 MB。
- weight-only RAR cleanup：6 个 RAR，6553.384 MB。
- resolved duplicate `experiments.rar`：8091.026 MB。
- Cycle-NCE `45.rar`：507.452 MB；删除前已提取 curated nonweight package，且 curated 包内 weight-extension count 为 0。

远程主树保留：

- SchrodingerBridge current/formal anchors。
- SaMAM central curve checkpoints。
- valid data/latent/cache roots。
- expanded `experiments` evidence。
- Cycle-NCE evidence dirs 和 curated 45 nonweight package。

远程主树仍缺：

- cache duplicate 不能只凭 hash equality 删除；需要 canonical cache-root migration、symlink/junction policy、offline eval verification。
- 只在新 exact whitelist 能证明时继续清理。

## 5. 远程 TokenizerClean 结论

TokenizerClean 是独立 evidence surface。不能并入远程主树解释。

已完成 TokenizerClean 清理：

- uncited exploratory checkpoints：141 个，5198.991 MB。
- no-summary probe/calibration checkpoints：18 个，362.391 MB。
- pure orphan probe dirs/weights：3 个目录和相关权重，170.017 MB。
- uncited generated media：43008 个文件，11883.246 MB。
- training-log-only no-summary payload weights：7 个 checkpoint，248.429 MB。

TokenizerClean 当前保留：

- 26 个 cited/current media dirs：policy media 46483 files，manifest 已映射到 summary/CSV/grid/generated/checkpoint 代表路径。
- 7 个 trained no-summary payload dirs：5 个 metadata-only，2 个 evidence-bearing。
- retained no-summary payload 权重面：3 个文件，130.883 MB。
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` 的 missing resume anomaly 已注释为 metadata-only，不可作为 clean lineage/evaluated result 推广。

TokenizerClean 仍缺：

- retained media 的 owner archive/migration choice。
- path-preserving migration policy；没有这个不能移动或删除 cited/current media。

## 6. 实验脉络

当前 lineage 应按阶段读：

1. 2026-02 到 2026-03：legacy style transfer / Cycle-NCE / early baseline history。
2. 2026-03 到 2026-04：CUT、CycleGAN、StyleID、SDEdit、StarGAN 等 baseline 和 protocol outputs。
3. 2026-05 上旬：SchrodingerBridge / LANCET / LBM legacy strict750 和 review ablation。
4. 2026-05 下旬：WikiArt512、Distinct5、highres、seedream、baseline timing/eval 扩展。
5. 2026-06 初：Distinct5 formal evidence、SaMAM remote baseline、local WSL evidence。
6. 2026-06-03 以后：AAAI2027 / TokenizerClean closing surface，用于 claim closing 和 tokenizer/endpoint/flow variants。

这条脉络的作用是防止把 dry-run、smoke、failed probe、diagnostic media、legacy baseline、current formal evidence 混为同一种“实验结果”。

## 7. Timing 结论

Timing 已有证据但还没变成最终 paper-facing 表：

- docs timing master：419 行。
- archaeology timing overlay：1093 行。
- missing-doc claim candidates source-open：26 行。
- 原始单位保留，缺失值留空。

仍缺：

- 370 条 docs master rows 缺 overlay/source-open 覆盖。
- owner 选择哪些 rows promotion 到后续 paper-facing timing table。
- 区分 train wall time、full-eval wall time、pure inference/generation time、smoke/audit-only。

## 8. 清理原则

已经执行的清理都必须满足：

- exact path。
- policy CSV/MD。
- deletion ledger。
- post-delete verification。
- 不按扩展名、大小、媒体数量、hash equality 做广义删除。

不能动：

- 论文 TeX/PDF。
- 当前 source code dirty files。
- Related_Works 用户/其他线程改动。
- tracked CUT raw outputs，除非先有 tracked-file policy。

## 9. 8 小时级继续计划

### 第 1 小时：本地 archive/tmp/paper scratch provenance

- exact path 打开 root `archive`、`tmp`、paper snapshots。
- 分类 duplicate archive、render artifact、paper build output、active dirty file。
- 不碰 TeX/PDF。

### 第 2 小时：CUT raw tracked-file policy

- 对 `raw_results` 和 `raw_results_val` 的 tracked HTML/media 写迁移/保留/删除策略。
- 任何删除前必须先有 untracking/migration plan。

### 第 3 小时：retained media archive choices

- 本地 `seedream_gap`、quick/fine sweeps：等待 owner choice。
- 远程 TokenizerClean 26 retained media：等待 owner choice。
- 没有 owner choice 时不删除。

### 第 4 小时：timing promotion

- 从 1093 overlay 和 26 source-open rows 中标出可复用 rows。
- 保留原始单位。
- 写 train/full-eval/inference 边界。

### 第 5 小时：dataset-by-dataset 中文重写

- 从 `final_by_dataset/*.csv` 逐个写中文结论。
- 不用旧乱码字段当最终可读结论。

### 第 6 小时：远程主树剩余 archive/cache exact review

- 只处理新可证明 exact whitelist。
- cache 需要 migration proof。

### 第 7 小时：README/count consistency

- 对 README、cleanup summary、direct index、authoritative index 做 count audit。

### 第 8 小时：完成性审计

- 对目标逐条核验：本地、远程、所有 dataset、timing、cleanup、提交范围。
- 只有所有项有强证据才可声明完成。

## 10. 当前完成性判断

未完成。

已经有大量索引、清理、manifest 和结论层；但完整目标要求“整个仓库、本地和远程、所有实验数据、每个目录手动 check、最大限度整理清理并提交”。当前证据还不足以证明每个剩余目录都完成 owner choice、tracked-file policy、timing promotion 和最终一致性审计。因此任务继续保持 active。
