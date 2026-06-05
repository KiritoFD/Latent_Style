# Latent_Style 当前实验考古清晰结论 - 2026-06-05

本文件是当前清晰入口，不是“全任务已完成”的声明。它把已经落地的
master/timeline/per-dataset CSV、手工目录核验、远程 I 盘核验、清理账本和
仍缺的 owner 决策放在同一个可读报告里。

写入范围：仅 `EXPERIMENT_ARCHAEOLOGY`。本轮没有修改论文 TeX/PDF、源码、
Related_Works 脚本或用户已有脏文件。

## 1. 直接结论

### 本地 G:\GitHub\Latent_Style

本地不是单个实验目录，而是一个混合证据仓库：

- `SchrodingerBridge`：当前 LANCET/LBM 主线、WikiArt512、Distinct5、
  timing/full-eval、诊断、消融、representation/tokenizer 相关证据。
- `Related_Works`：SaMAM、SaMST、CUT、CycleGAN、S2WAT、StyleID、
  Seedream 等 baseline 复现和评测证据。
- `Cycle-NCE`：历史方法/旧实验/视频/summary/metrics/source snapshot 证据。
- `Dataset`、`style_data`、`latent-*`、`clip-feats-vitb32`、`eval_cache`：
  数据、latent、CLIP feature、模型/eval cache 依赖，不是可直接删除的
  checkpoint。
- `archive`、`tmp`、root `exp`、paper workspace：混合 scratch 和历史包，
  必须按 provenance 处理，不能按扩展名或大小扫删。

本地已经完成的索引层：

- `final_master_experiments.csv`：22629 行，覆盖本地和远程实验证据。
- `final_timeline.csv`：7829 个事件。
- `final_by_dataset/*.csv`：25 个 dataset/setting 分文件。
- `manual_top_level_directory_index_20260605.csv`：67 行，其中 32 行是本地
  top-level/family 手工分类。
- `manual_family_walkthrough_20260605.csv`：31 行，按实验家族梳理。
- `manual_coverage_matrix_20260605.csv`：41 行，记录每个区域的打开深度。

本地已经完成的清理：

- checkpoint-like 清理：875 个文件，46032.053 MB。
- eval cache 失败残留：55.994 MB。
- root duplicate archive / launcher residue：约 1503.203 MB。
- dataset/cache 失败下载残留：63.948 MB。
- remaining surface 白名单清理：237.860 MB。
- CUT video 中间帧目录：3068.463 MB。
- generated-media pass4 精确重复项：101.913 MB。

本地仍缺：

- `seedream_gap` 和 `inference_param_sweep_t01e8_*` 已在 pass5 逐目录打开，
  但仍是 `retain_pending_owner`。没有 owner 决策前不删。
- CUT `raw_results/raw_results_val` 已在 pass5 逐目录打开，但包含 tracked
  HTML/media 或 mixed tracked/ignored payload，不能当普通 generated-media 删。
- `archive/tmp/paper scratch` 还需要单独 provenance pass，且不能碰论文 TeX/PDF。

### 远程 I:\Github\Latent_Style

远程 I 盘不是空目录。2026-06-05 本轮重新 SSH 只读核验结果：

- `I:\Github` 当前有 3 个目录、2 个文件：
  `26AI-H`、`26AI-H.zip`、`Latent_Style`、
  `Latent_Style_TokenizerClean`、`find_clip_remote.bat`。
- `I:\Github\Latent_Style` 存在：23 个目录、53 个文件。
- `I:\Github\Latent_Style\SchrodingerBridge\exp` 存在：123 个目录、1 个文件。
- `I:\Github\Latent_Style\Cycle-NCE` 存在：26 个目录、79 个文件。
- `I:\Github\Latent_Style\Cycle-NCE\45.rar` 在本轮后续白名单步骤中已删除，
  释放 507.452 MB。
- `I:\Github\Latent_Style\experiments.rar` 已不存在。
- `I:\Github\Latent_Style\Cycle-NCE\experiments.rar` 已不存在。
- `I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605`
  存在：递归 6086 个文件，145.512 MB，weight 扩展文件 0。这里包含
  6084 个非权重 payload 加 2 个顶层 manifest/removed-weight CSV。

远程主树顶层含义：

- `.git`、`.torchinductor_cache`、`.vscode`：仓库/环境痕迹。
- `Cycle-NCE`：历史证据、RAR provenance、curated 45 nonweight package。
- `data`、`style_data`、`latent-*`、`latents*`：数据和 latent/cache 根。
- `eval_cache`：CLIP/ArtFID/VAE/DINO/eval 依赖。
- `experiments`：`experiments.rar` 删除后保留的 expanded evidence。
- `Related_Works`：远程 baseline 证据。
- `SchrodingerBridge`：远程主线训练/eval 证据。
- `seedream45_api`、`StarGAN`、root `exp`：历史/辅助证据面。

远程主树已经完成的清理：

- SchrodingerBridge epoch thinning：84 个 checkpoint，4961.604 MB。
- SaMAM alias checkpoint：7 个 redundant `last*.ckpt`，1931.291 MB。
- data/cache/archive residue：11 个精确目标，381.807 MB。
- duplicate/stale archive：3 个 archive，3290.714 MB。
- weight-only RAR：6 个 RAR，6553.384 MB。
- resolved duplicate `experiments.rar`：1 个 archive，8091.026 MB。
- Cycle-NCE `45.rar` 原 archive：1 个 archive，507.452 MB；非权重证据已
  保留在 `_curated_45_nonweight_20260605`。

远程主树仍缺：

- cache duplicate 不能只靠 SHA256 一致删除；需要 canonical cache root
  migration、symlink/junction policy 和离线 eval 验证。

### 远程 I:\Github\Latent_Style_TokenizerClean

TokenizerClean 是独立证据树，不应和远程主树合并解释。

本轮重新 SSH 只读核验结果：

- `I:\Github\Latent_Style_TokenizerClean` 存在：17 个目录、37 个文件。
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp` 存在：
  142 个目录、23 个文件。

TokenizerClean 顶层含义：

- `SchrodingerBridge`：AAAI2027/tokenizer 主证据树。
- `Cycle-NCE`、`Related_Works`、`final_works`、`lambda_grid`、
  `step_count_sweep`、`latent_cyclegan`、`review_additional_experiments_aggregates`：
  旧实验、baseline、辅助对照和聚合证据。
- `eval_cache`、`style_data`：依赖和数据根。
- `efficiency`、`fast_infer_ablate43`、`o20_d3`：效率/快推理/附加实验记录。

TokenizerClean 已完成的清理：

- uncited exploratory checkpoints：141 个，5198.991 MB。
- no-summary probe/calibration checkpoints：18 个，362.391 MB。
- pure orphan probe dirs/weights：3 个目录和相关权重，170.017 MB。
- uncited generated media：43008 个文件，11883.246 MB。
- training-log-only no-summary payload weights：7 个 checkpoint 文件，
  248.429 MB；只删 `.pt`，保留 `config.json` 和 `logs\training_*.csv`。

TokenizerClean 当前保留：

- 26 个 cited/current media dirs：46483 个媒体文件，7501.518 MB，
  均为 `keep_no_delete`，待 archive migration。
- 7 个 trained no-summary payload dirs：清理后只剩 3 个 weight 文件，
  130.883 MB；其中 5 个 training-log-only 目录已变成 metadata-only，
  2 个外部/下游证据目录继续保留权重。
- 一个明确异常：
  `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` 的 config 指向不存在的
  `epoch_0004.pt`，需要修复或注释 lineage。

TokenizerClean 仍缺：

- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` 的缺失 resume
  checkpoint anomaly 需要修复或注释，不应作为干净 lineage 推广。
- 26 个 cited/current media dirs 需要 citation-to-artifact manifest，之后才能
  谈迁移/压缩/删冗余媒体。

## 2. 数据集和实验面归纳

当前 master 表按 `dataset_key` 的核心分布如下：

| dataset_key | rows | local | remote | metric | summary | log | timing | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cycle_nce` | 11794 | 10590 | 1204 | 10590 | 824 | 380 | 0 | 最大历史证据面；不能当垃圾目录处理。 |
| `schrodingerbridge_exp_general` | 4051 | 3138 | 913 | 3239 | 481 | 304 | 36 | LANCET/LBM 主实验和一般实验面。 |
| `schrodingerbridge_weight_sweep` | 1285 | 811 | 474 | 800 | 402 | 72 | 11 | 权重/损失/配置 sweep，已 thinning 但仍保留 anchor。 |
| `legacy_style_transfer_experiments` | 1120 | 0 | 1120 | 0 | 420 | 700 | 0 | 远程旧实验日志/summary 面，辅助背景。 |
| `schrodingerbridge_grid_search` | 1013 | 544 | 469 | 544 | 220 | 249 | 0 | grid/search 探索证据。 |
| `schrodingerbridge_vae_backend` | 699 | 81 | 618 | 81 | 386 | 232 | 0 | VAE/backend/decode 相关探索。 |
| `schrodingerbridge_frontier` | 692 | 600 | 92 | 600 | 8 | 84 | 0 | frontier/promoted 候选面。 |
| `schrodingerbridge_representation_probe` | 567 | 503 | 64 | 471 | 58 | 6 | 0 | representation/tokenizer/latent probe。 |
| `distinct5_512` | 417 | 305 | 112 | 278 | 10 | 46 | 156 | 当前核心 Distinct5-512 claim 面。 |
| `wikiart512_5style` | 200 | 108 | 92 | 144 | 9 | 44 | 22 | WikiArt512 5-style formal/timing 面。 |
| `schrodingerbridge_root_legacy` | 197 | 72 | 125 | 64 | 42 | 77 | 14 | SchrodingerBridge root legacy 证据。 |
| `legacy256_overfit50` | 131 | 126 | 5 | 123 | 3 | 0 | 17 | legacy256 / overfit50 对照。 |
| `run511_5domain` | 120 | 48 | 72 | 3 | 64 | 8 | 46 | 5-domain baseline 汇总。 |
| `schrodingerbridge_aaai2027` | 87 | 0 | 87 | 0 | 2 | 36 | 49 | AAAI2027/tokenizer closing 面。 |
| `strict_protocol_750` | 79 | 77 | 2 | 29 | 2 | 0 | 24 | strict 750 eval protocol 面。 |
| `photo_monet_5x5` | 42 | 17 | 25 | 0 | 16 | 8 | 2 | Photo-Monet 5x5 baseline/qualitative 面。 |

长尾小面包括 `path_family_final_works`、`schrodingerbridge_docs_experiments`、
`schrodingerbridge_destructive_ablation`、`unclassified_curated_experiments`、
`schrodingerbridge_review_additional`、`related_works_baselines`、`s2wat`、
`path_family_step_count_sweep` 等，均已在 per-dataset CSV 中拆出。

## 3. 实验脉络

当前能整理出的主线如下：

1. 2026-02 到 2026-03：legacy style-transfer、no-edge、no-tokenized、
   overfit50、IDT/no-op sanity，主要用于早期 baseline 和负例。
2. 2026-03 到 2026-04：Cycle-NCE / Latent AdaCUT / CUT / CycleGAN / 旧
   baseline 形成历史证据面。
3. 2026-05：SchrodingerBridge/LANCET 大规模探索，包括 grid/search、
   weight sweep、frontier、VAE backend、representation/tokenizer probe。
4. 2026-05-30 到 2026-06-02：WikiArt512 和 Distinct5-512 formal/full-eval/
   timing 证据成型。
5. 2026-06-03 以后：AAAI2027/TokenizerClean 用于 claim closing，包括
   flow-loss、SA-SWD、tokenizer execution、localization、time-to-parity。

结论：不能把 smoke、dry-run、failed probe、no-op control、历史 baseline 和
当前 formal claim 混在一起。当前 claim 应优先使用 Distinct5-512、WikiArt512、
strict protocol 750、docs timing master 和 source-open timing rows。

## 4. Timing 状态

Timing 不是空白，但还没完全转成最终 paper-facing 表。

- docs timing master：419 行。
- archaeology timing overlay：1093 行。
- missing-docs source-open timing candidates：26 行，已逐条 source-open。
- 当前策略：保留原始单位；训练时间不强转秒；缺失值留空；区分 generation-only、
  full-eval wall time、train+eval、smoke、audit-only。

仍缺：

- 370 个 docs timing master 行还没有 overlay/source-open 覆盖。
- 需要 owner 选择哪些 timing rows 可进入最终论文/标定表。

## 5. 清理原则和当前结论

已经执行的删除都是“精确路径 + policy CSV/MD + ledger + post-delete verify”
模式。当前不能执行的删除：

- 不能按图片数量删除 generated media。
- 不能按 `.pt/.pth/.ckpt` 扩展名删除所有权重。
- 不能按 SHA256 cache 重复直接删 cache。
- 不能删除 tracked CUT raw outputs。
- 不能碰论文 TeX/PDF 和用户已有脏文件。

下一批最明确的清理方向：

1. TokenizerClean missing-resume anomaly：给
   `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` 写 lineage 注释或修复
   `epoch_0004.pt` 引用。
2. TokenizerClean 26 个 cited/current media dirs：先建 citation-to-artifact
   manifest，再决定是否 archive/migrate。
3. 本地 `seedream_gap` 和 `inference_param_sweep_t01e8_*`：需要 owner 决策，
   若删除必须先保留参数清单和代表性样例。
4. 本地 `archive/tmp/paper scratch`：单独 provenance pass，不动 TeX/PDF。

## 6. 8 小时级继续计划

### 第 1 小时：远程 45.rar 删除闭环 - 已执行

- 已写 `45.rar` 删除白名单 CSV/MD。
- 已删除远程原 `I:\Github\Latent_Style\Cycle-NCE\45.rar`。
- post-delete verify 已通过：archive absent，curated nonweight package
  present，manifest present，removed-weight ledger present，weight_ext_files=0。
- 本块后续只需提交。

### 第 2 小时：TokenizerClean trained no-summary payload - 已执行权重清理

- 已逐个打开 7 个 trained no-summary payload 的 config/training/weight/log。
- 5 个无外部证据的 training-log-only 目录已按 exact whitelist 删除 7 个
  checkpoint weight，释放 248.429 MB。
- 未删除任何目录、`config.json`、`logs\training_*.csv` 或 source snapshot。
- 2 个 evidence-bearing payload 继续保留 3 个 weight，合计 130.883 MB。
- 新增 live recheck CSV/MD 记录每个目录当前权重数、resume 字段、
  training CSV 尾行和 summary/full_eval 缺失状态。

### 第 3 小时：TokenizerClean cited/current media manifest

- 为 26 个 retained media dirs 建 citation-to-artifact manifest。
- 映射 docs/paper/source 引用到 summary、CSV、grid、images、checkpoint。
- 暂不删除 current/cited media。

### 第 4 小时：本地 archive/tmp/paper scratch provenance

- 只读打开 root `archive`、`tmp`、paper snapshot/scratch。
- 分类 duplicate archive、render artifact、paper build output、active dirty file。
- 不碰 TeX/PDF；只对白名单重复包写删除 ledger。

### 第 5 小时：本地 pass5 owner 决策候选

- 为 `seedream_gap` 和 inference sweeps 生成参数/样例 manifest。
- 如果 owner 同意，删除完整图片目录并 post-delete verify manifest。
- CUT raw outputs 只做 tracked migration 方案，不直接删。

### 第 6 小时：timing promotion

- 从 1093 overlay 和 26 source-open candidates 中标出 paper-facing rows。
- 更新或另写标定复用表，保留原始单位。
- 明确 full-eval wall time vs pure inference time。

### 第 7 小时：dataset-by-dataset 中文结论重写

- 以 `final_by_dataset/*.csv` 为源，重写 25 个 dataset/setting 中文结论。
- 替换/旁路已有 mojibake `conclusions_by_dataset.csv` 可读字段。

### 第 8 小时：最终一致性审计

- 校验 master/timeline/per-dataset/timing/cleanup README 一致性。
- `Import-Csv` 校验新 CSV。
- `git diff --check -- EXPERIMENT_ARCHAEOLOGY`。
- 只提交 `EXPERIMENT_ARCHAEOLOGY`。

## 7. 当前未完成声明

完整目标还没完成。当前已经完成的是：大范围索引、多个手工目录 pass、远程 I
盘主树和 TokenizerClean 清理、timing source-open、pass5 逐目录核验，以及本
清晰结论入口。仍要继续按 8 小时计划推进，直到每个剩余目录都有明确
retain/delete/migrate 结论和账本。
