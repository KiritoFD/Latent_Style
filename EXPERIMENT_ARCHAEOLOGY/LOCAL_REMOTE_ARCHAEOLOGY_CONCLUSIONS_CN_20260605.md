# 本地与远程实验考古结论 - 2026-06-05

这份文档是给人读的结论入口，不替代 CSV。逐目录机器索引见：

- `manual_top_level_directory_index_20260605.csv`
- `manual_cleanup_retention_and_next_candidates_20260605.csv`
- `manual_timing_evidence_20260605.csv`
- `manual_directory_classification_20260605.csv`

本轮只写 `EXPERIMENT_ARCHAEOLOGY/**`。没有改论文 tex/pdf，没有动源码脏文件，没有 stage 用户已有脏文件。

## 一句话结论

本地仓库已经不是“checkpoint 垃圾堆”，而是一个混合证据库：当前可引用证据在 `SchrodingerBridge/docs/experiments` 和少量 `SchrodingerBridge/exp`，历史/负结果主要在 `Cycle-NCE`、`final_works`、旧 `SchrodingerBridge/exp` 家族和 `Related_Works`，大体积 `.pt` 多数是 latent/cache/dependency，不是训练 ckpt。

远程 `I:\Github\Latent_Style` 也不是空盘或未整理状态：真正需要保留的主线权重集中在 `SchrodingerBridge/exp` 的 101 个 current/lineage 权重和 `Related_Works` 的 19 个 SaMAM 中央曲线 ckpt；其它大头主要是 latent backend cache、eval cache、scale dataset tensors、历史 archive。

远程 `I:\Github\Latent_Style_TokenizerClean` 是当前 AAAI/tokenizer 干净工作树，不是垃圾目录。它自己的 docs 已经规定：`exp/` 不能 mass delete，因为 docs/master log 仍直接引用路径。

## 当前本地状态

### 本地剩余 weight-like 文件是什么

刷新后的本地精确扫描：

| path | count | size MB | 结论 |
|---|---:|---:|---|
| `SchrodingerBridge/exp` | 9 | 363.490 | 8 个 WikiArt512 正式 epoch 权重 + 1 个 ArtFID 依赖 |
| `Related_Works` | 31 | 814.913 | VGG/Inception/LPIPS 依赖 + tiny fake eval placeholders |
| `Cycle-NCE` | 3 | 1.054 | eval_cache/ref_feats，不是训练 ckpt |
| `final_works` | 0 | 0 | 已无训练权重 |
| `lambda_grid` | 0 | 0 | dry-run/status，不是实际训练 |
| `step_count_sweep` | 0 | 0 | dry-run/status，不是实际推理 |
| `eval_cache` | 30 | 4383.348 | metric/model cache，不能按 ckpt 清理 |
| `clip-feats-vitb32` | 10361 | 1303.526 | per-image CLIP features |
| `latent-256` | 10361 | 177.702 | per-image latent tensors |
| `seedream45_api` | 1 | 0.001 | tiny fake eval placeholder |

之前记录的 `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_vramprobe` 3 个 probe 权重在当前扫描中已经不存在，因此已从“可删候选”修正为“过期候选，无当前删除动作”。

### 本地逐目录结论

| 目录 | 定位 | 清理结论 |
|---|---|---|
| `Dataset` | Distinct5/WikiArt/legacy 输入数据 | 保留，不是 ckpt |
| `style_data` | legacy style/content 数据 | 保留 |
| `latent-256` | legacy latent 数据 | 保留 |
| `clip-feats-vitb32` | CLIP feature cache | 保留 |
| `eval_cache` | CLIP/VAE/ArtFID/LPIPS 等 eval 依赖 | 保留 |
| `SchrodingerBridge` | 当前主线 + docs + exp + configs | 保留并继续细分 |
| `SchrodingerBridge/docs/experiments` | 当前论文/实验引用最可信证据面 | 保留，作为 timing/claim 主入口 |
| `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` | WikiArt512 正式训练/评测锚点 | 保留 8 个 epoch 权重 |
| `SchrodingerBridge/exp/timing_20260601/20260602` | timing-only 证据面 | 保留 summary/log/metrics；图片 payload 可按引用关系另行清 |
| `SchrodingerBridge/exp/_smoke*` | 本地 smoke/calibration | 不作为 paper evidence；当前未发现大权重 |
| `SchrodingerBridge/exp/style_representation*` | tokenizer/representation 探索 | 作为探索证据保留，不直接引用论文 |
| `SchrodingerBridge/exp/vae_backend*` | VAE backend/decode 探索 | 保留为负结果/瓶颈证据 |
| `Related_Works` | 外部 baseline 与 reproduced runs | 已清掉非主线 ckpt；剩余主要是依赖和 placeholder |
| `Related_Works/results/metrics_summary` | baseline timing/metrics 汇总 | 保留，注意 smoke/unfair 行要降级 |
| `Related_Works/runs/cut_5x5` | CUT baseline 评测残留 | checkpoint 目标已复查清理 |
| `Related_Works/runs/cyclegan_5x5*` | CycleGAN baseline 评测残留 | checkpoint 目标已复查清理 |
| `final_works` | legacy final baseline metrics | 无剩余训练权重，保留 metrics |
| `Cycle-NCE` | 二三四月历史线索和报告 | 保留；下一步是 archive policy，不是 ckpt sweep |
| `lambda_grid` | dry-run sweep shell | 保留为负证据，禁止把 0.001s 当训练时间 |
| `step_count_sweep` | dry-run step sweep shell | 保留为负证据，禁止把 0.000s 当推理时间 |
| `efficiency` | profile json | 保留 |
| `fast_infer_ablate43` | fast inference 工具脚本 | 保留 |
| `latent_cyclegan` | legacy code/docs | 保留 |
| `exp` | 根目录少量 probe/log | 无权重 |
| `logs` | repo move/sync 操作日志 | 保留 |
| `archive` | 本地旧 cleanup archive | 保留，未展开删除 |
| `tmp` | 论文/PDF review scratch | 不属于实验清理对象 |
| `review_additional_experiments_aggregates` | review additional 汇总 CSV | 保留 |
| `o20_d3` | 旧 config/log/src root | 保留待 archive policy |
| `wikiart_fewshot` | fewshot 数据 | 保留 |
| `Plan_Docs` | 计划/状态文档 | 保留 |
| `PaperOrchestra-0.2.0` | 工具包 | 非实验输出 |

### 本地清理是否干净

已清理干净的 checkpoint 目标：

- `Related_Works/runs/cut_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5_smoke/checkpoints`
- `Related_Works/final_works/trial_0016`
- `Related_Works/final_works/trial_0019`
- `Related_Works/final_works/trial_0044`
- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync`

当前本地没有一个明确“又大、又非证据、又不被引用”的模型 ckpt 可以继续删。下一类可删对象不是 ckpt，而是：

- tiny `fake_eval_checkpoint.pt` placeholders：几乎不省空间，可能破坏 metadata；
- timing/generated images：需要逐个检查 docs 引用；
- old archives：需要 archive policy。

本次 continuation 做了一个安全的空目录清理：

- 删除 `SchrodingerBridge/exp/seedream_distill_adapter`；
- 删除 `SchrodingerBridge/exp/style_embedding_mainline_calibration`；
- 删除 `SchrodingerBridge/exp/tmp_genonly_autonogrid_probe`；
- 删除 `SchrodingerBridge/exp/vae_backend`。

这 4 个目录树删除前递归文件数都是 0，git 跟踪文件数都是 0，因此删除只减少空壳目录，不释放有效字节，也不影响实验证据。记录见 `cleanup/manual_empty_directory_cleanup_20260605.csv`。

## 当前远程 `I:\Github\Latent_Style` 状态

### 远程根目录结论

`I:\Github` 下有：

- `Latent_Style`：本任务主远程仓；
- `Latent_Style_TokenizerClean`：当前 AAAI/tokenizer 干净工作树；
- `26AI-H` / `26AI-H.zip`：不属于 Latent_Style，本轮不纳入清理。

### 远程主仓逐目录结论

| 远程目录 | 定位 | weight-like 状态 | 结论 |
|---|---|---:|---|
| `data` | legacy 数据 | 3422 / 218.828 MB | 数据，保留 |
| `style_data` | style/content 数据 | 8284 / 530.470 MB | 数据，保留 |
| `latents` | latent 数据 | 10361 / 663.374 MB | 数据，保留 |
| `latents_overfit50` | overfit50 latent | 100 / 1.713 MB | 数据，保留 |
| `latent-256` | legacy latent | 10361 / 177.702 MB | 数据，保留 |
| `latent-256-flux1` | backend latent cache | 10361 / 3900.374 MB | cache，保留 |
| `latent-256-flux2` | backend latent cache | 10361 / 5195.436 MB | cache，保留 |
| `latent-256-kl-f4` | backend latent cache | 10361 / 3899.171 MB | cache，保留 |
| `latent-256-kl-f4-mode` | backend latent cache | 10361 / 3899.171 MB | cache，保留 |
| `latent-256-sd15-ema` | backend latent cache | 10361 / 1310.264 MB | cache，保留 |
| `latent-256-sdxl` | backend latent cache | 10361 / 1310.264 MB | cache，保留 |
| `latent-256-sdxl-fp32` | backend latent cache | 10361 / 1310.264 MB | cache，保留 |
| `eval_cache` | remote eval deps/cache | 29 / 6077.946 MB | cache，保留 |
| `SchrodingerBridge` | current + historical LBM evidence | raw 11469 / 9441.059 MB | 必须拆分，不能按 raw count 删 |
| `SchrodingerBridge/exp` | 当前 Distinct5/SADD/path evidence | 101 / 5945.063 MB | 当前证据，保留 |
| `SchrodingerBridge/scale/datasets` | scale dataset tensors | 11349 / 2859.902 MB | 数据，不删 |
| `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0` | historical strict750 gate | 8 / 345.642 MB | 保留 |
| `SchrodingerBridge/review_additional_experiments` | historical review evidence | 9 / 289.800 MB | 保留到 archive policy |
| `Related_Works` | baseline evidence/deps | 27 / 5394.883 MB | 中央 SaMAM 曲线 + deps，保留 |
| `Related_Works/.../SaMAM.../step_checkpoints` | SaMAM Distinct5 曲线 | 19 / about 5242 MB | 当前 baseline 曲线，保留 |
| `Cycle-NCE` | historical archive/cache/deps | 37 / 937.553 MB | archive policy，不直接删 |
| `StarGAN` | baseline placeholders | 4 / 0.006 MB | tiny placeholder |
| `experiments` | Feb-Apr legacy archive | 3 / 319.141 MB | archive policy |
| `exp` | highres/probe root | 0 | 无 ckpt |
| `seedream45_api` | Seedream output | 1 / 0.00146 MB | tiny placeholder |

### 远程主仓清理是否干净

远程主仓的“非主线 checkpoint 清理”已经进入干净状态，原因是：

- remote SaMAM 已只剩中央 19 个 `step_checkpoints`；
- remote SchrodingerBridge/exp 剩下的是 current/lineage evidence，而不是未分类垃圾；
- remote 大体积 `.pt` 主要是 latent/cache/data。

还没清的是 archive/cache 层面，不是 checkpoint 层面：

- `latent-*` 后端缓存是否保留，需要决定哪些 backend 仍要复现；
- `Cycle-NCE` 与 `experiments` 的 rar/zip/旧目录是否保留，需要历史 archive policy；
- SaMAM 19 个 ckpt 是否 thinning 到 cited steps，需要确认比较文档是否还要完整曲线。

## 当前远程 `I:\Github\Latent_Style_TokenizerClean` 状态

### 结论

`TokenizerClean` 是当前 AAAI2027/tokenizer/Distinct5 干净工作树。它不能按“非主线远程目录”删。

打开过的关键文档：

- `docs/experiments/2026-06-03-exp-surface-classification.md`
- `docs/experiments/2026-06-03-timing-artifact-prune.md`
- `docs/experiments/2026-06-03-repo-cleanup-and-archive-pass.md`
- `docs/experiments/aaai2027_master_experiment_log.csv`
- `docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`
- `docs/experiments/2026-06-03-saswd-axis-ablation/README.md`
- `docs/experiments/2026-06-03-tokenizer-execution-alignment-l-family/README.md`

### TokenizerClean 权重分布

| root | count | size MB | 结论 |
|---|---:|---:|---|
| `SchrodingerBridge` total | 334 | 11822.873 | 当前 evidence + artifacts |
| `SchrodingerBridge/exp` normal | 326 | 11375.355 | tokenizer/WikiArt/AAAI chains |
| displayed `exp?saswd*` paths | 6 | 403.558 | SA-SWD semantic/random evidence，路径显示异常但是真证据 |
| `SchrodingerBridge/artifacts` | 1 | 43.635 | tokenizer artifact |
| `SchrodingerBridge/eval_cache` | 1 | 0.324 | eval cache |
| other top-level roots | 0 | 0 | mirror/tooling/docs |

### TokenizerClean 清理边界

不能删：

- `aaai2027_*` formal packets；
- `saswd_axis_*` semantic/random packet；
- `wikiart_distinct5_ema_lancet_spectralstat_*` 链；
- tokenizer carrier/factorized/direct atom chains；
- master log 和 docs 引用的 exp paths。

能进入下一轮策略讨论的只有：

- docs 已标记 retired/local-smoke 的 image payload；
- 未被 docs/master log 引用的 generated images；
- 被引用图谱迁移后的旧 probe directories。

当前不适合做 mass delete。

## 实验脉络总览

### 1. Feb-Mar：数据、latent、Cycle-NCE、legacy style transfer

目标是验证 latent/style-transfer 基础可行性，目录分布在：

- `style_data`
- `latent-256`
- `clip-feats-vitb32`
- `Cycle-NCE`
- remote `experiments`

这条线贡献了大量历史 metrics 和可视化，但 timing 稀疏，method 命名混乱，不能直接作为当前论文主 claim。

### 2. Mar-Apr：外部 baseline 和 final works

目标是建立 AdaIN/CUT/CycleGAN/SaMST/StarGAN/SDEdit 等 baseline 参考。

主要目录：

- `Related_Works`
- `final_works`
- remote `StarGAN`

结论：

- baseline evidence 有用；
- 剩余权重主要是依赖和 tiny placeholders；
- timing 行必须带 evidence grade，因为有 strict/smoke/failed/unfair 混合。

### 3. May：SchrodingerBridge/LANCET 大规模搜索

目标是找出 LBM/LANCET 的可用结构、loss、VAE backend、frontier。

主要目录：

- `SchrodingerBridge/exp/frontier`
- `SchrodingerBridge/exp/weight_sweep_40`
- `SchrodingerBridge/exp/kinetic_sweep`
- `SchrodingerBridge/exp/review_additional_experiments`
- `lambda_grid`
- `step_count_sweep`

关键纠偏：

- `lambda_grid` 和 `step_count_sweep` 当前根目录 manifest 是 `dry_run=true`；
- 里面 `0.001s` 只能作为 dry-run 证据，不能作为训练/推理时间。

### 4. May 30-Jun 2：WikiArt512 与 Distinct5 正式 evidence

目标变为更严格的 512 latent / 750 eval / timing claim。

本地可信路径：

- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`

远程可信路径：

- `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_*`
- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512...`

结论：

- WikiArt512 有正式 full-eval timing anchor；
- Distinct5 当前主证据集中在 LBM vs SaMAM/SaMST/no-op 的对照；
- SaMAM 的 19 个中央 ckpt 仍然是 baseline 曲线证据。

### 5. Jun 3 后：AAAI2027 / TokenizerClean / claim-closing packets

目标是围绕 Distinct5 的 theory/metric/tokenizer claim 做 formal packets。

主要目录：

- remote `Latent_Style_TokenizerClean/SchrodingerBridge/exp`
- remote `Latent_Style_TokenizerClean/SchrodingerBridge/docs/experiments`

结论：

- flow-loss 第一组三臂被 config audit 降级为 near-null；
- repaired endpoint metric packet 是负结果闭环；
- SA-SWD semantic/random packet 完成，但 random runtime 异常，只能 quality-only；
- tokenizer H-family packet 因 H e1 payload 缺失 blocked，L e1 是 successor family，不是同族 fallback。

## 训练/推理时间结论

强可信 timing 来源：

- `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `Related_Works/results/metrics_summary/timing_summary.csv`
- remote TokenizerClean `aaai2027_master_experiment_log.csv`
- remote TokenizerClean SA-SWD/endpoint metric docs

不能当 timing 的来源：

- root `run_summary.json` 的 dry-run rows；
- `lambda_grid/status.csv` 中没有真实训练支撑的 0.x/0.001s；
- runtime-anomalous SA-SWD random arm的速度。

缺口：

- DisDict 512 timing 未找到；
- 一些 legacy baseline 只有 metrics，缺完整 train/infer wall；
- SaMST 缺 matched time-to-parity curve；
- TokenizerClean H-family 缺 H e1 payload。

## 还能怎么继续清

当前最大限度安全清理的边界已经到 checkpoint 层了。下一步如果继续释放空间，应分三类，不要混在一起：

1. Evidence thinning：remote `SchrodingerBridge/exp` 每个 family 只留 cited/best epoch，其余打包或删。
2. Baseline thinning：SaMAM 19 个中央 ckpt 只留 cited steps + last，需要先确认 comparison docs。
3. Cache/archive policy：latent backend caches、remote `experiments`、`Cycle-NCE` rar/zips，要决定是否还能复现。

在没有这三条 policy 前，继续 broad delete 会误删证据或复现输入。

## 8 小时级别后续计划

如果继续完整推进，建议按下面顺序：

| block | 时间 | 目标 | 产物 |
|---|---:|---|---|
| 1 | 0.5h | 冻结当前证据图谱 | 当前 CSV/MD 提交，git status 干净范围确认 |
| 2 | 1.0h | local `SchrodingerBridge/exp` 全 family 逐目录分级 | per-family keep/archive/delete CSV |
| 3 | 1.0h | local `Related_Works` baseline 逐方法检查 | baseline dependency vs placeholder vs result index |
| 4 | 1.0h | remote `SchrodingerBridge/exp` 101 ckpt thinning policy | family -> keep epochs -> delete candidates |
| 5 | 1.0h | remote SaMAM 19 ckpt 曲线证据核对 | cited steps list + possible thinning ledger |
| 6 | 1.0h | TokenizerClean citation graph 检查 | docs/master log referenced path map |
| 7 | 1.0h | archive/cache 大户核对 | latent/cache/archive policy proposal |
| 8 | 0.5h | 执行已批准删除 + 校验 + commit | deletion CSV, post-delete count, commit |

当前这次 continuation 完成的是 block 1 的修正和归纳入口补齐，还没有执行 block 2-8 的破坏性清理。
