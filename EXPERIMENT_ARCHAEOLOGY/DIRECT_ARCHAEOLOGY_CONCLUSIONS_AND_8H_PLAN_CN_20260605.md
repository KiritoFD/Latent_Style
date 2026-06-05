# 直接考古结论与 8 小时继续计划 - 2026-06-05

本文是当前最直接的结论层。它不把脚本扫描当成完成证明，只把已经有
CSV/MD、固定路径打开、删除 ledger、post-delete verify 的内容作为结论证据。

当前判断：任务未完成，但已经有可审计的主干脉络、索引、清理记录和下一步
清理边界。

## 1. 当前事实

| 项 | 当前值 | 证据 |
| --- | ---: | --- |
| master experiment rows | 22629 | `final_master_experiments.csv` |
| timeline rows | 7829 | `final_timeline.csv` |
| dataset/setting splits | 25 | `conclusions_by_dataset.csv`; `final_by_dataset/*.csv` |
| docs timing master rows | 419 | `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` |
| timing quality overlay rows | 1093 | `timing_quality_master_20260605.csv` |
| current archaeology output files | 237 top-level files | local shallow listing of `EXPERIMENT_ARCHAEOLOGY` after adding this report and status CSV |

当前 Git 状态：

- `EXPERIMENT_ARCHAEOLOGY` 在本报告写入前是干净的。
- 仓库仍有用户/其他线程脏文件，集中在 `Related_Works` scripts、SaMST repo、
  `SchrodingerBridge/aaai_submission`、`SchrodingerBridge/src` 和少量 untracked
  paper/config snapshot。它们没有被本考古任务修改、回滚或提交。

## 2. 本地结论

本地 `G:\GitHub\Latent_Style` 不是一个实验目录，而是混合证据工作区。

| local path | shallow state | 当前解释 |
| --- | --- | --- |
| `G:\GitHub\Latent_Style` | 35 dirs / 39 files | 根工作区，含数据、缓存、主实验、baseline、历史面、paper scratch |
| `SchrodingerBridge` | 17 dirs / 15 files | 当前 LANCET/LBM 主实验、docs/config/src/exp |
| `SchrodingerBridge\exp` | 110 dirs / 4 files | 本地当前和历史 LANCET/LBM 输出面 |
| `Related_Works` | 9 dirs / 1 file | baseline/reproduction/results 面 |
| `Cycle-NCE` | 39 dirs / 58 files | 历史 Cycle-NCE/AdaCUT 证据面 |
| `EXPERIMENT_ARCHAEOLOGY` | 3 dirs / 235 files | 当前考古输出面 |

本地已经明确的保留/不删边界：

- `archive/tmp/paper scratch` 已单独复核 5 个固定路径，全部
  `delete_whitelist=no`。`archive` 是旧清理历史，`tmp` 和 paper snapshot 是
  paper/PDF scratch，`aaai_submission` 是 active tracked dirty paper workspace。
- `seedream_gap` 与 quick/fine inference sweeps 已有 owner manifest：29 行，
  10750 JPG，142.724 MB，全部 `retain_pending_owner`。
- CUT `raw_results/raw_results_val` 已逐 target 打开：14 行 policy，1218.627 MB，
  训练日志记录已提取，推理 wall time 缺失留空。结论是 tracked/mixed tracked
  boundary，不能直接删。
- `Related_Works` 和 `final_works` 是 baseline evidence，不再按“图片很多”或
  “看起来是中间结果”删除。

本地还没有完成：

- `seedream_gap`、quick/fine sweeps 需要 owner 选择 keep/migrate/delete。
- `archive/tmp/paper scratch` 如果要清理，必须开单独 paper-scratch whitelist。
- CUT raw 如果要删除，必须先做 tracked-file migration/untracking policy。
- `final_by_dataset/*.csv` 还需要逐数据集写成可读中文结论，不能继续使用旧乱码字段。

## 3. 远程主树结论

远程主树不是空盘。

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
I:\Github\Latent_Style
```

| remote path | current state | 当前解释 |
| --- | --- | --- |
| `I:\Github\Latent_Style` | 23 dirs / 53 files | 远程主树存在且非空 |
| `SchrodingerBridge` | 32 dirs / 175 files | 远程 LANCET/LBM、review、archive、src/config/docs 混合面 |
| `SchrodingerBridge\exp` | 123 dirs / 1 file | epoch thinning 后仍有大量 current/formal anchors |
| `Related_Works` | 5 dirs / 0 files | 远程 baseline/source/dependency 面 |
| `Related_Works\runs` | only `cut_5x5` | 比本地更干净，但仍有 qualitative media |
| `Cycle-NCE` | 26 dirs / 78 files | 历史 Cycle-NCE 面仍在 |

远程主树已清理且验证的重点：

- `SchrodingerBridge\exp` epoch thinning 已删除 84 个 checkpoint，保留 17 个
  anchors。
- SaMAM alias cleanup 已删除 7 个 redundant aliases，保留 12 个 step checkpoints。
- data/cache/archive residue、duplicate/stale archives、weight-only RAR、
  `experiments.rar`、`Cycle-NCE\45.rar` 已按 exact whitelist 清理并 post-verify。
- `Cycle-NCE\45.rar` 当前 absent；curated nonweight package 当前 present，6086
  files，145.512 MB，0 weight-extension files。

远程主树当前不能删的重点：

- `SchrodingerBridge\review_additional_experiments.rar` 存在，约 2991.423 MB。
  同名解压目录存在且有 58151 files、1270.619 MB、9 weights、77 summaries、9
  training CSVs。但远程当前没有 `7z/rar/unrar` 可可靠列 RAR 内容，所以不能声明
  archive 已完全被目录替代，不能删。
- `Related_Works\runs\cut_5x5\infer_5x5` 有 2427 JPG 和 1 个 1531-byte
  `fake_eval_checkpoint.pt`，没有 summary/metrics/meta。它是 qualitative media，
  不是正式指标证据，是否删除仍需 owner 选择和代表样本保留。
- `baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`
  是大包：12128 files，7232.586 MB，12 ckpt，24 metric-like files。它是 SaMAM
  diagnostic/curve anchor，不能按大小删。

## 4. 远程 TokenizerClean 结论

`I:\Github\Latent_Style_TokenizerClean` 是独立 evidence surface，不能并入远程主树
一起概括。

| path/surface | current state | 当前解释 |
| --- | --- | --- |
| root | 17 dirs / 37 files | 独立 TokenizerClean tree |
| `SchrodingerBridge\exp` | 142 dirs / 23 files | 清理后仍有 current/cited/tokenizer evidence |
| cited/current media manifest | 26 dirs, 46483 media files, 11977.341 MB, 118 weights | 全部等待 owner archive/migration |
| training-log-only live recheck | 7 dirs, 3 remaining weights, 130.883 MB | 5 个 metadata-only，2 个 evidence-bearing |
| post-delete verify | 20 checks pass | 已删除的 7 个 training-log-only weights 闭环 |
| missing resume anomaly | resume target `epoch_0004.pt` absent | 只能作为 metadata-only archaeology |

TokenizerClean 已清理：

- 141 uncited checkpoints。
- 18 no-summary probe checkpoints。
- 3 orphan probe dirs/weights。
- 43008 uncited generated media files。
- 7 training-log-only checkpoint weights。

TokenizerClean 不能继续盲删：

- 26 个 cited/current media dirs 需要 owner archive/migration 选择。
- 3 个 remaining weights 是 evidence-bearing 或 downstream-resume payload。
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` 的 resume anomaly 不能作为
  clean lineage 或 evaluated result 宣传。

## 5. 实验脉络

当前实验史应按阶段理解：

1. 2026-02 到 2026-03：legacy style-transfer、no-edge、no-tokenized、
   overfit50、IDT/no-op sanity。主要作为早期 baseline 和负例。
2. 2026-03 到 2026-04：Cycle-NCE、Latent AdaCUT、CUT、CycleGAN、StyleID、
   SDEdit、StarGAN 等旧 baseline 形成历史证据面。
3. 2026-05：SchrodingerBridge/LANCET 大规模探索，包括 grid/search、weight
   sweep、frontier、VAE backend、representation/tokenizer probe。
4. 2026-05-30 到 2026-06-02：WikiArt512 与 Distinct5-512 formal/full-eval/timing
   证据成型。
5. 2026-06-03 之后：AAAI2027/TokenizerClean closing surface，用于 tokenizer、
   endpoint、flow-loss、time-to-parity、localization 等 claim closing。

不能混淆的类别：

- dry-run / smoke / failed probe / no-op control / diagnostic media / historical
  baseline / current formal evidence 不是同一种结果。
- 当前 claim 应优先使用 Distinct5-512、WikiArt512、strict protocol 750、docs
  timing master 和 source-open timing rows。

## 6. 数据集与 setting 分布

`conclusions_by_dataset.csv` 拆出 25 个 dataset/setting 面。核心分布如下：

| dataset_key | rows | local | remote | metric | train timing | infer timing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cycle_nce` | 11794 | 10590 | 1204 | 10590 | 0 | 0 |
| `schrodingerbridge_exp_general` | 4051 | 3138 | 913 | 3239 | 12 | 36 |
| `schrodingerbridge_weight_sweep` | 1285 | 811 | 474 | 800 | 11 | 0 |
| `legacy_style_transfer_experiments` | 1120 | 0 | 1120 | 0 | 0 | 0 |
| `schrodingerbridge_grid_search` | 1013 | 544 | 469 | 544 | 0 | 0 |
| `schrodingerbridge_vae_backend` | 699 | 81 | 618 | 81 | 0 | 0 |
| `schrodingerbridge_frontier` | 692 | 600 | 92 | 600 | 0 | 0 |
| `schrodingerbridge_representation_probe` | 567 | 503 | 64 | 471 | 0 | 0 |
| `distinct5_512` | 417 | 305 | 112 | 278 | 55 | 113 |
| `wikiart512_5style` | 200 | 108 | 92 | 144 | 14 | 10 |
| `strict_protocol_750` | 79 | 77 | 2 | 53 | 15 | 24 |
| `photo_monet_5x5` | 42 | 17 | 25 | 0 | 0 | 2 |

后续必须逐个打开 `final_by_dataset/*.csv` 写中文解释，尤其是：

- 哪些是历史上下文。
- 哪些是 formal claim。
- 哪些只有 timing 或 qualitative evidence。
- 哪些只能作为 audit/smoke/negative result。

## 7. Timing 结论

Timing 不是空白，但还不是最终 paper-facing 表。

| timing class | rows | 用途 |
| --- | ---: | --- |
| `full_eval_summary_wall_time_tokenizerclean` | 744 | audit full-eval wall time only |
| `quick_eval_or_probe_wall_time` | 234 | audit only |
| `full_eval_wall_time` | 51 | candidate claim support with caveat |
| `historical_timing_context` | 28 | historical context |
| `partial_training_or_missing_eval` | 20 | audit only |
| `smoke_or_failed_probe` | 7 | exclude formal claim |
| `training_log_only` | 2 | audit training cost only |
| `train_and_eval_wall_time` | 2 | candidate claim support with caveat |

原则：

- 保留原始单位。
- 训练时间不强转秒。
- 缺失推理时间留空。
- 区分 train wall time、full-eval wall time、pure inference/generation time、
  smoke/audit-only。

已补新增 CUT timing required-fields CSV：

- `manual_local_cut_raw_timing_required_fields_20260605.csv`

仍缺：

- 370 条 docs master rows 还缺 overlay/source-open 覆盖。
- 需要 owner 选择哪些 rows promotion 到最终标定复用表。

## 8. 清理结论

已经执行的清理遵守同一个规则：

```text
exact path -> policy CSV/MD -> deletion ledger -> post-delete verification
```

不能按扩展名、大小、图片数量、hash equality 做 broad delete。

当前 cleanup ledger synthesis 记录：已释放约 `93020.641 MB`。该数字不是
“任务完成”证明，只说明已执行 cleanup blocks 的汇总。

重点已闭环清理包括：

- local likely non-mainline checkpoint cleanup。
- local eval/dataset/cache/root archive residue cleanup。
- local CUT video work-frame dirs。
- local generated-media duplicate cleanup。
- remote SchrodingerBridge epoch thinning。
- remote SaMAM alias cleanup。
- remote data/cache/archive residue cleanup。
- remote RAR and `experiments.rar` cleanup。
- remote `Cycle-NCE\45.rar` curated extraction 后删除。
- remote TokenizerClean uncited checkpoints/no-summary probes/orphan probes/
  uncited generated media/training-log-only weights。

当前不能继续清理的边界：

- paper TeX/PDF 和 active paper workspace。
- 当前 dirty source/code files。
- local retained generated media without owner choice。
- remote `review_additional_experiments.rar` without archive listing proof。
- remote CUT qualitative media without owner choice。
- TokenizerClean cited/current media without archive/migration choice。
- cache duplicate without loader/path-reference/migration proof。

## 9. 8 小时继续计划

### 第 1 小时：把结论层补齐并提交

- 输出本文和 requirement-status CSV。
- README 和 current/grand index 指向新的清晰结论。
- 提交只包含 `EXPERIMENT_ARCHAEOLOGY`。

### 第 2 小时：dataset-by-dataset 中文重写

- 逐个打开 25 个 `final_by_dataset/*.csv`。
- 对每个 dataset/setting 写：时期、方法、指标、timing、claim 使用边界。
- 替换旧乱码 conclusion/gap 的可读层，不删除原始 CSV。

### 第 3 小时：timing promotion

- 从 1093 overlay、419 docs master、26 source-open candidates、CUT timing rows
  中挑出可复用标定 rows。
- 输出 `timing_promotion_candidates_20260605.csv` 和中文说明。
- 保留原始单位和缺失值。

### 第 4 小时：remote archive proof

- 针对 `SchrodingerBridge\review_additional_experiments.rar` 获取可靠 RAR
  listing 或受控解包 proof。
- 只有证明非权重/指标证据已保留，才写 delete whitelist。
- 没有 proof 就继续 retain。

### 第 5 小时：owner media migration policy

- local `seedream_gap`、quick/fine sweeps。
- remote TokenizerClean 26 cited/current media dirs。
- 输出 keep/migrate/delete 选项和 path-preserving archive policy。
- 没有 owner 选择不删除。

### 第 6 小时：cache duplicate 和 tiny probe exact review

- cache duplicates 只做 path-reference/migration proof，不按 hash 删。
- tiny empty/probe dirs 可列 candidate，但只有 owner 批准才删。

### 第 7 小时：最终一致性审计

- 校验 README、direct index、authoritative index、cleanup summary、dataset split、
  timing tables 数字一致。
- 检查所有 delete whitelist 都有 ledger 和 post-delete verify。

### 第 8 小时：完成性审计与提交

- 逐条对照用户要求：本地、远程、每个数据集、timing、cleanup、文档、提交。
- 只有每条都有当前证据时才能声明完成。
- 否则保持 active，并列出剩余 exact-path queue。

## 10. 当前完成性判断

未完成。

已经完成的是：索引、时间线、数据集拆分、多个本地/远程 exact-path 清理与复核、
本地 CUT tracked-boundary policy、远程 main surface recheck、远程 TokenizerClean
surface recheck、以及本结论层。

还没完成的是：每个剩余目录的 owner 决策、dataset-by-dataset 可读中文结论、
timing promotion、remote archive proof、cache migration proof 和最终一致性审计。
