# 当前实验考古状态与结论 - 2026-06-05

这份文档是当前可读的总归纳入口。它不宣称整个任务已经完成；它把已经逐项打开并有证据支撑的结论、已经执行的清理、仍然不能删除的内容、以及下一步 8 小时级别计划写清楚。

写入范围仍限于 `EXPERIMENT_ARCHAEOLOGY`。论文 TeX/PDF、源码、Related_Works 脏文件没有被修改、stage 或回滚。

## 结论先行

当前仓库不是一个单一实验目录，而是三块证据面叠在一起：

| 区域 | 当前结论 | 证据入口 | 当前状态 |
| --- | --- | --- | --- |
| 本地 `G:\GitHub\Latent_Style` | 混合考古面：SchrodingerBridge 当前实验、Related_Works baseline、Cycle-NCE 历史、数据/latent/cache、root archive/tmp、论文工作区同时存在。 | `manual_coverage_matrix_20260605.csv`, `manual_top_level_directory_index_20260605.csv`, `manual_local_*_policy_20260605.csv` | 已做 top-level 和主要 family 手动覆盖；nested generated media 和 archive provenance 仍未全收口。 |
| 远程主树 `I:\Github\Latent_Style` | SchrodingerBridge/exp、SaMAM、data/cache/archive、RAR、Cycle-NCE 已经做过多轮手动分类和白名单清理。 | `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md`, `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md`, `MANUAL_REMOTE_RAR_DEEP_PROVENANCE_20260605.md` | checkpoint、失败缓存、重复 archive、weight-only RAR 清理基本完成；`45.rar` 仍是唯一历史证据包，不能直接删。 |
| 远程 TokenizerClean | 145 个 exp 顶层目录已建 internal evidence/citation graph；uncited ckpt、probe、zero-hit media 已清；cited/current/no-summary payload 保留。 | `MANUAL_REMOTE_TOKENIZERCLEAN_CITATION_GRAPH_20260605.md`, `MANUAL_REMOTE_TOKENIZERCLEAN_CITED_CURRENT_MEDIA_POLICY_20260605.md`, `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_OWNER_DECISION_20260605.md` | 已完成清理策略分层；5 个 trained no-summary payload 仍是 training-log-only，需要 summary recovery 或 owner decision。 |
| timing | 已有 docs timing master、1093 行 archaeology overlay、26 行 missing-docs candidate source-open 表。 | `TIMING_MASTER_RECONCILIATION_20260605.md`, `TIMING_CANDIDATE_MISSING_DOCS_SOURCE_OPEN_20260605.md` | 26 个缺口已逐条打开；docs master 未改；370 个 docs rows 仍缺 overlay/source-open 覆盖。 |
| 清理 | 已执行的删除都有 policy、ledger、post-delete verify；没有按扩展名/大小泛删。 | `cleanup/CLEANUP_AUDIT_SUMMARY.md`, `cleanup/*.csv` | 当前可核算释放约 `92162.847 MB`，但仍有保留项，不等于“仓库清空”。 |

## 本地结论

本地根目录当前仍包括 `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `eval_cache`, `SchrodingerBridge`, `Related_Works`, `Cycle-NCE`, `archive`, `tmp`, `exp`, `seedream45_api`, `final_works`, `lambda_grid`, `step_count_sweep` 等目录。当前顶层已重新列出确认；这些不是同一种清理对象。

| 本地区域 | 当前理解 | 已做清理 | 保留原因 | 仍缺 |
| --- | --- | --- | --- | --- |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale`, `horse2zebra` | 数据、latent、CLIP feature、VAE 依赖，不是训练 ckpt。 | 删除失败 HF dataset cache `63.948 MB`。 | 仍被训练/评估依赖；代表性 tensor 已只读打开确认类型。 | 只做 retention policy，不应按 `.pt` 泛删。 |
| `eval_cache` | ArtFID/CLIP/VAE/DINO/offline pairing/ref feature cache 混合依赖。 | 删除 invalid `.incomplete` 和空 ModelScope temp，`55.994 MB`。 | 完整 eval/model/speed cache 保留。 | cross-cache 需要 canonical root/symlink 迁移后才能删重复项。 |
| `SchrodingerBridge/exp` | 当前 LANCET/LBM 证据和 formal eval 根。 | 本地非主线 ckpt 已按 ledger 清理；正式 WikiArt512 与 ArtFID 依赖保留。 | 当前论文/实验 timing 和 metric anchor。 | 新增实验后仍需 epoch thinning policy。 |
| `Related_Works` | baseline pipeline、SaMST、SaMAM、CUT/CycleGAN 等 baseline 证据面。 | 非主线 ckpt 和部分中间帧已清。 | baseline log/metric/protocol 仍是对比证据。 | nested generated media 还要 owner-level review。 |
| `Cycle-NCE` | 早期历史与生成/评估证据。 | 空 eval_cache/HF 残留、本地 duplicate tar 已清。 | 历史 metrics/summary/video/archives 有证据价值。 | archive policy 不能用 checkpoint sweep 替代。 |
| root `archive`, `tmp`, `exp` | 混合历史 archive、论文 scratch、失败 launcher residue、generated image evidence。 | 删除 duplicate `Cycle-NCE.tar`、stale launcher residue、空 probe dir。 | `tmp` 里包含近期 paper/PDF/TEX/PNG scratch，当前边界不允许碰论文文件。 | 需要单独 paper-temp/archive policy。 |

本地结论：已经不是“没看”。但不能说“每个 nested 目录都已 owner-level 收口”。当前可信表述是：top-level 和主要 evidence family 已手动覆盖，清理只发生在白名单 ledger 内，剩余本地缺口集中在 nested generated media 和 archive/temp provenance。

## 远程主树结论

远程顶层当前确认有：

```text
I:\Github\26AI-H
I:\Github\26AI-H.zip
I:\Github\Latent_Style
I:\Github\Latent_Style_TokenizerClean
```

主线考古对象是 `I:\Github\Latent_Style` 和 `I:\Github\Latent_Style_TokenizerClean`，不是把整个 I 盘当同一种实验目录。

| 远程主树区域 | 当前理解 | 已清理 | 保留原因 | 仍缺 |
| --- | --- | --- | --- | --- |
| `I:\Github\Latent_Style\SchrodingerBridge\exp` | 远程 Distinct5/SADD/phase-space/AAAI2027 证据面。 | 删除 84 个非保留 epoch ckpt，释放 `4961.604 MB`。 | 保留 17 个 anchor/probe/cited epochs。 | 只可按新 policy 继续 thinning。 |
| SaMAM central `step_checkpoints` | SaMAM 曲线和修复证据。 | 删除 7 个 redundant `last*.ckpt` alias，释放 `1931.291 MB`。 | 保留 12 个 step checkpoints，覆盖 curve/cited/repair/last roles。 | 不能破坏曲线证据。 |
| data/cache/archive residue | 失败缓存、lock/tmp、完整数据/cache 混合。 | 删除 11 个失败/残留 target，释放 `381.807 MB`。 | 完整 data、latent、eval_cache、baseline repo 保留。 | cache duplicate 仍需 loader/path migration proof。 |
| duplicate/stale archives | eval_cache.zip、legacy checkpoint zip、exact duplicate archive 等。 | 删除 3 个 archive，释放 `3290.714 MB`。 | 已确认保留 evidence root 仍存在。 | 不能扩展成按 archive 大小泛删。 |
| RAR provenance | Gate/Attn/chess/experiments/45.rar 分开处理。 | 删除 weight-only RAR `6553.384 MB`；删除 resolved duplicate `experiments.rar` `8091.026 MB`。 | `45.rar` 含 configs、summaries、metrics、6008 images、ma-probe、weights，仍是唯一历史包。 | 若要删 `45.rar`，必须先抽取 curated nonweight evidence。 |

远程主树结论：主树已经有大量清理和 post-delete verification，但 `45.rar` 与 cache duplicate 不是垃圾；它们是后续 archive/migration 任务，不是当前可删项。

## 远程 TokenizerClean 结论

TokenizerClean 是单独证据面，不能和远程主树混为一谈。

| TokenizerClean 区域 | 当前理解 | 已清理 | 保留原因 | 仍缺 |
| --- | --- | --- | --- | --- |
| 145 个 exp 顶层目录 | 已建 internal evidence 和 citation graph。 | 删除 141 个 uncited exploratory checkpoints，释放 `5198.991 MB`。 | cited/current/formal packet 保留。 | 仍需最终 README/count consistency。 |
| 28 个 no-summary dirs | 已分为 probe/calibration、pure orphan probe、trained payload。 | 删除 18 个 probe/calibration ckpt `362.391 MB`；删除 3 个 pure orphan probe dirs/weights `170.017 MB`。 | 7 个 trained payload 有 config/training/source/weights。 | 5 个 training-log-only 仍需 recovered summary 或 owner delete approval。 |
| generated media | zero-hit summary-backed media 已清。 | 删除 43008 个 uncited generated media，释放 `11883.246 MB`。 | 26 个 cited/current media dirs 已逐个打开，合计 46483 files / 7501.518 MB，全部 `keep_no_delete`。 | 需要 citation-to-artifact manifest 和 optional archive tar。 |
| special path | SA-SWD semantic 成功证据目录显示为 `exp?saswd...`，但实际字符不是 ASCII `?`。 | 无删除。 | 真实目录分隔字符码为 61532，必须通过目录对象路径打开。 | 后续脚本要避免把 console `?` 当 literal path。 |

TokenizerClean 结论：已经清理的是 uncited/probe/zero-hit 层；真正 current/cited/no-summary payload 目前按证据保留。下一步不是继续删，而是做 summary recovery、owner decision 和 archive manifest。

## Timing 结论

| timing 面 | 当前事实 |
| --- | --- |
| docs timing master | 419 rows，未在本轮修改。 |
| archaeology timing overlay | 1093 rows，有质量标签。 |
| claim candidate rows | 53 rows；其中 27 已在 docs master，26 不在 docs master。 |
| missing-docs source-open | 26/26 已逐条打开，产物为 `timing_candidate_missing_docs_source_open_20260605.csv`。 |
| docs rows lacking overlay | 370 rows，仍需 source-open 后才能进 prose/claim。 |

Timing 结论：已经把训练/推理时间的“候选缺口”挖出来并逐条开源核验，但这还不是最终 paper-facing timing table。后续需要 owner 选择哪些行进入 docs timing master，并单独更新 docs 表。

## 实验脉络

当前脉络按阶段理解如下：

| 阶段 | 时间 | 主线 | 解释 |
| --- | --- | --- | --- |
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/Cycle-NCE | 早期 style-transfer、sanity、失败/历史实验面。 |
| Phase B | 2026-03 到 2026-04 | legacy256、StyleID、IDT、tokenized/no-tokenized | baseline 和 sanity 面，很多 method/dataset 仍需补标签。 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 历史大指标面，保留 metrics/summary/ref cache。 |
| Phase D | 2026-05 | SchrodingerBridge/LANCET sweeps | grid/search/frontier/vae_backend/representation，主要是探索面。 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 和 Distinct5 formal evidence | 当前 timing/efficiency claim 的核心证据面。 |
| Phase F | 2026-06-03 起 | AAAI2027 / TokenizerClean claim closing | flow-loss、SA-SWD、tokenizer execution、time-to-parity，仍需 review-grade 整理。 |

Dataset 层面，当前 `final_by_dataset/` 和 `conclusions_by_dataset.csv` 已经把证据拆到 dataset/setting 级别；核心 claim 面是 `distinct5_512`, `wikiart512_5style`, `strict_protocol_750`。`cycle_nce`, `legacy_style_transfer_experiments`, `legacy256_overfit50` 更偏历史/负例/背景；`schrodingerbridge_grid_search`, `weight_sweep`, `vae_backend`, `representation_probe` 更偏探索轨迹。

## 清理总账

清理原则是：只按 policy CSV + per-file ledger + post-delete verification 删除；不按扩展名、大小、目录名批量清。

当前可核算释放空间约 `92162.847 MB`。重要块包括：

| 清理块 | 释放 |
| --- | ---: |
| 本地非主线 checkpoint 初筛 | 11575.670 MB |
| 本地 root misc/cache/dataset/remaining surface/generated frame 等后续手动块 | 约 49338.376 MB 本地总账内 |
| 远程 SchrodingerBridge epoch thinning | 4961.604 MB |
| 远程 SaMAM alias cleanup | 1931.291 MB |
| 远程 data/cache/archive residue | 381.807 MB |
| 远程 duplicate/stale archives | 3290.714 MB |
| 远程 RAR weight-only archives | 6553.384 MB |
| 远程 resolved duplicate `experiments.rar` | 8091.026 MB |
| 远程 TokenizerClean checkpoints/probes/media | 17614.039 MB |

这些数字的权威入口是 `cleanup/CLEANUP_AUDIT_SUMMARY.md` 和各 `cleanup/*.csv`。旧 README 中某些中间 count 可能早于后续 pass；最终一致性审计仍需做。

## 仍未完成

当前不能宣称完成，原因很具体：

- 本地 nested generated media 还没有全部 owner-level 决策。
- 本地 archive/temp/paper scratch 仍需单独 provenance policy；当前不碰 TeX/PDF。
- 远程 `Cycle-NCE\45.rar` 需要先抽取 curated nonweight evidence 才能考虑删除。
- cache duplicate 已做 hash 和 loader/path audit，但删除仍需 canonical cache-root migration、symlink/junction policy 和 offline eval verification。
- TokenizerClean 5 个 trained no-summary payload 仍是 training-log-only，缺 recovered summary 或 owner delete approval。
- TokenizerClean 26 个 cited/current media dirs 只做了 keep/no-delete policy，尚未创建 archive tarball 和 citation manifest。
- docs timing master 还没更新；370 个 docs timing rows 仍需 source-open。
- `final_by_dataset`, `final_timeline`, `README`, cleanup totals 需要最终一致性审计。

## 8 小时级别后续计划

| block | 预算 | 目标 | 产物 |
| --- | ---: | --- | --- |
| 1 | 1.0h | 本地 nested generated-media owner review 继续 | owner policy CSV/MD，可能的 whitelist cleanup |
| 2 | 1.0h | `45.rar` curated nonweight extraction policy | entry class table、extract manifest、delete/retain decision |
| 3 | 1.0h | TokenizerClean 5 个 training-log-only payload summary recovery/owner decision | recovered summary 或 owner decision CSV |
| 4 | 1.0h | cited/current media archive manifest | citation-to-artifact manifest，代表图清单 |
| 5 | 1.0h | docs timing master promotion planning | owner-selected timing rows，docs update plan |
| 6 | 1.0h | 370 docs timing rows source-open sampling/triage | docs timing coverage table |
| 7 | 1.0h | dataset split、timeline、README count consistency | consistency audit CSV/MD |
| 8 | 1.0h | 最终 requirement-by-requirement completion audit | updated completion audit，最终 gap list 或 completion proof |

## Git hygiene

当前提交只发生在 `EXPERIMENT_ARCHAEOLOGY`。全局仍有用户/主线脏文件，包括 Related_Works 脚本、SchrodingerBridge paper TeX/PDF、`src/run.py`、`src/trainer.py`、本地 config snapshots 等；这些文件没有被 stage 或回滚。
