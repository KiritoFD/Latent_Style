# 文档材料映射表

更新时间：2026-04-09

这份表回答三个问题：
- 材料现在放在哪
- 它属于“原材料 / 中间稿 / 正式稿”哪一层
- 当需要写中期答辩时应该优先引用哪一份

## 1. 总体映射

| 材料类型 | 当前位置 | 用途 | 备注 |
| --- | --- | --- | --- |
| 历史考古旧草稿 | `raw/legacy/` | 保留早期判断、零散调查记录 | 不直接当正式结论 |
| 原 `docs` 文档树 | `raw/docs_old/docs/` | 保留旧报告、旧实验说明、快照分析 | 适合回查旧口径 |
| `docs_49` 模块专档 | `raw/gen49/docs_49/` | 模块级检索、局部实现说明 | 适合作为局部补充证据 |
| 根目录 PDF/PPT/HTML/DOCX 资产 | `raw/assets/` | 旧版汇报、可视化页面、成品材料 | 更偏展示素材 |
| 重建历史资料库 | `notes/2026-04-09_rebuilt_history/` | 当前最完整的中间工作稿 | 是正式报告的主要来源 |
| 中期答辩正式稿 | `reports/` | 直接汇报用 | 当前主文稿在本层 |

## 2. 重点材料与推荐用途

### 2.1 讲“模型怎么演进”

优先使用：
- [01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md)
- [02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md)
- [03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md)
- [04_ERA_D_CROSS_ATTN_TO_CGW.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/04_ERA_D_CROSS_ATTN_TO_CGW.md)
- [08_COMMIT_TIMELINE_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/08_COMMIT_TIMELINE_DETAILED.md)

### 2.2 讲“核心模块为什么变”

优先使用：
- [09_MODULE_STYLE_MODULATORS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/09_MODULE_STYLE_MODULATORS.md)
- [10_MODULE_SKIP_DECODER_AND_BACKBONE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/10_MODULE_SKIP_DECODER_AND_BACKBONE.md)
- [11_MODULE_TRAINER_AND_INFRA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/11_MODULE_TRAINER_AND_INFRA.md)
- [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)
- `raw/gen49/docs_49/`

### 2.3 讲“Loss 是怎么演化的”

优先使用：
- [05_LOSS_EVOLUTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/05_LOSS_EVOLUTION.md)
- [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)
- `raw/gen49/docs_49/loss_*.md`

### 2.4 讲“实验如何支撑这些设计”

优先使用：
- [06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md)
- [12_EXPERIMENT_GROUP_EXP_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/12_EXPERIMENT_GROUP_EXP_DETAILED.md)
- [13_EXPERIMENT_GROUP_ABLATION_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/13_EXPERIMENT_GROUP_ABLATION_DETAILED.md)
- [14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md)
- [17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md)
- [18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md)
- [20_EXPERIMENT_CURVE_ARCHETYPES.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/20_EXPERIMENT_CURVE_ARCHETYPES.md)

## 3. 适合中期答辩直接引用的材料

最推荐直接作为“答辩底稿”的是：
- [2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)

适合答辩问答时回查的支撑文档：
- [08_COMMIT_TIMELINE_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/08_COMMIT_TIMELINE_DETAILED.md)
- [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)
- [18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md)
- [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)

## 4. 当前资料库的结构判断

这套资料库现在已经形成了比较清楚的三层逻辑：

1. 原材料层：保留历史痕迹，不追求统一口径。
2. 工作笔记层：把 git、实验目录、旧文档拉通，形成可追溯的分析。
3. 正式报告层：面向答辩、汇报、论文写作，强调高层逻辑和可讲述性。

后续如果再加入新材料，建议也沿这个逻辑放置，避免重新回到“所有东西都堆在一个目录”的状态。
