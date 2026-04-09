# Cycle-NCE 历史重整理目录

状态：第一手资料草稿  
整理时间：2026-04-09  
整理依据：

- `C:\Users\xy\repo.git` 中的完整 git 历史
- 当前工作区 `G:\GitHub\Latent_Style`
- `Y:\experiments` 下的实验目录、`EXPERIMENT_RECORD_FULL_DATA.csv`、`RESULTS_INDEX_20260330.csv`
- 旧草稿文档：`Cycle-NCE/his_doc/*.md`

这套文档的目标不是“写成一篇短总结”，而是把后续真正能引用的材料先拆开、归档、建立索引。

## 建议阅读顺序

1. `00_METHOD_AND_INDEX.md`
2. `01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md`
3. `02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md`
4. `03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md`
5. `04_ERA_D_CROSS_ATTN_TO_CGW.md`
6. `05_LOSS_EVOLUTION.md`
7. `06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md`
8. `07_OPEN_QUESTIONS_AND_RECOVERY_NOTES.md`
9. `08_COMMIT_TIMELINE_DETAILED.md`
10. `09_MODULE_STYLE_MODULATORS.md`
11. `10_MODULE_SKIP_DECODER_AND_BACKBONE.md`
12. `11_MODULE_TRAINER_AND_INFRA.md`
13. `12_EXPERIMENT_GROUP_EXP_DETAILED.md`
14. `13_EXPERIMENT_GROUP_ABLATION_DETAILED.md`
15. `14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md`
16. `15_CONFIG_NAMING_AND_SEARCH_SPACE.md`
17. `16_SOURCE_EVIDENCE_DIFFS.md`
18. `17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md`
19. `18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md`
20. `19_MODULE_COMMIT_MATRIX.md`
21. `20_EXPERIMENT_CURVE_ARCHETYPES.md`

## 目录说明

- `00_METHOD_AND_INDEX.md`
  - 说明这次整理怎么做，哪些是直接证据，哪些是推断。
- `01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md`
  - 从 `Thermal` / `NCE_SWD` 到 `Cycle-NCE` 早期可运行原型。
- `02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md`
  - 从“必须带 style reference”到“style 蒸馏进模型”的关键转折。
- `03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md`
  - 以 `TextureDictAdaGN`、skip 过滤、decoder 轻重化为主线。
- `04_ERA_D_CROSS_ATTN_TO_CGW.md`
  - cross-attn、feature attention、CGW backbone、micro-batch。
- `05_LOSS_EVOLUTION.md`
  - 每一代 loss 的实现、保留原因、被淘汰原因。
- `06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md`
  - 把 `Y:\experiments` 的实验组和对应架构阶段连起来。
- `07_OPEN_QUESTIONS_AND_RECOVERY_NOTES.md`
  - 当前还没完全坐实、后续可以继续挖的点。
- `08_COMMIT_TIMELINE_DETAILED.md`
  - 关键提交按时间顺序展开，适合后面做更严的历史追溯。
- `09_MODULE_STYLE_MODULATORS.md`
  - `AdaGN` / `TextureDictAdaGN` / `CrossAttnAdaGN` 专档。
- `10_MODULE_SKIP_DECODER_AND_BACKBONE.md`
  - skip、decoder、conv/global/window/CGW 骨架演化。
- `11_MODULE_TRAINER_AND_INFRA.md`
  - trainer、mixed precision、checkpointing、micro-batch 与评估缓存。
- `12_EXPERIMENT_GROUP_EXP_DETAILED.md`
  - 主线实验组细化记录。
- `13_EXPERIMENT_GROUP_ABLATION_DETAILED.md`
  - A 系列与 `abl_*` 结构消融的整理。
- `14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md`
  - color 路线和 style_oa 联合优化路线。
- `15_CONFIG_NAMING_AND_SEARCH_SPACE.md`
  - 当前 `config_*.json` 命名簇和后期搜索空间备忘。
- `16_SOURCE_EVIDENCE_DIFFS.md`
  - 关键提交之间的源码差异证据摘要。
- `17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md`
  - `summary_history` 视角下的 epoch 甜点与曲线结论。
- `18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md`
  - 按 style、FID、art-FID 和启发式均衡分数整理实验榜单。
- `19_MODULE_COMMIT_MATRIX.md`
  - 把关键模块与关键提交直接绑定起来。
- `20_EXPERIMENT_CURVE_ARCHETYPES.md`
  - 把实验曲线整理成中期甜点型、后期漂移型、平台型等类别。

## 这套草稿的使用方式

- 如果后面要写正式历史报告，这套目录可以直接变成“资料库”。
- 如果要做论文 related work / method / ablation 章节，这里的“模型阶段”和“实验组映射”可以直接抽取。
- 如果要恢复老版本实现，可以先看每个阶段文档里的“代表提交”和“核心模块”。
