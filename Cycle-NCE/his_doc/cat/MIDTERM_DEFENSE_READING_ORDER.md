# 中期答辩材料阅读顺序

更新时间：2026-04-09

这份顺序不是“全量阅读路径”，而是面向中期答辩的最短讲述路径。目标是让材料从“资料库”变成“能讲清楚的故事”。

## 1. 第一层：先抓主线

先读：
1. [2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)

这一层要回答：
- 研究目标是什么
- 模型为什么不是一次成型，而是多阶段推进
- 每一代设计对应验证了什么
- 到目前为止哪些结论是稳定的

## 2. 第二层：准备“老师追问”

如果老师追问“具体怎么演进的”，按下面顺序补看：
1. [08_COMMIT_TIMELINE_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/08_COMMIT_TIMELINE_DETAILED.md)
2. [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)
3. [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)

如果老师追问“为什么这些实验能支持你的结论”，按下面顺序补看：
1. [06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md)
2. [17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md)
3. [18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md)
4. [20_EXPERIMENT_CURVE_ARCHETYPES.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/20_EXPERIMENT_CURVE_ARCHETYPES.md)

## 3. 第三层：拆成答辩页

建议答辩页可以按下面结构组织：

1. 任务定义
2. 模型演进四阶段
3. 核心模块演进
4. Loss 设计演进
5. 实验系统与分组
6. 代表性结果与失败经验
7. 当前最佳认识与下一步计划

## 4. 建议重点讲的结论

- 这项工作不是简单调参，而是从参考图条件化，逐步演进到风格内嵌、空间先验、结构化调制、attention backbone、micro-batch 训练组织的一条系统路线。
- 实验已经形成主线实验、单因素消融、路径消融、结构搜索、联合优化几层结构，不再是零散试错。
- 目前最重要的认识不是“谁 style 最高”，而是已经明确识别出 style 冲高和内容/分布退化之间的典型张力。
- 当前项目已经具备进入中期答辩的材料成熟度，因为有模型演进线、实验支撑线、失败经验线三条并行证据。
