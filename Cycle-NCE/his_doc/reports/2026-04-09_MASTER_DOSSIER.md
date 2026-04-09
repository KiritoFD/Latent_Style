# Cycle-NCE 总档案

更新时间：2026-04-09

这份文档是当前 `Cycle-NCE/his_doc` 的总入口大档。它的目标不是替代全部子文档，而是把现有资料库组织成一份可以连续阅读、也可以像 linker 一样跳转的总档案。

它服务三类用途：
- 快速理解项目全貌
- 作为中期答辩、组会、论文写作的总底稿
- 作为继续扩写历史和实验结论时的统一入口

## 0. 使用方式

如果你想一口气读完项目主线，按本文顺序读。

如果你想跳着查材料，用下面这组核心入口：
- 目录总入口：[cat/README.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/README.md)
- 材料映射：[cat/MATERIAL_MAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MATERIAL_MAP.md)
- 中期答辩正式稿：[reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)
- 历史重建资料库：[notes/2026-04-09_rebuilt_history/README.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/README.md)

## 1. 资料库结构

当前 `his_doc` 已经按功能拆成五层：

- `cat`
  - 目录、阅读顺序、材料映射
- `raw`
  - 原材料归档，不强行统一口径
- `notes`
  - 工作笔记与历史重建
- `reports`
  - 正式汇报稿
- `appx`
  - 预留附录区

其中最重要的三个路径是：
- [cat](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat)
- [notes/2026-04-09_rebuilt_history](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history)
- [reports](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports)

## 2. 项目一句话总结

`Cycle-NCE` 是一条围绕 VAE 潜空间图像风格迁移展开的长期实验线。它的核心问题不是单纯提高风格分数，而是在风格表达、内容保持、颜色稳定和整体分布质量之间找到可解释、可复现、可扩展的平衡点。

从现有材料看，这个项目已经经历了：
- 参考图条件化原型
- 风格内嵌与空间先验
- 调制器、skip、decoder 系统重构
- cross-attention / CGW backbone / micro-batch 阶段

与此同时，实验体系也从零散试错逐渐长成了：
- 主线实验
- 单因素消融
- 注入路径消融
- decoder / NCE / SWD / color 路线组
- 曲线分析与多指标均衡排序

## 3. 总阅读地图

### 3.1 如果你关心模型演进

按下面顺序看：
1. [01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md)
2. [02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md)
3. [03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md)
4. [04_ERA_D_CROSS_ATTN_TO_CGW.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/04_ERA_D_CROSS_ATTN_TO_CGW.md)
5. [08_COMMIT_TIMELINE_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/08_COMMIT_TIMELINE_DETAILED.md)

### 3.2 如果你关心核心模块

按下面顺序看：
1. [09_MODULE_STYLE_MODULATORS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/09_MODULE_STYLE_MODULATORS.md)
2. [10_MODULE_SKIP_DECODER_AND_BACKBONE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/10_MODULE_SKIP_DECODER_AND_BACKBONE.md)
3. [11_MODULE_TRAINER_AND_INFRA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/11_MODULE_TRAINER_AND_INFRA.md)
4. [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)
5. [raw/gen49/docs_49](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49)

### 3.3 如果你关心 loss

按下面顺序看：
1. [05_LOSS_EVOLUTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/05_LOSS_EVOLUTION.md)
2. [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)
3. [raw/gen49/docs_49](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49)

### 3.4 如果你关心实验体系

按下面顺序看：
1. [06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md)
2. [12_EXPERIMENT_GROUP_EXP_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/12_EXPERIMENT_GROUP_EXP_DETAILED.md)
3. [13_EXPERIMENT_GROUP_ABLATION_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/13_EXPERIMENT_GROUP_ABLATION_DETAILED.md)
4. [14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md)
5. [17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md)
6. [18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md)
7. [20_EXPERIMENT_CURVE_ARCHETYPES.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/20_EXPERIMENT_CURVE_ARCHETYPES.md)

## 4. 模型演进总叙事

### 4.1 阶段 A：先证明任务成立

这一阶段的代表是参考图条件化原型。模型仍依赖 `style_ref`，说明风格信息还没有真正内嵌进模型参数。这个阶段最重要的工作不是把结果做得多漂亮，而是先证明：
- 潜空间风格迁移这条路线是可走通的
- 调制类结构值得作为主线

对应材料：
- [01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/01_ERA_A_REFERENCE_CONDITIONED_BOOTSTRAP.md)

### 4.2 阶段 B：把风格放进模型

阶段 B 是真正的第一轮架构转折。这里出现了 `style_emb`、`style_spatial_id_16/32`、多 patch SWD 等关键设计，项目从“带参考图推理”转向“风格蒸馏进模型”。这一步把风格表示拆成了两层：
- 全局风格身份
- 空间风格先验

对应材料：
- [02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/02_ERA_B_STYLE_ID_AND_SPATIAL_PRIOR.md)
- [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)

### 4.3 阶段 C：问题转向“怎么注入”而不是“有没有风格”

随着风格表示逐渐稳定，问题开始变成：
- 风格通过什么模块作用于特征
- skip 会不会泄漏 identity shortcut
- decoder 是在生成，还是在偷拷 source 高频

因此这一阶段围绕 `TextureDictAdaGN`、`NormFreeModulation`、`StyleRoutingSkip`、no-norm decoder、NCE/SWD/color 的更系统配方展开。项目的大部分结构消融和参数消融，也是在这一阶段真正成型。

对应材料：
- [03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/03_ERA_C_TEXTURE_DICT_SKIP_AND_DECODER.md)
- [09_MODULE_STYLE_MODULATORS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/09_MODULE_STYLE_MODULATORS.md)
- [10_MODULE_SKIP_DECODER_AND_BACKBONE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/10_MODULE_SKIP_DECODER_AND_BACKBONE.md)

### 4.4 阶段 D：attention、backbone 与训练组织联合进入主线

阶段 D 出现了 `CrossAttnAdaGN`、`SpatialSelfAttention`、CGW backbone、shifted window、micro-batch 等关键词。项目的关注点从单个调制器继续扩大到：
- 更强的特征交互
- 更强的 backbone 组织
- 在显存约束下保持训练有效

这说明项目已经进入“系统架构优化”而不只是单点试错。

对应材料：
- [04_ERA_D_CROSS_ATTN_TO_CGW.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/04_ERA_D_CROSS_ATTN_TO_CGW.md)
- [11_MODULE_TRAINER_AND_INFRA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/11_MODULE_TRAINER_AND_INFRA.md)
- [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)

## 5. 核心模块总链接

### 5.1 风格调制器链条

主线认识：
- `AdaGN` 是起点
- `TextureDictAdaGN` 把低秩纹理先验和调制链条绑在一起
- `CrossAttnAdaGN` 把风格交互推进到 attention 风格

深读：
- [09_MODULE_STYLE_MODULATORS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/09_MODULE_STYLE_MODULATORS.md)
- [19_MODULE_COMMIT_MATRIX.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/19_MODULE_COMMIT_MATRIX.md)
- [mode_AdaGN.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_AdaGN.md)
- [mode_TextureDictAdaGN.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_TextureDictAdaGN.md)
- [mode_CrossAttnAdaGN.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_CrossAttnAdaGN.md)

### 5.2 Skip / Decoder / Backbone

主线认识：
- skip 与 decoder 不是边角料，而是决定是否走捷径的核心
- backbone 强化是后期增益放大器，但不是一切的起点

深读：
- [10_MODULE_SKIP_DECODER_AND_BACKBONE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/10_MODULE_SKIP_DECODER_AND_BACKBONE.md)
- [mode_StyleRoutingSkip.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_StyleRoutingSkip.md)
- [mode_StyleAdaptiveSkip.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_StyleAdaptiveSkip.md)
- [mode_DecoderTextureBlock.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_DecoderTextureBlock.md)
- [mode_SpatialSelfAttention.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/mode_SpatialSelfAttention.md)

### 5.3 训练器与基础设施

主线认识：
- trainer 重构不是纯工程清理，而是训练上限的组成部分
- micro-batch 的引入意味着算力与架构已深度耦合

深读：
- [11_MODULE_TRAINER_AND_INFRA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/11_MODULE_TRAINER_AND_INFRA.md)
- [trainer_49.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/trainer_49.md)

## 6. Loss 系统总链接

Loss 系统是项目行为的控制面板。当前可概括为三层：
- 保任务成立
- 保内容结构
- 保风格与分布平衡

总说明：
- [05_LOSS_EVOLUTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/05_LOSS_EVOLUTION.md)
- [16_SOURCE_EVIDENCE_DIFFS.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/16_SOURCE_EVIDENCE_DIFFS.md)

模块级 loss 文档：
- [loss_AdaCUTObjective.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/loss_AdaCUTObjective.md)
- [loss_calc_swd_loss.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/loss_calc_swd_loss.md)
- [loss_calc_spatial_agnostic_color_loss.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/loss_calc_spatial_agnostic_color_loss.md)
- [loss_soft_repulsive_loss.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/loss_soft_repulsive_loss.md)
- [loss__swd_distance_from_projected.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49/loss__swd_distance_from_projected.md)

## 7. 实验体系总链接

### 7.1 实验分层

当前实验可以理解为五层：
- 主线实验与对照
- 单因素参数消融
- 注入路径与结构消融
- decoder / NCE / SWD / color 路线实验
- 曲线分析与联合优化

总映射：
- [06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/06_EXPERIMENT_GROUPS_AND_MODEL_MAPPING.md)

### 7.2 主线实验组

代表：
- `exp_1_control`
- `exp_2_zero_id`
- `exp_3_macro_strokes`
- `exp_4_zero_tv`
- `exp_5_signal_overdrive`
- `exp_S1_zero_id`
- `exp_S2_color_blind`
- `final_demodulation`

深读：
- [12_EXPERIMENT_GROUP_EXP_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/12_EXPERIMENT_GROUP_EXP_DETAILED.md)

### 7.3 单因素消融组

代表：
- `ablate_A0_base_p5_id045_tv005`
- `ablate_A1_p7_id045_tv005`
- `ablate_A2_p11_id045_tv005`
- `ablate_A3_p5_id030_tv005`
- `ablate_A4_p5_id070_tv005`
- `ablate_A5_p5_id045_tv003`

深读：
- [13_EXPERIMENT_GROUP_ABLATION_DETAILED.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/13_EXPERIMENT_GROUP_ABLATION_DETAILED.md)

### 7.4 颜色与联合优化组

代表：
- `style_oa_*`
- `color_*`
- `clocor1_*`

深读：
- [14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md)

### 7.5 曲线与 checkpoint 选择

关键认识：
- 最后一个 checkpoint 往往不是最好那个
- 中期甜点型、后期漂移型、平台型已经可以分型

深读：
- [17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/17_EXPERIMENT_CURVES_AND_EPOCH_SELECTION.md)
- [20_EXPERIMENT_CURVE_ARCHETYPES.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/20_EXPERIMENT_CURVE_ARCHETYPES.md)

### 7.6 榜单与综合平衡

关键认识：
- style 冠军不等于最佳模型
- `exp_G1_edge_rush`、`exp_3_macro_strokes`、`G0-Base-Gain0.5` 这类结果更适合被当作均衡候选

深读：
- [18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/18_EXPERIMENT_LEADERBOARDS_AND_BALANCE.md)

## 8. 当前最重要的结论

### 8.1 模型主线已经清楚

项目不是一堆互不相干的版本，而是一条相对连续的演进链：
- 参考图条件化
- 风格内嵌
- 空间先验
- 调制器和 decoder 重构
- attention 与 backbone 搜索

### 8.2 风格调制是核心矛盾

backbone 很重要，但比它更核心的是：
- 风格如何表示
- 风格如何路由
- 风格如何被损失约束

### 8.3 只追 style 是错误方向

`exp_S1_zero_id` 一类结果已经反复说明，style 上升常常伴随内容和分布退化。当前项目真正有价值的地方，是已经逐渐从“单指标优化”转向“多指标平衡”。

### 8.4 实验体系已经具备研究型结构

当前的实验已经能支撑中期答辩、论文实验章节和后续继续优化，因为它们不再是散点，而是被组织成了：
- 主线对照
- 单因素验证
- 模块与路径验证
- 曲线验证
- 综合排序

## 9. 中期答辩入口

如果你是为了答辩，推荐直接从这里开始：
- [2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)
- [cat/MIDTERM_DEFENSE_READING_ORDER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MIDTERM_DEFENSE_READING_ORDER.md)

## 10. 原材料入口

如果你想回查旧材料、旧自动文档、旧展示资产，走这组入口：
- [raw/legacy](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/legacy)
- [raw/docs_old](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/docs_old)
- [raw/gen49/docs_49](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/gen49/docs_49)
- [raw/assets](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/raw/assets)

## 11. 后续扩写建议

这份总档后面还可以继续长成真正的“总索引母文档”。比较值得继续补的有：
- 每个关键 commit 的源码差异摘录
- 每个时代对应的最佳实验图例
- 面向答辩 PPT 的页级讲稿
- 面向论文写作的 method / experiment 章节母版

## 12. 这一份文档的定位

如果说 `notes/2026-04-09_rebuilt_history` 是证据库，`reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md` 是答辩稿，那这份 `MASTER_DOSSIER` 就是把整个资料库串起来的总 linker。

后面无论是继续考古、写答辩、写论文、回查模块、补实验，只要先打开这份文档，基本都能找到下一跳入口。
