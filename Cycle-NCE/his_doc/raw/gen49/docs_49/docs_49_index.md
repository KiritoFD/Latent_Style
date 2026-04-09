# docs_49_index.md

> 更新时间: 2026-04-09  
> 用途: 给 `docs_49/` 当前文档做分组导航，防止材料越堆越散

---

## 1. 从哪里开始读

如果第一次进入这批文档，建议按下面顺序：

1. `plan_49.md`
2. `exp_schedule.md`
3. `model_transition_43_to_49.md`
4. `exp_series_scoreboard.md`
5. 再根据需要进入具体支线或模块文档

---

## 2. 总导航类

- `plan_49.md`
- `docs_49_index.md`
- `module_index_49.md`
- `exp_schedule.md`
- `model_transition_43_to_49.md`
- `exp_series_scoreboard.md`

这些文档负责回答：

- 这批材料在做什么
- 现在做到哪了
- 时间线怎么走
- 支线之间怎么互相对照

---

## 3. 实验支线台账

- `exp_46.md`
- `exp_Aline120.md`
- `exp_chess.md`
- `exp_freq.md`
- `exp_repulse.md`
- `exp_49.md`

这些文档负责回答：

- 某一条实验线研究什么
- 配置是什么
- 结果是什么
- 在大历史里属于什么类型

---

## 4. 架构概览与过渡

- `model_43.md`
- `model_49.md`
- `model_transition_43_to_49.md`

这些文档负责回答：

- 某个时点模型长什么样
- 43 到 49 这段为什么会变成现在这样

---

## 5. 模型模块文档

- `mode_AdaGN.md`
- `mode__BaseStyleModulator.md`
- `mode_TextureDictAdaGN.md`
- `mode_GlobalDemodulatedAdaMixGN.md`
- `mode_CrossAttnAdaGN.md`
- `mode_ResBlock.md`
- `mode_SpatialSelfAttention.md`
- `mode_AttentionBlock.md`
- `mode_SemanticCrossAttn.md`
- `mode_NormFreeModulation.md`
- `mode_DecoderTextureBlock.md`
- `mode_StyleAdaptiveSkip.md`
- `mode_StyleRoutingSkip.md`
- `mode_StyleMaps.md`
- `mode_LatentAdaCUT.md`
- `mode_in_idt_loss.md`
- `mode_structure_tuning.md`

这些文档负责回答：

- 模块在干什么
- 为什么这样设计
- 解决了什么问题
- 和实验结果怎么对应

---

## 6. 损失函数文档

- `loss_soft_repulsive_loss.md`
- `loss__masked_l1_mean.md`
- `loss__masked_mse_mean.md`
- `loss_calc_spatial_agnostic_color_loss.md`
- `loss_calc_swd_loss.md`
- `loss__db_tsw.md`
- `loss__swd_distance_from_projected.md`
- `loss_AdaCUTObjective.md`

---

## 7. 历史记录类

- `commits_0403_0409_detail.md`

这类文档负责保留提交记录和阶段性第一手线索。

---

## 8. 当前阅读建议

### 如果想看主线结构史

读：

- `exp_schedule.md`
- `model_transition_43_to_49.md`
- `exp_46.md`

### 如果想看极端高 style 支线

读：

- `exp_Aline120.md`
- `exp_series_scoreboard.md`

### 如果想看病理排查

读：

- `exp_chess.md`

### 如果想看频谱/identity 关系

读：

- `exp_freq.md`

### 如果想看单个模块

先看：

- `module_index_49.md`

再跳去具体 `mode_*.md`

---

## 9. 2026-04-09 新增入口补记

本轮新补的高价值入口：

- `exp_45.md`
  - 专门处理 `45` 的老格式 `summary.json`，把它从“看起来像没结果”纠正成“已有一手结果”
- `exp_Gate.md`
  - 当前 `Gate` 还是计划矩阵文档为主，但配置链已经完整，值得独立阅读
- `trainer_49.md`
  - 用来解释这些实验目录背后的训练平台是怎么工作的，适合在看大量实验前先补背景
- `mode_attn_gate_mode.md`
  - 用来解释 `Gate` 系列最核心的结构旋钮到底在控制什么
- `loss_aux_delta_variance.md`
  - 用来解释 `07/08_aux_loss_*` 这些配置名背后的真实损失含义
