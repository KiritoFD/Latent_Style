# Tasks: 620 整体动力学白化问题 — 数学思考→方案→实验→修正理论

## Round A: 基线审计与问题重述

- [x] Task A.1: 通读仓库与 620 文档，重建当前问题定义
  - [x] 审计 `docs/620/`、`docs/620/fog/`、`configs/620*.json`、`src/*620*`
  - [x] 梳理最近实验、当前最优 checkpoint、已失败分支、证据冲突点
  - [x] 形成“当前最可信结论与未决问题”文档，写入 `docs/620/fog/`

- [x] Task A.2: 回收远程 3060 最近实验痕迹（当前远程环境不可用，本地审计已覆盖最近实验；远程部分待恢复）
  - [ ] 登录远程 WSL，确认代码路径、运行目录、checkpoint、training csv、eval summary
  - [ ] 对齐“刚才的实验”实际使用的 config、轮次、输出与结论
  - [ ] 建立远程产物索引，写入 `docs/620/fog/`

## Round M: 整体动力学理论工作

- [x] Task M.1: 建立 620 整体动力学问题表述 (`docs/620/fog/theory/overall_dynamics.md`)
  - [x] 定义 style encoder、bridge、norm、endpoint head、target projection、solver、eval 的整体状态变量
  - [x] 定义“白化/雾化”与“平凡解”的系统级可观测表现
  - [x] 明确训练时与推理时的动力学路径，以及二者可能失配的位置

- [x] Task M.2: 推导平凡解形成机理 (`docs/620/fog/theory/trivial_solution.md`)
  - [x] 从整体目标函数与结构参数化出发，推导何时会进入 shrinkage / trivial basin
  - [x] 区分 loss 驱动、norm 驱动、cross-attn 驱动、endpoint 驱动、solver 驱动的不同进入机制
  - [x] 给出每种机理对应的可证伪预测和可观测量

- [x] Task M.3: 推导训练—推理失配机理 (`docs/620/fog/theory/train_infer_mismatch.md`)
  - [x] 分析 endpoint、velocity、target projection 与 solver trace 的关系
  - [x] 推导为什么某些 checkpoint 会出现“训练指标看似正常但图片雾化”
  - [x] 明确 late-stage mismatch 的理论判据

- [x] Task M.4: 推导层内统计塌缩机理 (`docs/620/fog/theory/stat_collapse.md`)
  - [x] 分析 GroupNorm/LayerNorm/AdaLN 对振幅、通道方差、颜色统计的影响
  - [x] 分析 cross-attn 注入后再归一化导致 style 信号被洗掉的条件
  - [x] 形成与层内探针对应的数学预测

- [x] Task M.5: 推导候选修复路径与否证条件 (`docs/620/fog/theory/intervention_map.md`)
  - [x] 针对 NSWD、endpoint path 修复、norm 改造、target projection 改造分别给出理论动机
  - [x] 明确每个方案的最小实验、预期收益、失败信号与淘汰条件
  - [x] 明确 DINO 作为高成本先验的收益门槛与砍掉标准

## Round P: 指标与探针体系

- [x] Task P.1: 建立白化/雾化定量指标
  - [x] 定义图像空间指标：亮度、对比度、动态范围、饱和度、颜色方差塌缩、综合白化分数
  - [x] 定义潜空间指标：endpoint alpha、high-frequency alpha、effective rank、channel std ratio、cov trace
  - [x] 定义“后处理补偿需求”指标，量化模型原生输出离目标统计有多远
  - [x] 与 Seedream IDT 建立统一对照口径

- [x] Task P.2: 建立整体动力学探针
  - [x] 在 style encoder、每层 bridge block、cross-attn、endpoint head、loss debug 中加入可观测量
  - [x] 增加 norm 前后、attention 前后、head 输出前后的统计记录
  - [x] 让训练日志与 eval summary 持久化这些指标

- [x] Task P.3: 建立远程 probe runbook
  - [x] 固定远程 3060 的运行命令、路径、批次、采样点与产物回收方式
  - [x] 固定需要分析的 checkpoint 集合：当前最优、最近失败、候选修复
  - [x] 固定输出目录到 `docs/620/fog/` 对应子目录

## Round E1: 当前模型整体诊断

- [x] Task E1.1: 对“刚才的实验”和当前最优模型做静态诊断
  - [x] 跑白化指标与已有 eval 指标，建立当前基线
  - [x] 对比 source / target / generated / repaired 的统计差异
  - [x] 输出 `docs/620/fog/baseline_audit/`

- [x] Task E1.2: 对当前模型做内部 probe
  - [x] 跑 `t=0 / 0.5 / 0.875` 的 endpoint 与 solver 诊断
  - [x] 逐层定位白化首次出现的位置和主要统计塌缩类型
  - [x] 输出 `docs/620/fog/gradient_probe/` 与相关子目录

- [x] Task E1.3: 用实验结果修正理论
  - [x] 对照 Round M 的预测，标记被支持、被否证和待补证的部分
  - [x] 更新 `overall_dynamics.md`、`trivial_solution.md` 等理论文档
  - [x] 形成“下一轮只允许验证的候选方案列表”

## Round E2: 最小必要修复实验

- [x] Task E2.1: 为第一优先候选方案设计最小实验
  - [x] 每个方案只允许最小改动，不并行堆叠多个新想法
  - [x] 先明确理论预测、成功门槛、失败信号
  - [x] 生成 smoke / formal 的本地实验计划

- [x] Task E2.2: 在本地 RTX 4070 上执行最小实验
  - [x] 优先验证最能解释平凡解的方案
  - [x] 记录训练日志、probe、eval、图片与白化指标
  - [x] 输出 `docs/620/fog/round_e2/`

- [x] Task E2.3: 用实验结果修正理论与主线方案
  - [x] 保留被证据支持的方案
  - [x] 对不支持的理论写清否证原因
  - [x] 更新下一轮实验顺序，避免盲目扫参

## Round E3: 白化压制验收

- [x] Task E3.1: 对最优修复方案跑正式评测
  - [x] 保存 eval 图片
  - [x] 计算白化指标、已有任务指标、内部 probe 指标
  - [x] 与当前最优基线和 Seedream IDT 做并排对比

- [x] Task E3.2: 判断是否达到白化放行门
  - [x] 白化/雾化指标压到接近 Seedream IDT 水平
  - [x] 不以严重损害 `clip_style`、`content_lpips` 为代价
  - [x] 形成“通过/未通过”的正式结论文档

- [x] Task E3.3: 若未通过，返回理论循环
  - [x] 汇总剩余症状
  - [x] 修正整体动力学解释
  - [x] 回到 Round E2 继续下一轮最小方案验证

## Round E4: 白化通过后的 620 后续优化

- [ ] Task E4.1: 恢复 text 与 cross-attn 相关实验
  - [ ] text 条件引入必须复用白化指标
  - [ ] cross-attn 改造必须验证不会重新引入雾化

- [ ] Task E4.2: 执行 620 原计划中剩余高价值实验
  - [ ] skip 比例、Per-Region SWD、注意力稀疏化、OT 配对等按优先级恢复
  - [ ] 所有实验都必须先过“对白化无明显恶化”检查

- [ ] Task E4.3: 处理 DINO 去留
  - [ ] 先做无 DINO 对照
  - [ ] 收益不显著则直接砍掉
  - [ ] 仅在显著收益时再考虑多尺度 DINO

## Round D: 文档与决策闭环

- [x] Task D.1: 为每一轮输出结构化文档
  - [x] theory/：整体动力学、平凡解、失配、统计塌缩、方案地图
  - [x] baseline_audit/、gradient_probe/、round_e2/、wfi_benchmark/、dino_evaluation/
  - [x] final_summary.md：最终结论、放行门、保留方案、淘汰方案

- [x] Task D.2: 建立决策台账
  - [x] 记录每个假设的提出时间、证据、结论、下一步动作
  - [x] 明确哪些方案已否证，避免重复试错

# Task Dependencies

```
Round A (审计) ──→ Round M (理论表述) ──→ Round P (指标/探针) ──→ Round E1 (当前模型诊断)
                                                           │
                                                           └──→ Round E2 (最小修复实验) ──→ Round E3 (白化验收)
                                                                                               │
                                                                                               ├── 未通过：返回 Round M / E2
                                                                                               └── 通过：进入 Round E4 (后续优化)
```

- Round A 是全部工作的入口
- Round M 与 Round P 可以部分并行，但都必须在 Round E1 前形成可执行版本
- Round E1 必须先回答“平凡解是否成立、症状首先出现在哪里、哪些旧理论已失效”
- Round E2 每次只推进少量候选方案，避免无依据的大规模扫参
- Round E3 未通过时必须回写理论文档，再进入下一轮
- Round E4 只有在白化正式过门后才允许启动
- Round D 在每轮结束后即时更新，不能集中拖到最后
