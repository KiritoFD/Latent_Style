# 620 整体动力学白化问题决策树 Spec

## Why
620 的白化/雾化不是单个模块的局部故障，而是整个模型动力学已经落入平凡解盆地后的系统性失效。只要这一整体问题没有被数学上刻画清楚、实验上被探针确认、方案上被闭环验证，模型就会持续产出看似稳定但本质失效的结果，后续 text、cross-attn、DINO 等优化也都会建立在错误基础上。

## What Changes
- 将现有 `620-whitening-fix` 从“单一路径修复草案”升级为“诊断优先、证据闭环、分阶段放行”的总计划。
- 将问题定义提升为“整体模型动力学导致的平凡解坍缩”，而不是局部白化补丁问题。
- 明确第一阶段目标是解决白化/雾化，而不是并行推进所有 620 结构改动。
- 强制采用“数学思考 → 提出方案 → 实验验证 → 修正理论”的循环，作为整个变更的执行框架。
- 将整个执行计划改写为决策树，所有推进都由诊断结论和放行门触发，而不是线性扫参。
- 建立白化专用指标体系，覆盖图像空间、潜空间、层内统计和后处理补偿需求。
- 建立模型内部探针链路，定位白化究竟发生在 style token、cross-attn、norm、endpoint head 还是 loss/target projection。
- 统一远程 3060 WSL 实验流程与产物回收规则，要求所有关键结论都写入 `docs/620/fog/`。
- 增加严格放行门：白化未达标前，不启动 text、cross-attn 改造、DINO 扩展和 Phase 4 广谱扫描。
- 为 DINO 增加“高成本先验必须显著收益否则砍掉”的强约束。

### 决策树主线
```text
Root: 620 输出出现白化/雾化，怀疑整体动力学落入平凡解
|
+-- Node A: 现有证据是否足以统一问题定义？
|   |
|   +-- 否 -> 审计仓库、远程实验、checkpoint、日志、文档 -> 形成基线
|   |
|   +-- 是 -> 进入 Node B
|
+-- Node B: 数学上能否把问题表述为整体动力学平凡解？
|   |
|   +-- 否 -> 继续建立整体状态变量、训练/推理路径、可观测量、候选机制
|   |
|   +-- 是 -> 进入 Node C
|
+-- Node C: 当前证据支持哪类主导机理？
|   |
|   +-- C1 loss / target projection 主导
|   +-- C2 norm / 统计塌缩主导
|   +-- C3 cross-attn / style 注入失效主导
|   +-- C4 endpoint / solver mismatch 主导
|   +-- C5 证据不足 -> 先补 probe，不得进入修复实验
|
+-- Node D: 对主导机理是否已给出可证伪预测？
|   |
|   +-- 否 -> 继续数学推导，补可观测量与最小实验
|   |
|   +-- 是 -> 进入 Node E
|
+-- Node E: 最小必要实验是否支持该理论？
|   |
|   +-- 否 -> 修正理论，回到 Node B/C/D
|   |
|   +-- 是 -> 进入 Node F
|
+-- Node F: 白化指标是否已压到接近 Seedream IDT 水平？
|   |
|   +-- 否 -> 保持只做白化主线，继续 Node C/D/E 循环
|   |
|   +-- 是 -> 进入 Node G
|
+-- Node G: 是否恢复 620 后续优化？
    |
    +-- Text / cross-attn 改动：仅在不反弹白化时放行
    +-- DINO：收益不显著立即砍掉
    +-- 其余 620 计划：按白化门后顺序恢复
```

## Impact
- Affected specs: `620_spatial_bridge`, `620/fog` 诊断与实验流程
- Affected code: `src/model620.py`, `src/blocks620.py`, `src/style_encoder620.py`, `src/losses620.py`, `src/utils/run_evaluation.py`, `src/utils/training.py`, `tools/experiments/*`, `configs/620*.json`

## ADDED Requirements
### Requirement: 白化问题必须先完成仓库与实验审计
系统 SHALL 在任何新改动前，先整理当前仓库、最近实验、现有文档和远程运行链路，形成一份统一的基线判断。

#### Scenario: 统一基线建立
- **WHEN** 开始处理 620 白化问题
- **THEN** 必须先审计 `docs/620/`、`docs/620/fog/`、`configs/620*.json`、`src/*620*`、远程 launcher 与最近 checkpoint/日志痕迹
- **THEN** 必须形成一份“当前最可信基线、已失败分支、待验证假设、下一步唯一优先级”的书面结论并归档到 `docs/620/fog/`

### Requirement: 白化问题必须补齐可检验的数学推导文档
系统 SHALL 为白化/雾化诊断建立与实验互相校验的数学推导任务，并将理论结论写入 `docs/620/fog/theory/`。

#### Scenario: 理论文档落地
- **WHEN** 需要解释白化/雾化的成因、阶段性现象或修复方向
- **THEN** 必须输出对应理论文档，覆盖 loss/目标投影/归一化/attention/endpoint path 等至少一个可检验机制
- **THEN** 理论文档必须明确假设、推导过程、可观测量、如何被探针或实验验证、哪些结论仍待证伪
- **THEN** 理论工作不得替代实验验收，而应作为实验设计和结果解释的证据补充

### Requirement: 必须建立整体模型动力学表述
系统 SHALL 将 620 模型视为包含 style encoder、cross-attn bridge、norm、endpoint head、training target projection、solver 与 evaluation 的整体动力学系统进行分析，而不是只对单一 loss 或单一模块下结论。

#### Scenario: 动力学问题表述完成
- **WHEN** 开始构建白化/雾化的理论解释
- **THEN** 必须定义状态变量、关键映射、损失项、训练时与推理时的演化路径及其耦合关系
- **THEN** 必须明确“平凡解”在这一系统中的定义、可观测表现、进入条件与退出条件
- **THEN** 必须把局部模块假设放回整体动力学里判断，而不能孤立地解释 norm、attention 或 endpoint

### Requirement: 必须执行理论—方案—实验—修正理论循环
系统 SHALL 以迭代研究循环而不是单次修复方案推进白化问题。

#### Scenario: 单轮闭环成立
- **WHEN** 进入任意一轮白化问题推进
- **THEN** 必须先给出当前理论解释与可证伪预测
- **THEN** 必须据此提出最小必要方案与实验设计
- **THEN** 必须在远程 3060 上执行实验并记录结果
- **THEN** 必须根据结果修正理论、更新下一轮假设与决策文档

### Requirement: Spec 必须以决策树组织
系统 SHALL 使用决策树而非单一路线计划来组织 620 白化问题的研究、修复与放行。

#### Scenario: 决策树驱动执行
- **WHEN** 需要决定下一步做数学分析、probe、修复实验或后续优化
- **THEN** 必须先回答当前位于哪一个决策节点
- **THEN** 必须写清该节点的输入证据、判断条件、允许动作、禁止动作和回退节点
- **THEN** 若节点条件不足，优先补证据与数学分析，不得跳步进入更大范围实验

### Requirement: 白化问题必须有专门的定量指标体系
系统 SHALL 为白化/雾化建立独立于 `clip_style` 与 `content_lpips` 的量化指标，并将其纳入训练与评测产物。

#### Scenario: 指标定义完成
- **WHEN** 为 620 模型做白化诊断或修复评估
- **THEN** 必须同时输出图像空间指标，例如亮度/对比度/动态范围/饱和度/颜色方差塌缩及综合白化分数
- **THEN** 必须同时输出潜空间与模型内部指标，例如 endpoint alpha、high-frequency alpha、effective rank、channel std ratio、covariance trace、后处理补偿量
- **THEN** 必须给出与 Seedream IDT 的对照口径，作为是否“白化压到同一水平”的判断标准

### Requirement: 白化机理必须通过模型内部探针确认
系统 SHALL 提供可复用的探针，逐层确认白化/雾化发生的位置、时间和机制。

#### Scenario: 探针闭环成立
- **WHEN** 对当前最优 checkpoint 或新修复分支做诊断
- **THEN** 必须覆盖 style encoder 输出、cross-attn 输入输出、各层 norm 前后统计、endpoint head 分支、loss 中的 `x_t/projected_target/z_hat1`
- **THEN** 必须能回答“白化首先出现在哪里、哪一类统计量塌缩、哪一步把风格信号洗掉了”
- **THEN** 所有探针结论都必须文档化到 `docs/620/fog/` 对应子目录

### Requirement: 远程 3060 必须是主执行环境并可复现
系统 SHALL 以用户指定的远程 3060 WSL 环境作为白化问题主实验场，并将运行方式、命令、路径、日志、checkpoint 和回收产物标准化。

#### Scenario: 远程复现实验
- **WHEN** 需要验证最近实验、跑探针、做 smoke/formal 训练或 full eval
- **THEN** 必须优先使用 `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62` 对应环境
- **THEN** 必须明确使用的仓库路径、config、checkpoint、日志路径和产物回收位置
- **THEN** 必须把远程实验摘要与关键产物同步回 `docs/620/fog/`

### Requirement: 白化修复必须通过阶段放行门
系统 SHALL 在白化未通过验收前，禁止恢复更广范围的 620 优化矩阵。

#### Scenario: 阶段放行
- **WHEN** 白化问题尚未被压制到接近 Seedream IDT 的水平
- **THEN** 只允许进行与白化直接相关的指标建设、探针、机理确认、endpoint path 修复和必要对照实验
- **THEN** 不得启动 text 条件、cross-attn 结构大改、DINO 扩展和 `docs/620/phase4_plan.md` 的全量扫描

### Requirement: 数学分析优先级高于结构试错
系统 SHALL 在每个关键决策节点优先推进数学建模、可证伪预测和理论修正，而不是直接扩大实验矩阵。

#### Scenario: 进入复杂分支前
- **WHEN** 某个候选方向涉及高开销训练、外来先验或大范围结构修改
- **THEN** 必须先产出对应数学解释、失败模式与最小验证实验
- **THEN** 若没有足够理论支撑，只允许进行补 probe 与补推导，不允许直接进入正式训练

### Requirement: 白化通过后再恢复 620 后续优化
系统 SHALL 在白化验收通过后，按优先级恢复 `620` 既定优化计划。

#### Scenario: 恢复后续计划
- **WHEN** 白化指标达到与 Seedream IDT 可接受对齐的水平，且核心副作用可控
- **THEN** 才能继续 text 引入、cross-attn 方式调整、skip 比例、Per-Region SWD 等后续实验
- **THEN** 后续实验也必须沿用白化指标，确保不会以提升 `clip_style` 为代价重新引入雾化

### Requirement: DINO 必须执行收益门控
系统 SHALL 将 DINO 视为高开销外来先验，只有在带来明确收益时才保留。

#### Scenario: DINO 去留判断
- **WHEN** 白化问题已基本解决并进入后续优化阶段
- **THEN** 必须先对比“无 DINO”与“有 DINO”的同成本实验
- **THEN** 若 DINO 未带来显著且稳定的白化改善或整体指标提升，则必须从主线计划中移除
- **THEN** 只有在收益显著超过成本时，才允许继续多尺度 DINO 或更重先验实验

## MODIFIED Requirements
### Requirement: 620 白化修复主线
620 白化修复主线不再默认等同于某一个预设理论或单一修复方案。理论文档、NSWD 假设、endpoint 结构修复、norm/cross-attn 假设都属于可检验候选路径，只有经过最近实验复盘、定量指标和内部探针共同支持后，才可升级为主线方案。

### Requirement: 620 文档落点
所有与白化问题有关的分析、远程实验、探针结果、验收对比和阶段决策，都必须优先写入 `docs/620/fog/` 下的结构化文档，而不能只停留在分散日志、README 注释或远程机器目录中。

## REMOVED Requirements
### Requirement: 白化修复默认等同于 NSWD
**Reason**: 现有仓库已经出现与早期“SWD 梯度平坦”叙事不一致的新证据，不能在未完成最新审计与探针闭环前，把 NSWD 视为唯一主线。
**Migration**: 将 NSWD 视为候选修复路径之一，与 endpoint path 修复、norm 归因、target projection 归因、late-training mismatch 归因并列纳入统一评审与验收框架。
