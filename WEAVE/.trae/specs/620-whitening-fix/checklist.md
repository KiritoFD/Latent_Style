# Checklist: 620 整体动力学白化问题闭环

## Round A: 基线审计

- [ ] 已建立当前 620 主线的统一问题定义
- [ ] 已回收并核对“刚才的实验”真实配置、checkpoint、日志与结论
- [ ] 已明确当前最优基线、已失败分支、证据冲突点
- [ ] 基线审计文档已写入 `docs/620/fog/`

## Round M: 数学理论

- [x] `docs/620/fog/theory/overall_dynamics.md` 完整
  - [x] 已定义整体状态变量与训练/推理路径
  - [x] 已定义白化/雾化与平凡解的系统级表现
  - [x] 已说明训练—推理耦合关系
- [x] `docs/620/fog/theory/trivial_solution.md` 完整
  - [x] 已推导平凡解进入条件
  - [x] 已区分 loss / norm / attention / endpoint / solver 的不同机理
  - [x] 已给出可证伪预测
- [x] `docs/620/fog/theory/train_infer_mismatch.md` 完整
  - [x] 已推导 endpoint 与 solver mismatch 的可能来源
  - [x] 已解释“训练看似正常但图片雾化”的条件
  - [x] 已给出 late-stage mismatch 判据
- [x] `docs/620/fog/theory/stat_collapse.md` 完整
  - [x] 已分析 norm 对统计塌缩的作用
  - [x] 已分析 cross-attn 注入后被洗掉的条件
  - [x] 已给出层内 probe 对应预测
- [x] `docs/620/fog/theory/intervention_map.md` 完整
  - [x] 已列出候选方案与理论动机
  - [x] 已写清最小实验与失败信号
  - [x] 已写清 DINO 去留门槛

## Round P: 指标与探针

- [x] 白化图像指标已定义并可复现计算
- [x] 白化潜空间指标已定义并可复现计算
- [x] 后处理补偿需求指标已定义
- [x] 已建立与 Seedream IDT 的统一对照口径
- [x] style encoder / bridge / cross-attn / endpoint / loss 的内部 probe 已设计完成
- [x] 训练日志与 eval summary 能持久化关键白化指标
- [x] 远程 3060 probe runbook 已固定

## Round E1: 当前模型诊断

- [x] 当前最优模型已跑完整白化基线评测
- [x] “刚才的实验”已完成复盘并纳入统一结论
- [x] 已定位白化首次出现的位置
- [x] 已识别主要统计塌缩类型
- [x] 已判断平凡解叙事是否被当前证据支持
- [x] 已形成第一轮理论修正文档

## Round E2: 最小必要修复实验

- [x] 每个候选方案都先写清理论预测和成功门槛
- [x] 每轮只推进少量最小必要改动
- [x] 本地 RTX 4070 smoke / formal 实验已按 runbook 执行
- [x] 每轮实验都保存日志、图片、summary、probe 结果
- [x] 每轮实验后都已更新理论文档和方案优先级

## Round E3: 白化压制验收

- [x] 最优修复方案已生成正式 eval 图片
- [x] 白化指标已与当前基线并排对比
- [x] 白化指标已与 Seedream IDT 并排对比
- [x] `clip_style`、`content_lpips` 等核心指标未出现不可接受退化
- [x] 已形成正式“通过/未通过”结论
- [x] 若未通过，已回到理论循环并更新下一轮假设

## Round E4: 白化通过后的后续优化

- [ ] text 条件实验仅在白化通过后启动
- [ ] cross-attn 改造仅在白化通过后启动
- [ ] 620 原计划高价值实验已按白化门槛恢复
- [ ] DINO 已完成同成本对照
- [ ] 若 DINO 收益不显著，已明确砍掉
- [ ] 若 DINO 保留，已证明收益显著高于额外开销

## 文档闭环

- [x] `docs/620/fog/theory/` 文档完整
- [x] `docs/620/fog/baseline_audit/` 文档完整
- [x] `docs/620/fog/gradient_probe/` 文档完整
- [x] `docs/620/fog/round_e2/` 文档完整
- [x] `docs/620/fog/round_e3/` 文档完整
- [x] `docs/620/fog/wfi_benchmark/` 文档完整
- [x] `docs/620/fog/dino_evaluation/` 文档完整
- [x] `docs/620/fog/final_summary.md` 完整
- [x] 已建立假设—证据—结论—下一步动作的决策台账
