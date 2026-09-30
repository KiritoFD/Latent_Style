# Task: 代码/配置简化减法消融 (Deli_AutoResearch Iteration 1)

## 目标
用最少的代码和配置保持 T11 性能 (clip=0.7213, lpips=0.2868)。

## 里程碑
1. 审计 config_schema.py + 源码, 找出可硬编码的战术参数
2. 执行减法: 硬编码已确认最优参数, 从 config 中剔除
3. 删除剩余死分支/无用机制
4. Smoke test 验证 T11 baseline 不破坏
5. 记录 findings + 反向实验提案

## 成功标准
- T11 smoke test 全部通过 (imports/forward/3 solver/4 adain/stochastic DWT)
- config_schema.py 字段数显著减少
- 源码行数减少 (净删除)
- findings.jsonl 记录所有改动 + 反向实验提案

## 约束
- 零交互 (Deli 行为约束 1)
- 状态持久化到文件 (约束 4)
- 不破坏 T11 baseline 性能
- 不删除仍在使用的机制
