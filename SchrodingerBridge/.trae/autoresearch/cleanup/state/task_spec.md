# Task: 远程+本地实验数据整理清理

## Goal
整理清理远程I盘和本地的实验目录、数据集、文档，去重、归并、重排布。所有数据与真实实验对齐。

## Milestones
1. [远程] 删除可清理项（40个smoke + 28个失败probe + 14个invalid aaai2027）
2. [远程] 创建 exp_archive/ 按 baseline/samam/ours_stage1/ours_stage2/historical 分组
3. [远程] 写 docs/exp/remote_experiments.md 实验清单（含时间/模型/数据集/训练时长/推理时长）
4. [本地] 探查并去重实验目录
5. [本地] 数据集去重检查
6. [本地] 写 docs/exp/local_experiments.md
7. [本地] 对齐所有文档与真实实验数据
8. 写 docs/exp/README.md 总入口

## Success Criteria
- 远程I盘 /mnt/i/Github/Latent_Style/exp_archive/ 下按 baseline/samam/ours 分目录
- 释放 ≥10G 磁盘空间（删除smoke/probe/invalid）
- docs/exp/ 包含完整实验清单（每实验含 mtime/模型/数据集/训练时长/推理时长）
- 本地所有 *.md 文档中的实验数据与真实文件对齐
- 无重复目录、无废弃probe

## Constraints
- 零交互：不询问用户，自行决策并写日志
- baseline 与 ours 必须分目录存放
- SaMam 关键训练（44G的 samam_distinct5_512_scratch_7k）必须保留
- final_works/ 全部保留
- 删除前先记录到 findings.jsonl，可追溯

## State Files
- progress.json: 迭代计数与状态
- findings.jsonl: 删除/移动记录
- iteration_log.jsonl: 每轮总结
