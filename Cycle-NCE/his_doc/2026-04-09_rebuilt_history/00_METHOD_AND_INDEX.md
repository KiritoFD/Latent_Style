# 方法说明与索引

## 1. 本次整理的证据来源

### 1.1 直接证据

- `git --git-dir=C:\Users\xy\repo.git log --all`
- `git show <commit>:Cycle-NCE/src/model.py`
- `git show <commit>:Cycle-NCE/src/losses.py`
- `git show <commit>:Cycle-NCE/src/trainer.py`
- `Y:\experiments\EXPERIMENT_RECORD_FULL_DATA.csv`
- `Y:\experiments\RESULTS_INDEX_20260330.csv`
- `Y:\experiments\DEEP_EXPERIMENT_ANALYSIS_REPORT_20260326.md`
- `Y:\experiments\EXPERIMENT_RECORD_REPORT.md`

### 1.2 推断性结论

下面这些内容我会明确写成“推断”而不是“事实”：

- 某次提交的设计动机，如果提交信息没有写全，只能根据代码改动与实验命名反推。
- 某个实验目录和某个源码快照之间的对应关系，如果目录中只是复制过 `model.py/losses.py/trainer.py`，但没有显式保存 commit id。
- 某个 loss 被保留或淘汰的原因，如果只能从后续提交、实验指标和旧草稿倒推。

## 2. 本次重建出来的主线

我把这段历史先分成四个大的架构时代：

1. Era A：reference-conditioned 原型阶段
2. Era B：style-id 蒸馏与空间先验阶段
3. Era C：TextureDict / skip-routing / decoder 调制阶段
4. Era D：cross-attn -> feature attention -> CGW -> micro-batch 阶段

这四个阶段不是严格互斥，而是主导设计重心不同。

## 3. 关键提交索引

| 提交 | 日期 | 作用 |
| --- | --- | --- |
| `ae596d1` | 2026-02-08 | `Cycle-NCE/src/model.py` 初成型，仍然依赖 `style_ref` |
| `d916277` | 2026-02-08 | “蒸馏把风格放进模型”，开始摆脱推理时参考图 |
| `83ffe10` | 2026-02-12 | map16 / map32 双尺度风格注入思路明确 |
| `1f818cc` | 2026-02-23 | SWD instance vs domain 结论定型，配套脚本扩展 |
| `f7b328c` | 2026-02-26 | AdaGN 修改，笔触变化明显 |
| `c619fda` | 2026-03-05 | no-norm decoder、评估缓存、trainer 重构 |
| `4992e06` | 2026-03-08 | NCE loss 被重新确认有效 |
| `4699637` | 2026-03-10 | tokenizer 单独蒸馏，style embedding 简化 |
| `ed596c0` | 2026-03-22 | 通道映射回 RGB 的缩略图 color loss 获胜 |
| `c8577e0` | 2026-03-26 | 亮度约束 + cross-attn 版本 |
| `426ae0a` | 2026-03-29 | attention 效果明显，开始系统架构搜索 |
| `cfdbaba` | 2026-03-30 | 全部换成 c-g-w backbone |
| `c405b9d` | 2026-03-30 | window attention 增加 shift，修 channel-last 问题 |
| `4e166f0` | 2026-04-02 | micro-batch 大幅改善训练效果 |

## 4. 从实验台账反推出来的分组结构

`EXPERIMENT_RECORD_FULL_DATA.csv` 里已经有相对规范的实验分组：

- 主线实验与对照组（EXP）
- A 系列参数消融组
- 快速扫描与微型回归组
- 注入路径消融组
- decoder 结构与配方组
- NCE / SWD 路线组
- 空间调制与颜色策略组
- Master Sweep

这意味着：到 2026-03 下旬，项目已经不只是“连续试错”，而是形成了比较明确的实验生产线。

## 5. 本轮整理的重点

这次文档不是重复旧草稿，而是优先补齐三类缺口：

1. 每一代模型设计里，核心模块到底是什么。
2. 每一代 loss 是怎么实现的，为什么加，为什么删。
3. `Y:\experiments` 里的实验组，究竟在验证哪一代设计。

## 6. 第二层材料索引

为了避免所有内容都挤在“大报告”里，这次又拆出第二层文档：

- `08_COMMIT_TIMELINE_DETAILED.md`
  - 面向“时间”的索引。
- `09_MODULE_STYLE_MODULATORS.md`
  - 面向 style modulation 模块的索引。
- `10_MODULE_SKIP_DECODER_AND_BACKBONE.md`
  - 面向 backbone / skip / decoder 的索引。
- `11_MODULE_TRAINER_AND_INFRA.md`
  - 面向训练系统和评估系统的索引。
- `12_EXPERIMENT_GROUP_EXP_DETAILED.md`
  - 面向主线实验的索引。
- `13_EXPERIMENT_GROUP_ABLATION_DETAILED.md`
  - 面向结构/参数消融的索引。
- `14_EXPERIMENT_GROUP_COLOR_AND_STYLE_OA.md`
  - 面向 color 与后期联合优化的索引。
- `15_CONFIG_NAMING_AND_SEARCH_SPACE.md`
  - 面向配置命名簇和后期搜索空间的索引。
