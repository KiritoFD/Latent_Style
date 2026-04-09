# 模块专档：Trainer 与 Infra 演化

## 1. 为什么 trainer 值得单独写

这个项目的历史里，trainer 不是被动执行器，而是直接影响结论可信度的部分。

具体表现：

- mixed precision / BF16
- checkpointing
- micro-batch
- channels-last
- evaluation cache
- classifier / metric 流程

这些都不是纯工程细节，它们会反过来决定“哪个结构看起来更好”。

## 2. 早期 trainer：功能很多，但系统还不稳

早期特点：

- 同时承载大量 loss、诊断、实验功能
- 很多阶段靠 overfit50 做快速回路
- 会反复因为 batch、VRAM、评估口径出问题

## 3. 2026-02-16：BF16 与随机数/流水线优化

`12dfe7c` 表明训练系统已经从“能跑”进入“要跑得更顺”。

这一阶段开始重视：

- 精度与吞吐折中
- 不打断流水线
- 减少显存压力

## 4. 2026-03-05：评估缓存与 trainer 重构

`c619fda` 是 trainer / infra 的一个重锚点：

- evaluate cache added
- trainer 大改
- 配套回填评估脚本出现

这说明项目已经不满足于临时评估，而是开始做：

- 可复用评估资产
- 批量回填
- 系统汇总

## 5. 2026-03-08：checkpointing 变必要项

`dbcf851` 的提交信息非常直白：

- 梯度检查点真的要开，不然显存爆炸

这说明模型复杂度到这里已经明显跨过了显存临界点。

## 6. 2026-03-30 到 2026-04-02：吞吐优化与 micro-batch

两个关键节点：

- `1e25659`：`56s/epoch`
- `4e166f0`：micro batch 效果大好

这两步合在一起说明：

- 后期最有效的改进不只是模块级，而是训练范式级
- trainer 被进一步简化后，反而更稳、更高效

## 7. 当前 trainer 的特征

当前 `AdaCUTTrainer` 已经体现出比较成熟的工程状态：

- accumulation
- attention metric 统计
- onecycle / warmup
- structured epoch metrics
- 明确的 checkpoint 恢复逻辑

## 8. 历史判断

如果要给 trainer 历史下一个结论：

- 这个项目不是“模型结构先赢，然后 trainer 只是托底”，而是结构与 trainer 一起共演化；后期不少关键提升其实来自训练组织方式本身。

