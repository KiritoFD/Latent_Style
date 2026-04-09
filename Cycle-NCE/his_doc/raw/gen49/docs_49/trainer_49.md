# trainer_49.md

> 日期: 2026-04-09
> 主题: `src/trainer.py` 在当前 49 时点到底负责什么
> 直接证据:
> - `src/trainer.py`
> - 当前 `src/model.py`
> - 当前 `src/losses.py`

---

## 1. 先把角色说清楚

`trainer.py` 不是一个“把 loss.backward() 套起来”的薄壳。

在现在这个仓库阶段，它承担的是一整层实验基础设施职责:

1. 把 config 里的训练策略真正落到 PyTorch
2. 负责模型构建、compile、AMP、channels_last、warmup、scheduler
3. 管理 checkpoint 恢复和源码快照
4. 记录实验过程中的主损失、子损失、attention 指标和吞吐指标

如果说 `model.py` 负责回答“这个网络怎么生成”，那 `trainer.py` 负责回答“这套想法如何被稳定地训练、观察、复现”。

---

## 2. 顶层职责分层

当前 `AdaCUTTrainer` 大致可以拆成六个职责块:

### 2.1 初始化训练环境

在 `__init__()` 里，它会处理:

- `torch.set_float32_matmul_precision("high")`
- TF32 开关
- cudnn benchmark 策略
- `channels_last`
- gradient checkpointing
- `torch.compile`
- AMP dtype

### 2.2 构建模型与损失

它会:

- 用 `build_model_from_config(model_cfg, use_checkpointing=...)` 构建模型
- 用 `AdaCUTObjective(config)` 构建损失对象

### 2.3 构建优化器与学习率调度

当前支持:

- `AdamW`
- `CosineAnnealingLR`
- `MultiStepLR`
- `OneCycleLR`
- 额外的 warmup 逻辑

### 2.4 跑 epoch / step

`train_epoch()` 负责:

- dataloader 取数
- batch 上设备
- autocast
- 调 `loss_fn.compute(...)`
- backward
- accumulation
- clip grad
- optimizer step
- scheduler step
- 指标累计与 tqdm 展示

### 2.5 写日志与留痕

初始化时它就会:

- 在 checkpoint 目录下保存 `config.json`
- 拷贝 `trainer.py / losses.py / model.py / dataset.py / run.py`
- 建立训练 CSV

### 2.6 checkpoint 恢复

它支持:

- 显式指定 `resume_checkpoint`
- 自动寻找最新 `epoch_*.pt`
- 恢复 model / optimizer / scheduler / global_step / start_epoch

---

## 3. 初始化阶段到底做了什么

### 3.1 性能相关开关

`__init__()` 一上来会把许多执行层设置归到 config:

- `allow_tf32`
- `cudnn_benchmark`
- `channels_last`
- `use_gradient_checkpointing`
- `use_compile`
- `compile_backend`
- `compile_mode`
- `compile_fullgraph`
- `use_amp`
- `amp_dtype`
- `fused_adamw`

这背后的设计思路很明显:

- 训练器不想把“是否能跑快、是否省显存”留给命令行临时处理
- 而是要把这些也变成实验配置的一部分

### 3.2 模型构建与 unwrap

训练器里有一个小但重要的工具:

- `_unwrap_model(module)` 会优先取 `_orig_mod`

这是为了兼容 `torch.compile` 后的包装模型。

### 3.3 config 与源码快照

训练器在 checkpoint 目录里会保存:

- `config.json`
- `trainer.py`
- `losses.py`
- `model.py`
- `dataset.py`
- `run.py`

这一步特别关键，因为 04 月这段 `model.py` / `losses.py` 变化很快，只存权重远远不够。

---

## 4. 优化器、scheduler、warmup 的设计意图

### 4.1 优化器

当前优化器是 `AdamW`，优先尝试 `fused=True`，不支持时退回普通版本。

### 4.2 scheduler 支持多种模式

当前逻辑支持:

- `cosine`
- `multistep`
- `onecycle`

而且 `onecycle` 是按 batch 级 step，其他默认按 epoch step。

### 4.3 warmup 不是附属件，而是正式机制

训练器支持两种 warmup 配置方式:

- 直接给 `warmup_steps`
- 给 `warmup_ratio`，再根据总 step 数估算

然后在真正 optimizer step 前调用 `_apply_warmup_lr()`。

---

## 5. `train_epoch()` 的核心流程

### 5.1 它先做的不是训练，而是建观测框架

每个 epoch 开始时，它先准备大量时间统计量:

- `data_time_total`
- `transfer_time_total`
- `fwd_loss_time_total`
- `backward_time_total`
- `optimizer_time_total`
- `step_overhead_time_total`
- `compute_time_total`

### 5.2 loss 计算是通过目标对象统一完成的

在 autocast 环境里，训练器调用:

```python
loss_dict = self.loss_fn.compute(
    self.model,
    content=content,
    target_style=target_style,
    target_style_id=target_style_id,
    source_style_id=source_style_id,
    epoch=epoch,
    num_epochs=self.num_epochs,
)
```

这说明 loss 在当前系统里不是单个标量函数，而是一整层训练编排逻辑。

### 5.3 accumulation 是正式一等公民

训练器显式围绕 accumulation 写了完整逻辑:

- `loss / self.accumulation_steps`
- 每 `accumulation_steps` 才 step 一次
- epoch 结尾处理“不整除”的尾批

### 5.4 grad clip 在真正 step 前执行

每次真正优化前，它会做:

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
```

这对于 SWD、repulsive、energy idt 这类强推型 loss 非常关键。

---

## 6. attention 指标为什么会被专门记录

训练器里一个特别值得记的点是 `_collect_attention_metrics()`。

它会:

1. unwrap compile 后的模型
2. 找 `body_blocks`
3. 读取各 block 的 `last_attn`
4. 计算:
   - `aent`: 注意力熵
   - `amax`: 注意力最大值

为什么要单独记这两个量:

- 当前阶段 attention 已经是主线结构变量
- 只看最终图像和总 loss 已经不够
- 需要一个中间观测层来帮助理解 `chess`、`Gate` 一类实验

---

## 7. 它记录了哪些训练指标

日志列里除了主损失，还包括:

- `identity_ratio`
- `sched_factor`
- `idt_anchor`
- `topo_align`
- `idt_repel`
- `aent`
- `amax`
- `lr`
- 各类时间统计
- `samples_per_sec`
- `compute_samples_per_sec`

这说明训练日志本身就是实验过程文档，而不是附属文件。

---

## 8. checkpoint 设计为什么值得单写

`save_checkpoint()` 保存的是:

- `epoch`
- `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `config`
- `metrics`

同时 `_maybe_resume()` 还能自动找最新 checkpoint。

这让训练流程天然适合长线迭代和中断续跑。

---

## 9. 它为什么对考古很关键

### 9.1 它把实验复现从口头描述变成了落盘事实

你现在能在很多实验目录里直接看到:

- `config.json`
- 当时的 `model.py`
- 当时的 `losses.py`
- 当时的 `trainer.py`

这不是偶然，是训练器主动做的。

### 9.2 它让不同系列之间可以更公平比较

很多实验共享:

- 同样的日志框架
- 同样的 checkpoint 格式
- 同样的 warmup / scheduler / step 逻辑

### 9.3 它说明项目已经进入平台化实验阶段

当前这个 `trainer.py` 展示的不是一次性脚本心态，而是研究平台心态:

- 每次新想法都能在同一套训练基础设施上快速插入
- 结果可以通过统一日志和 checkpoint 体系回收

---

## 10. 当前可下的结论

### 结论 1

`trainer.py` 在当前版本里已经是实验平台层，而不只是训练循环。

### 结论 2

它最重要的价值之一，是把源码快照、config、metrics、checkpoint 绑在一起，极大增强了历史可追溯性。

### 结论 3

attention 指标 `aent/amax` 被正式纳入日志，说明 attention 行为本身已经成为这一阶段的核心研究对象。

### 结论 4

如果不把 `trainer.py` 单独立档，就会低估很多实验差异背后的“训练基础设施演进”。

