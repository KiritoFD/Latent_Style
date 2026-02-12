# 基础设施说明（当前实现，中文详细版）

## 1. 文档范围

本文覆盖训练与评估基础设施：

- `src/run.py`
- `src/trainer.py`
- `src/losses.py`
- `src/utils/inference.py`
- `src/utils/run_evaluation.py`

目标是明确当前“怎么跑、哪里耗时、哪里占显存、如何稳定”。

## 2. 训练主流程（端到端）

### 2.1 入口：`src/run.py`

主要职责：

1. 读取配置和随机种子
2. 设置 CPU 线程与可选 affinity
3. 构建 `Dataset` + `DataLoader`
4. 构建 `AdaCUTTrainer`
5. 逐 epoch 执行：
  - `train_epoch`
  - `step_scheduler`
  - `log_epoch`
  - `save_checkpoint`
  - 可选 `run_full_evaluation`

### 2.2 训练器：`src/trainer.py`

初始化阶段：

- 根据配置构建模型（统一走 `build_model_from_config`）
- 设置 AMP、TF32、channels-last、优化器、调度器
- 初始化日志 CSV 字段
- 可选断点恢复

当前默认行为要点：

- `snapshot_source` 默认 `false`
- `use_gradient_checkpointing` 默认在 config 已打开
- `log_vram_interval` 默认已设置为 `50`

### 2.3 每步训练执行

`train_epoch` 中单 step 大致流程：

1. 统计 data wait 时间
2. `_move_batch` 把数据迁移到设备
3. autocast 下调用 `loss_fn.compute(...)`
4. backward + grad clip + optimizer step
5. 汇总指标并按 `log_interval` 打印
6. 可选 VRAM 统计打印

异常处理：

- CUDA OOM 且 `skip_oom_batches=true` 时，当前 batch 跳过并清缓存

## 3. 可观测性（Round 2 已落地）

### 3.1 显存观测

每个 epoch 开头：

- `torch.cuda.reset_peak_memory_stats()`

周期打印：

- `alloc`
- `reserved`
- `peak`

### 3.2 时间观测

当前已拆分并记录：

- `data_time_sec`
- `compute_time_sec`
- `epoch_time_sec`

可用于快速判断瓶颈在数据侧还是计算侧。

### 3.3 新增训练控制指标

日志与 CSV 已记录：

- `train_num_steps`
- `train_step_size`
- `train_style_strength`
- `heavy_loss_slot_id`
- `semigroup_samples`

这些指标可直接验证分时策略和采样策略是否按预期生效。

## 4. 损失基础设施（负载均衡机制）

`src/losses.py` 已实现三层负载控制。

### 4.1 间隔调度（interval）

支持分别配置：

- `stroke_interval_steps / offset`
- `nce_interval_steps / offset`
- `cycle_interval_steps / offset`
- `semigroup_interval_steps / offset`

作用：

- 把重分支错峰执行，减少“同一步堆叠峰值”。

### 4.2 重损失轮换（round-robin）

支持：

- `heavy_loss_rotation="round_robin"`
- `heavy_loss_rotation_sequence=[...]`

当前默认序列：

- `["stroke", "nce", "semigroup"]`

语义：

- 当前 slot 对应的重损失激活，其他重损失在该步关闭或弱化

### 4.3 Semigroup 子批次与频率控制

Semigroup 当前支持：

- `semigroup_batch_fraction`
- `semigroup_max_samples`
- `semigroup_prefer_transfer_samples`
- `semigroup_detach_midpoint`
- `semigroup_skip_on_cycle_steps`

这是 8GB 下非常关键的“收益/成本平衡器”。

## 5. 训练-推理一致性基础设施

### 5.1 统一模型构建

训练与推理都走：

- `build_model_from_config(...)`

减少 train/infer 参数漂移。

### 5.2 统一控制面

推理已支持并贯通：

- `num_steps`
- `step_size`
- `style_strength`
- `step_schedule`（名称或显式权重）

`run_full_evaluation` 会把上述参数透传给 `run_evaluation.py`。

## 6. 评估链路

`src/trainer.py::run_full_evaluation` 做：

1. 组装评估命令参数
2. 调用 `src/utils/run_evaluation.py`
3. 保存评估日志
4. 聚合 `summary_history.json`

当前支持：

- `--style_strength`
- `--step_schedule`（可逗号权重）

## 7. 8GB 显存推荐执行顺序

建议按下面顺序开功能：

1. 先固定基线：
  - checkpoint on
  - stroke patch 固定小尺寸
  - semigroup 低频 + 小子批次
2. 开启并观察日志：
  - VRAM peak
  - data/compute 时间
  - `train_num_steps`、`semigroup_samples`
3. 再逐步放宽：
  - 提高 semigroup 频率或子批次比例
  - 增加多步训练采样范围
  - 调整重损失轮换序列

## 8. 常见问题排查清单

问题：`it/s` 下降但显存不高

- 看 `data_time_sec` 是否异常高
- 检查 DataLoader worker 配置与存储吞吐

问题：显存峰值偶发尖刺

- 检查该步 `heavy_loss_slot_id` 与 `semigroup_samples`
- 增大 `semigroup_interval_steps`
- 降低 `semigroup_batch_fraction`
- 增大 `stroke_interval_steps`

问题：从 5 步改 3 步推理效果漂移

- 检查 `train_num_steps_*` 是否覆盖了多步场景
- 打开或加强 semigroup（在可承受范围内）
- 统一 `style_strength` 与 `step_schedule` 推理参数
