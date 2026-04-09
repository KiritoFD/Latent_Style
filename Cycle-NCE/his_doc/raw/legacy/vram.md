# 显存分析与调优（当前实现，中文详细版）

## 1. 当前模型量级（实测基线）

基于当前 `src/config.json` 和当前 `src/model.py` 实测：

- 可训练参数：`8,856,333`
- 单样本单步（B=1）Conv/Linear MAC：约 `2.41G`

最重算子（B=1）：

- `dec_conv`: `301,989,888`
- `hires_body.*.conv*`: 每层约 `150,994,944`
- `body.*.conv*`: 每层约 `150,994,944`
- `style_head_trunk.0`: `150,994,944`
- `down`: `134,217,728`

说明：

- 共享头开启后，热点从双分支 3x3 收敛到一个 `style_head_trunk.0`。

## 2. 显存占用构成

训练时显存主要分为两类：

1. 静态占用
  - 参数
  - 梯度
  - 优化器状态（AdamW 一二阶）
2. 动态占用
  - 前向中间激活
  - 反向保存的计算图
  - 重损失分支的临时张量（stroke/nce/semigroup/cycle）

实战中，峰值波动主要来自动态占用，而不是静态占用。

## 3. 每步前向数量与峰值关系

单步训练前向数量近似：

- student：`+1`
- teacher（若启用）：`+1`
- cycle（当步激活）：`+1`
- semigroup（当步激活）：`+3`

当多个重分支同一步激活时，峰值会明显抬升。

## 4. 为什么会“偶发爆峰”

常见触发模式：

- stroke 分支在高成本 patch 配置下叠加其他重分支
- semigroup 与 cycle 同步激活
- 无 checkpoint 时解码与主干激活堆积

当前版本已通过以下机制压制：

- interval 错峰
- round-robin 重损失轮换
- semigroup 子批次
- decoder + block checkpoint

## 5. 当前已接入的观测指标

训练过程中可直接观察：

- VRAM：
  - `alloc`
  - `reserved`
  - `peak`
- 时间：
  - `data_time_sec`
  - `compute_time_sec`
- 策略状态：
  - `train_num_steps`
  - `train_step_size`
  - `train_style_strength`
  - `heavy_loss_slot_id`
  - `semigroup_samples`

建议把这些指标一起看，不要只看 loss 或 peak 单点。

## 6. 8GB 推荐参数策略

### 6.1 建议起步配置

- `use_gradient_checkpointing=true`
- `log_vram_interval>0`
- `stroke_patch_sizes=[3]`
- semigroup：
  - 低频（较大 interval）
  - 小子批次（fraction + max_samples）
  - `semigroup_detach_midpoint=true`

### 6.2 逐步放宽顺序

1. 提升 semigroup 频率
2. 增加 semigroup 样本比例
3. 增大多步训练采样范围
4. 最后再提升风格强度相关增益

## 7. 排障流程（建议）

步骤 1：看峰值是否持续单调上升

- 若只是在一定区间波动，多数是缓存行为或分时策略正常波动
- 若长期单调抬升，需重点检查分支激活策略和 batch 设置

步骤 2：定位是数据瓶颈还是计算瓶颈

- `data_time_sec` 高：先查 DataLoader/IO
- `compute_time_sec` 高：先查重分支同步激活

步骤 3：按杠杆顺序降峰

1. 提高 `semigroup_interval_steps`
2. 降低 `semigroup_batch_fraction`
3. 提高 `stroke_interval_steps`
4. 关闭 round-robin（用于隔离问题）或调整序列
5. 临时关 semigroup 做对照

## 8. 与多步可控的关系

显存优化不能破坏“步数可控”目标，建议坚持：

- 保留 semigroup（但低频/小子批次）
- 保留训练中的多步采样（`train_num_steps_*`）
- 通过 `style_strength + step_schedule` 控制推理强度，而不是只靠大幅拉增益

这样可以在 8GB 下兼顾稳定性与可控性。
