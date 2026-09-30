# 训练与基础设施说明（简化版）

## 目标
文件：`src/run.py`、`src/trainer.py`、`src/losses.py`

本轮重构的目标是把训练基础设施恢复成“默认、可预期”的形态：
1. 损失每步正常计算，不做稀疏化/轮转魔改。
2. compile 走代码内置默认策略，配置只保留开关。
3. 配置项精简，减少历史策略残留带来的误判。

## 损失计算策略（关键变更）
文件：`src/losses.py`

当前策略：
- `cycle / struct / edge / stroke / nce / semigroup` 全部按权重直接计算。
- 不再使用 interval、round-robin、warmup/ramp、semigroup 子批次采样。

这意味着：
- 每一步的计算路径固定，训练信号一致。
- 日志中各 loss 的“active/eff”字段现在和权重直接对应，不再受调度器影响。

## compile 策略
文件：`src/trainer.py`

训练配置只保留：
- `training.use_compile`（开/关）

当开启时，代码固定使用：
- `torch.compile(..., backend="inductor", mode="default", fullgraph=False)`
- 自动关闭 cudagraph：`TORCHINDUCTOR_CUDAGRAPHS=0`
- 自动开启 `capture_scalar_outputs` 与 `suppress_errors`
- 自动启用 compile cache，目录固定为：`<checkpoint_dir>/torch_compile_cache/{inductor,triton}`

## 运行建议
1. 先用当前精简配置训练 20~50 epoch，看风格指标是否单调改善。
2. 如果风格仍偏弱，优先调模型风格增益和 `w_stroke_gram / w_color_moment`，不要先加复杂调度。
3. 只有在稳定后再考虑引入额外损失（如 NCE、semigroup），并逐项开启做 A/B。

## 配置约定
文件：`src/config.json`

当前配置只保留三类字段：
1. 模型结构与风格注入参数。
2. 直接生效的 loss 权重与训练步进参数。
3. 训练基础设施参数（AMP、compile 开关、DataLoader）。

原则：
- 每个配置项都必须有实际代码路径读取。
- 删除“仅历史兼容但当前不生效”的配置项。
