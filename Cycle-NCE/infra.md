# 当前训练与评估 Infra 总结（基于 `src/` 实现）

更新时间：2026-02-25  
覆盖范围：`src/run.py`、`src/trainer.py`、`src/dataset.py`、`src/utils/run_evaluation.py`、`src/utils/inference.py`

## 1. 主链路入口

当前主链路：

- 训练入口：`src/run.py`
- 训练执行器：`src/trainer.py`（`AdaCUTTrainer`）
- 评估入口：`src/utils/run_evaluation.py`
- 推理封装：`src/utils/inference.py`（`LGTInference`）

---

## 2. 数据与 DataLoader Infra

主训练数据集类是 `AdaCUTLatentDataset`（`src/dataset.py`）：

- 读取 `*.pt` / `*.npy` latent
- 每个风格域建立独立 tensor 池
- `__getitem__` 返回：
  - `content`
  - `target_style`
  - `target_style_id`
  - `source_style_id`

### 2.1 采样策略

- 内容域：在所有 style 中均匀采样
- 目标域：在所有 style 中均匀采样（含同域）
- 因此 identity 样本概率自然为 `1 / num_styles`

### 2.2 性能策略

- `set_epoch()` 预生成整轮随机索引缓存，降低 `__getitem__` CPU 开销
- 可选 `preload_to_gpu`，会按可用显存与保留比例做准入判断
- 可选 `allow_hflip`

### 2.3 DataLoader 策略（`run.py`）

- `drop_last=True`
- `num_workers=-1` 时自动推断（Windows 下更保守）
- 数据预加载到 GPU 时强制 `num_workers=0`
- `worker_init_fn` 做独立 seed + `torch.set_num_threads(1)` 防线程爆炸

---

## 3. 训练执行 Infra（`AdaCUTTrainer`）

## 3.1 精度与算子策略

- TF32 可控（默认开）
- AMP 可控（fp16/bf16，RTX30 默认偏向 fp16）
- `GradScaler` 仅在 fp16 AMP 下默认启用
- 可选 `channels_last`
- 可选 fused AdamW（不可用时自动回退）

## 3.2 编译与图优化

- 可选 `torch.compile`
- RTX30 默认可自动禁用 compile（稳定性优先）
- compile 缓存目录：`<checkpoint_dir>/torch_compile_cache/{inductor,triton}`
- 默认可关闭 cudagraph，减少长跑不稳定因素

## 3.3 优化与调度

- 优化器：AdamW
- 学习率调度：CosineAnnealingLR（可选）
- 支持梯度累计 `accumulation_steps`
- 支持梯度裁剪 `grad_clip_norm`

## 3.4 健壮性与诊断

- OOM 检测：可选择跳过 OOM batch (`skip_oom_batches`)
- OOM 落盘：`logs/oom_reports/*.json|*.txt`
- VRAM trace / 周期日志 / profiler / NVTX / Nsight 控制点
- 批次合法性检查（可配置频率）

注意：`loss_timing_interval` 与 `cuda_sync_debug` 在训练路径中被强制关闭，以避免同步开销拖慢吞吐。

## 3.5 配置热更新

每个 epoch 会尝试从配置文件热加载可变字段，包括：

- 训练节奏：`num_epochs`、`save_interval`、`full_eval_interval` 等
- 优化参数：`learning_rate`、`weight_decay`、`grad_clip_norm`、`accumulation_steps`
- 部分 loss 权重与采样区间

---

## 4. Checkpoint / Resume Infra

`AdaCUTTrainer.save_checkpoint()` 保存：

- `epoch` / `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `scaler_state_dict`
- `config`
- `metrics`

Resume 策略：

- 若未显式指定 checkpoint，则自动加载 `checkpoint_dir` 下最新 `epoch_*.pt`
- 兼容 compile 前缀 `_orig_mod.*`
- 严格加载失败时回退到 non-strict，并打印缺失/多余键

---

## 5. Full Evaluation Infra

`trainer.run_full_evaluation()` 通过子进程调用 `utils/run_evaluation.py`，并把训练配置映射成评估参数。

## 5.1 评估脚本模式

`run_evaluation.py` 支持：

1. 单 checkpoint 模式（显式 `--checkpoint --output`）  
2. 自动模式（扫描缺失 full_eval 的 checkpoint 并补跑）

## 5.2 Phase 1：生成

- 通过 `LGTInference` 加载模型
- 用 VAE 编码源图 -> latent
- 做 latent scale 对齐（model scale vs vae scale）
- 对每个 target style 生成并 decode
- 支持：
  - `--reuse_generated`（复用历史生成图）
  - `--generation_only`（仅生成不算指标）

## 5.3 Phase 2：指标

可选组件：

- LPIPS（含 CUDA OOM -> CPU fallback 机制）
- CLIP（默认离线优先，可选择联网）
- 图像分类器（优先）
- latent 分类器（回退）

指标输出：

- 样本级：`metrics.csv`
- 汇总：`summary.json`
- 可选 ArtFID/FID 相关统计（Inception 特征）

参考特征缓存：

- 共享缓存目录（`cache_dir`）
- 文件锁避免并发进程重复构建

---

## 6. 训练后历史聚合

`trainer._write_full_eval_history()` 会汇总 `full_eval/epoch_*/summary.json`，写入：

- `full_eval/summary_history.json`

内容包括：

- latest / mean
- best（按不同指标）
- 每轮明细 rounds

---

## 7. 辅助与非主路径模块

以下模块存在，但不在主训练链路 `run.py -> trainer.py` 中直接使用：

- `src/utils/checkpoint.py`
- `src/utils/dataset.py`

它们包含历史/辅助逻辑，当前主路径以 `trainer.py` 与 `dataset.py` 为准。

另外两个辅助训练入口：

- `src/utils/style_classifier.py`：训练 latent 风格分类器
- `src/utils/classify.py`：训练图像风格分类器（small CNN）

---

## 8. 产物目录约定

每次训练 run 主要产物在 `checkpoint.save_dir` 下：

- `epoch_XXXX.pt`
- `logs/training_*.csv`
- `logs/full_eval_epoch_XXXX.log`
- `full_eval/epoch_XXXX/{metrics.csv,summary.json,images/...}`
- `full_eval/summary_history.json`

