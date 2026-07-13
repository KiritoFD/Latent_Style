# WEAVE 训练与推理 Infra 优化方案

日期：2026-07-10

## 1. 目标与原则

当前问题不是简单的 batch 太小。8GB 环境已经出现 OOM，而训练峰值功率约 66W、推理端到端速度提升有限，说明
需要定位 GPU timeline 中的真实空洞、同步和小 kernel，而不是继续盲目增大 batch。

优化原则：

1. 先证明瓶颈，再修改实现；
2. 固定图像、checkpoint、步数和质量指标后比较；
3. 功率是症状，kernel timeline 和端到端 wall time 是主证据；
4. 任何加速必须通过输出一致性和四指标回归；
5. 训练与推理分别优化，不用单一技巧覆盖两条路径。

## 2. 当前基线

### 2.1 训练

- Distinct5 packed latent，batch 24，bf16，fused AdamW；
- 删除每 step 无条件 `.item()` 后，从 `36.3s/epoch` 降至约 `29.7s/epoch`；
- 五 epoch wall time `153.0s`；
- GPU 利用率均值约 `84.2%`，但功率均值 `58.8W`、峰值约 `66.7W`；
- profiler 已观察到 bf16 主干与 fp32 loss 之间存在大量 dtype conversion。

### 2.2 推理

- 8-step、750 图总 wall time：`93.88s`；
- bridge：`57.42s`；
- VAE decode：`34.31s`；
- PNG join：约 `0.008s`；
- 聚合 latent 后批量 VAE decode 已将该阶段从约 `53.75s` 降至约 `35s`；
- TorchScript decoder 解决冷加载，但没有显著缩短 decoder compute；
- `torch.compile` warm decoder 可到约 `23.9s`，但首次编译成本较高。

## 3. Phase I0：建立可信测量

每个 benchmark 运行一次 warmup，加五次正式测量，记录 median、p10/p90 和峰值显存。输出必须包含：

- checkpoint SHA256；
- git commit；
- 展平后的 config hash；
- schema 默认值展开后的 effective config，尤其记录未显式出现在 JSON 中的 endpoint mode；
- CUDA、PyTorch、driver、GPU 型号；
- batch、target chunk、num steps、VAE batch；
- bridge、VAE、save、load、总 wall time；
- GPU util、power、clock、显存的外部高频采样；
- 输出 latent checksum 和 16 张固定图的像素/感知回归。

同时建立两个固定 workload：

- `micro-bridge`：固定 2/4/8 个 latent，8-step，不解码；
- `full-750`：完整生成、VAE、落盘和四指标。

## 4. Phase I1：Timeline 定位

使用 `torch.profiler` 和 Nsight Systems 分别采集训练 20 steps、推理 5 个 style chunk。必须回答：

1. GPU kernel 之间是否存在明显 CPU launch gap；
2. Haar DWT/iDWT、norm、transpose、dtype cast 是否形成大量微小 kernel；
3. 是否仍有隐藏的 `.item()`、`cuda synchronize` 或 CPU tensor 分支；
4. H2D 是否与 compute 重叠；
5. VAE batch 之间是否存在 Python/PIL 或 allocator 空洞；
6. 低功率发生在 compute kernel 内，还是发生在 kernel gap；
7. Heun 每一步是否重复构造不变的 style/reference tensor。

验收产物不是一张利用率截图，而是一份按累计 GPU 时间排序的 op 表和一张带 NVTX 区间的 timeline。

## 5. Phase I2：训练优化

按以下顺序执行，每项独立 benchmark：

### I2.1 消除同步与重复统计

- 保持目前只在日志 milestone 执行 `.item()` 的实现；
- 检查 anomaly/debug/statistics 是否在正式训练中关闭；
- 将 epoch 累计指标保留为 GPU tensor，epoch 结束一次性回传；
- 禁止在 forward/loss 路径格式化 tensor 或构造 Python 标量字典。

### I2.2 收敛 dtype 边界

- 标出必须 fp32 的 WCT/eigh、归一化和数值稳定 loss；
- 其余 loss 尽量在单个 autocast 区域完成；
- 将连续的 bf16→fp32→bf16 往返合并为一次边界转换；
- 逐项做 loss 数值和梯度 cosine 回归，不能直接全局改成 bf16。

### I2.3 编译固定主干

- 先只 `torch.compile` velocity backbone，不编译 dataloader、logging 和动态评估；
- 固定 latent shape、batch 和 style token shape；
- 检查 graph break 数量和原因；
- 若 graph break 主要来自 debug 字典，将 debug 与训练主路径分离；
- 编译后要求 100 steps 稳定，无 recompilation storm。

### I2.4 CUDA Graph

仅在 compile 后 shape 完全固定时验证 CUDA Graph：

- 静态输入 buffer；
- 固定 batch 16/24 的独立 graph；
- optimizer step 纳入 graph 前先验证 model-only graph；
- 最后不足 batch 单独走 eager，不能 padding 后污染训练分布。

### I2.5 数据流水线

- packed latent 已较小，不再全量预加载 GPU；
- 测试 pinned host buffer + non-blocking H2D；
- 使用双 buffer 在 stream 上预取下一 batch；
- 只有 timeline 证明 H2D 是 gap 来源时才增加 worker 或预取复杂度。

训练阶段目标：在不改变 batch 和数值协议的前提下，将稳定 epoch 从约 `29.7s` 压到 `24s` 以下；stretch
目标为 `20--22s`。在 compute-heavy 区间，GPU 应接近硬件允许的持续功率；若功率仍低，必须能由 memory-bound
kernel 或 launch gap 的 profiler 证据解释。

## 6. Phase I3：Bridge 推理优化

bridge 占完整 750 图约 61%，优先级高于继续优化 PNG。

### I3.1 缓存不变量

- 每个 target style 的 style tokens/global embedding 只计算一次；
- target style latent 的 DWT 分解和 endpoint 统计只计算一次；
- Heun predictor/corrector 之间复用时间 embedding 和不变 conditioning；
- 固定 time schedule tensor，避免每 chunk Python 构造。

### I3.2 编译单步函数

- 将“DWT → velocity → iDWT”的 solver step 抽成纯 tensor 函数；
- 对固定 batch、固定 64x64 latent、固定 style shape 使用 `torch.compile`；
- 减少 Python 的 8-step 循环控制和 graph break；
- 比较 compiled Euler/Heun step，而不是直接编译整个评估器。

### I3.3 CUDA Graph replay

- 为常用 generation batch 2/4 捕获 8-step graph；
- style conditioning 写入静态 buffer；
- 不在 graph 内执行文件 I/O、PIL、日志或动态分支；
- 分别测量 graph replay 和 eager，避免把 capture 时间计入常驻服务结果。

### I3.4 融合小 kernel

若 timeline 显示 Haar、split/cat、cast、norm 占据大量 launch：

- 优先让 Inductor 融合；
- 无法融合时再考虑自定义 Triton Haar DWT/iDWT；
- 只有累计收益预计超过总 bridge 的 10% 才维护自定义 kernel；
- 自定义 kernel 必须通过正反变换、梯度和数值误差测试。

bridge 目标：从 `57.42s` 降到 `40s` 以下；stretch 目标 `30--35s`。

## 7. Phase I4：VAE 固定导出与流水线

### I4.1 常驻 decoder

- 多 checkpoint 评估使用常驻进程，只加载一次 VAE；
- TorchScript 用于跨进程冷启动；
- warm `torch.compile` 用于连续评估；
- 不在每个 checkpoint 重新编译 decoder。

### I4.2 Decoder benchmark 矩阵

固定 latent，比较：

- eager contiguous；
- eager channels-last；
- TorchScript；
- warm compile；
- batch 8/12/16，在 8GB 上记录峰值显存而非直接拉到 OOM。

### I4.3 阶段流水线

GPU bridge 与 GPU VAE 在同一设备上不应假设能够真正并行。推荐的是有界流水线：

1. bridge 产生一块 latent；
2. latent 进入固定容量 GPU/CPU queue；
3. VAE 按稳定 batch 解码；
4. CPU worker 异步量化和 PNG 编码。

queue 必须有显存预算和 back-pressure。若 GPU 同时驻留 bridge 中间量与过多待解码 latent，会重新触发 OOM。

VAE 目标：从 `34.31s` 降到 `25s` 以下；stretch 目标约 `22s`。

## 8. Phase I5：端到端验收

主目标：完整 8-step、750 PNG 从 `93.88s` 降至 `70s` 以下；stretch 目标 `55--60s`。

每项优化必须满足：

- 无 OOM，连续运行三个 checkpoint；
- 生成数量严格为 750，无重复、无缺图；
- fixed latent checksum 在约定容差内一致；
- 四指标变化小于 evaluator 重复运行误差；
- 不改变 num steps、solver、style refs 或后处理；
- 报告 cold-start 与 warm-service 两套时间，禁止混写。

功率验收不能单独替代性能验收，但在 bridge/decoder 的长 compute 区间，如果硬件功率上限允许，应观察到接近
100W 或接近 board limit 的持续区间。若达不到，报告必须用 timeline 说明是 memory-bound、occupancy 不足还是 launch gap。

## 9. 推荐执行顺序

1. 添加 NVTX 区间和统一 benchmark manifest；
2. 采集训练/bridge/VAE 三条 timeline；
3. 修复剩余同步和 dtype 往返；
4. compile velocity backbone 与单步 solver；
5. 缓存 style/DWT/endpoint 不变量；
6. 验证 CUDA Graph replay；
7. benchmark VAE channels-last/compile/batch；
8. 加入有界 latent→VAE→PNG 流水线；
9. 完整 750 图四指标与速度回归。

## 10. Phase I3.1 实施结果与时间报告（2026-07-11）

### 10.1 实施内容

Phase I3.1（缓存不变量）已完成实施和 benchmark 验证：

- `style_latent` 的 DWT 分解在 ODE 8 步中不变，预计算一次复用
- WCT 协方差 eigh 分解预计算一次（`_precompute_style_wct_stats`）
- 通过 `style_dwt_decomp` 和 `style_wct_stats` 参数在 `integrate_transport` 和 `_apply_endpoint_adain` 之间传递

代码位置：`src/model.py` — `_precompute_style_wct_stats()` + `integrate_transport()` 缓存传递

### 10.2 Benchmark 结果

| 指标 | 值 |
|------|-----|
| Bridge 加速 | **14.1%** |
| 每 batch 节省 | 23.6ms |
| 750 图总节省 | 8.9s |
| 输出数值一致性 | max_diff=0.00（完全一致）|

历史 benchmark 脚本已随旧分支删除；表中数值仅保留为实验记录。

### 10.3 训练时间（train）

| 配置 | batch | epochs | 每 epoch | 5-epoch wall time |
|------|------:|-------:|---------:|------------------:|
| Baseline (B0 T11) | 24 | 5 | 29.7s | 153.0s (2.1min) |
| S0 WEAVE | 16 | 10 | 28.8s | ~290s (5.0min) |

- 删除每 step 无条件 `.item()` 后从 36.3s/epoch 降至 29.7s/epoch
- GPU 利用率均值 84.2%，功率均值 58.8W、峰值 66.7W

### 10.4 推理时间（inf750）

| 阶段 | Baseline | + I3.1 Cache | 节省 |
|------|---------:|-------------:|-----:|
| Bridge | 57.42s | ~48.5s | 8.9s (14.1%) |
| VAE decode | 34.31s | 34.31s | 0s |
| PNG join | ~0.008s | ~0.008s | 0s |
| **总 inf750** | **93.88s** | **~85.0s** | **8.9s** |

- 配置：8-step Euler，750 图，batch_size=2，target_chunk_size=1
- VAE decode 已使用批量解码（从 53.75s 降至 34.31s）
- `torch.compile` warm decoder 可到 23.9s，但首次编译成本较高（未启用）

### 10.5 2026-07-12 远端一致性与高 batch 验证

- 3060 远端仓库根目录曾残留 `config_schema.py`、`blocks620.py`、`model620.py`、`spectral*.py` 等影子模块。交互式 `python -` 从根目录启动时会优先导入它们，而非 `src/`，可导致配置契约与实际训练不一致。
- 已删除这些根级影子文件；根目录 + `PYTHONPATH=src` 与 `python src/run.py` 均解析到同一份 `src/config_schema.py`。
- `batch=144` smoke 和训练均稳定：模型清理后为 `873,680` 参数，反传 reserved memory 约 `10.16GB`；稳态训练达到 `97–100%` GPU 利用率、`136–139W`、约 `273–285 samples/s`。
- 旧 baseline checkpoint 的已退役 global-style projection 现在在推理前显式过滤，25 图兼容性评估成功，避免非严格加载掩盖真正的不匹配。

### 10.6 后续优化优先级

| Phase | 目标 | 预期收益 | 状态 |
|-------|------|---------|------|
| I3.1 缓存不变量 | Bridge -15% | 8.9s | **已完成** |
| I3.2 编译单步函数 | Bridge -20% | ~11s | 待实施 |
| I4.1 VAE compile | VAE -30% | ~10s | 待实施 |
| I3.3 CUDA Graph | Bridge -10% | ~5s | 待实施（I3.2 后） |

Stretch 目标：inf750 从 93.88s → 70s 以下（I3.1+I3.2+I4.1 合计预期 -30s）
