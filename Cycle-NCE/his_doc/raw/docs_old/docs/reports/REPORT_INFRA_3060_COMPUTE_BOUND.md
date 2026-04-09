# 3060 训练 Infra 调查报告（算力高、显存带宽利用偏低）

## 1. 现象与初步判断

你给出的监控样本（GPU 利用率 88%~100%，显存带宽/内存控制器利用率约 37%~65%）更像是 **compute-bound（算力受限）**，而不是典型 memory-bound（带宽受限）：

- memory-bound 常见特征：SM 利用率不高，但显存带宽接近打满。
- 你这里相反：SM 高，带宽中等，说明大量算子更偏“计算密集 + 小核调度密集”。

结合代码实现，当前主因并不是“显存喂不满”，而是损失路径中存在大量高算子强度与多层循环。

## 2. 代码级证据（loss / model / trainer）

### 2.1 SWD + CDF 路径是核心热点（高计算密度 + 多重循环）

在 `calc_swd_loss` 中存在三层循环结构：`patch_sizes`、`num_projections/chunk`、`cdf_bins/bin_chunk`，每层都伴随 `conv2d`/`sigmoid`/累计：

- [`losses.py:229`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:229) `calc_swd_loss(...)`
- [`losses.py:299`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:299) `for p in patch_sizes`
- [`losses.py:307`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:307) `for start in range(0, num_projections, chunk)`
- [`losses.py:310`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:310) 每个 chunk 一次 `F.conv2d`
- [`losses.py:325`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:325) `for b0 in range(0, cdf_bins, bin_chunk)`

并且高频分支会再跑一遍 SWD（额外乘法器）：

- [`losses.py:340`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:340) `calc_hf_swd_loss(...)`
- [`losses.py:558`](/g:/GitHub/Latent_Style/Cycle-NCE/src/losses.py:558) 在 `compute()` 中叠加 HF SWD

这类路径通常表现为：SM 很忙、带宽未满、kernel launch 很碎。

### 2.2 梯度检查点会主动“以算换显存”

训练配置启用 `use_gradient_checkpointing` 时，前向中间激活不保留，反传重算：

- [`trainer.py:136`](/g:/GitHub/Latent_Style/Cycle-NCE/src/trainer.py:136) 读取配置
- [`model.py:349`](/g:/GitHub/Latent_Style/Cycle-NCE/src/model.py:349) `_run_block` 中 `ckpt.checkpoint(...)`
- [`model.py:392`](/g:/GitHub/Latent_Style/Cycle-NCE/src/model.py:392) `_run_decoder` 也 checkpoint

这会进一步提升“算力占比”，降低“带宽占比”，与当前现象一致。

### 2.3 部分路径强制 FP32，会削弱 Tensor Core 利用

上采样模糊在 CUDA 上明确关闭 autocast 并转 `float()`：

- [`model.py:448`](/g:/GitHub/Latent_Style/Cycle-NCE/src/model.py:448) `autocast(..., enabled=False)`
- [`model.py:450`](/g:/GitHub/Latent_Style/Cycle-NCE/src/model.py:450) `F.conv2d(h.float(), ...)`

这会把该段拉回 FP32，增加算力压力。

### 2.4 RTX30 默认禁用 `torch.compile`

trainer 默认在 RTX 30 系列关闭 compile：

- [`trainer.py:184`](/g:/GitHub/Latent_Style/Cycle-NCE/src/trainer.py:184)
- [`trainer.py:190`](/g:/GitHub/Latent_Style/Cycle-NCE/src/trainer.py:190)

这会减少图级融合机会，小算子可能更碎。

### 2.5 DataLoader 在 Windows 默认单进程

如果在 Windows，本项目会默认 `num_workers=0`（除非显式放开）：

- [`run.py:209`](/g:/GitHub/Latent_Style/Cycle-NCE/src/run.py:209)
- [`run.py:224`](/g:/GitHub/Latent_Style/Cycle-NCE/src/run.py:224)

虽然你当前 GPU util 已高，通常不是主瓶颈，但仍建议用日志确认 `data_time_sec` 占比。

## 3. 结论：当前更像“计算路径过重”而非“显存带宽瓶颈”

根因优先级（按影响排序）：

1. SWD/CDF 路径的多层循环 + HF 双路径重复计算  
2. Gradient checkpointing 的重算开销  
3. 局部 FP32 路径（upsample blur）  
4. compile 在 RTX30 默认关闭，融合机会不足  

## 4. 针对性排查与优化建议（按优先级）

## P0：先做“无侵入 A/B”确定主瓶颈（1 天内）

1. 对照关闭 HF SWD：`loss.swd_use_high_freq: true -> false`  
预期：吞吐提升明显（通常最直观），视觉锐度可能下降。

2. 保持指标不变，仅调 chunk：`loss.swd_projection_chunk_size: 64 -> 128/160`  
预期：减少 kernel 启动次数；若 OOM 再回退。

3. CDF 轻量化：
- `swd_cdf_num_bins: 32 -> 16`
- `swd_cdf_sample_size: 256 -> 128`
- `swd_cdf_bin_chunk_size` 适当增大（减少循环轮数）

4. 观察训练日志中 `data_time_sec` vs `compute_time_sec`  
如果 `data_time_sec` 占比 <10%，就别优先折腾 dataloader。

## P1：中风险高收益（建议小规模 smoke test）

1. 关闭 checkpoint 做 A/B（如果 VRAM 允许）
- `training.use_gradient_checkpointing: false`
- 若 OOM，可配合略降 `batch_size` 或 `swd_batch_size`

2. 在 RTX3060 上试开 compile（仅短跑验证稳定性）
- `training.use_compile: true`
- `training.disable_compile_on_rtx30: false`
- 若出现不稳定，再回滚。

3. 对 `upsample_blur` 做 AMP 兼容试验（代码级）
- 去掉强制 FP32 或改为可配置开关，仅在必要时 FP32。

## P2：代码级结构优化（需要改 loss 内核）

1. SWD 投影卷积批处理化/融合化  
目标：减少 `for start in ...` 分块次数，尽量扩大单次工作量。

2. CDF 路径矢量化  
目标：减少 Python 循环层级（尤其 bins 与 sample_chunk 双循环）。

3. 可选切换到排序距离（`distance_mode=sort`）做吞吐对照  
可能降低 CDF 计算负担，但需验证指标一致性。

## 5. 建议的执行顺序（最稳妥）

1. 固定当前 Anchor4 配置，新增 4 个 infra 对照短跑（例如每组 3~5 epoch）  
2. 每组记录：`samples_per_sec`, `compute_time_sec`, `data_time_sec`, GPU util, MEM util  
3. 先筛吞吐，再看视觉/指标回归（LPIPS、CLIP_style）  
4. 保留“效果不掉 + 吞吐提升”组合进入长期训练

## 6. 实操建议（你当前项目可直接用）

- 已有 warmup + grad clip，不建议再动这块（与“白化”问题更相关，而不是吞吐瓶颈）。  
- 先从 SWD 参数下手，不先改模型骨干：  
  - 先关 `swd_use_high_freq` 看上限  
  - 再调 `swd_projection_chunk_size`  
  - 再裁剪 CDF 参数  
- 若希望我下一步直接落地一个 `infra_probe` 生成器（自动产出 A/B 配置 + bat），我可以在 `ablate.py` 里再加一个专门 preset。

