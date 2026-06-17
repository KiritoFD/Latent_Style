# Infra 优化方案：训练与推理

> WSL (1TB EXT4, 917G 可用) + RTX 3060 12GB

---

## 一、训练 Infra

### 1.1 WSL 9P I/O 瓶颈：代码和 checkpoint 迁移到 EXT4

训练和 eval 当前都在 `/mnt/i/` 下运行。WSL2 通过 9P 协议访问 Windows NTFS 挂载盘，小文件随机读写极慢（Python import、checkpoint save、CSV flush、`__pycache__` 编译等）。WSL 根分区有 **917G 可用**，完全足够。

**操作步骤**：
```bash
# 1. 复制代码到 WSL 原生 EXT4
rsync -a /mnt/i/Github/Latent_Style/ ~/Latent_Style/

# 2. 在 config 中把 save_dir 指向 EXT4
#    例如 "save_dir": "/home/user/exp/aaai2027_phase2_xxx"

# 3. 训练结束后，把 exp 结果 rsync 回 Windows 盘做冷备
rsync -a ~/exp/ /mnt/i/Github/Latent_Style/exp/
```

**数据集**可以保留在 `/mnt/i/`（大文件顺序读取 9P 尚可接受），但代码和 checkpoint 必须在 EXT4。

**预期收益**：
- checkpoint save/load 速度提升 3-5x
- Python module import 提速（训练启动从 ~5s 降至 ~1s）
- 整体 eval 中的 I/O 开销从 ~10s 压缩到 ~3s

### 1.2 `num_workers` 的 WSL 暗病

当前已设置 `num_workers=0` 规避 WSL multiprocessing VRAM 泄漏，这是正确的。

**可尝试的替代**：
```python
DataLoader(..., num_workers=1, persistent_workers=False, prefetch_factor=2)
```
如果 VRAM 仍然泄漏，保持 `num_workers=0`。

### 1.3 训练与评估的异步化

当前 eval 是同步阻塞的（每个 epoch 结束后停训练 → 跑 eval → 恢复训练）。在单卡 3060 上，一次 full eval 约 241s，严重拖慢训练。

**建议**：
- 减少 eval 频率：前 5 个 epoch 每 2 个 eval 一次
- 或者只在 loss 收敛拐点处触发 eval（根据 `flow_loss` 的 EMA 下降率判断）

---

## 二、推理 Infra

### 2.1 推理步数与 SDE 噪声的权衡

当前 `num_steps=12`。从实验数据看，lancet_generation 占 eval 总时间的 58%（141s/241s）。

**建议**：
- 探索 `num_steps=8` 是否在 CLIP-S/LPIPS 上有显著退化
- 如果退化 < 2%，可以将推理时间压缩 33%

### 2.2 VAE 解码：编译加速

当前 `vae_compile_decoder=False`。开启后首次会花 ~30s 编译，但后续调用会大幅加速。

**建议**：在 config 中设置：
```json
"full_eval": {
    "vae_compile_decoder": true,
    "vae_compile_mode": "reduce-overhead"
}
```
配合 `runtime_model_cache=true`（已默认开启），编译后的 VAE 会跨 eval 复用。

---

## 三、已完成的 Eval Infra 改动（交接说明）

以下改动已直接提交到代码中：

### 3.1 `source_latent_cache` 默认开启

**文件**：`src/config_schema.py` L66
**改动**：`"source_latent_cache": False` → `"source_latent_cache": True`
**效果**：首次 eval 时，所有源图的 VAE encode 结果会缓存到 `eval_cache/source_latents_*.pt`。后续 epoch 直接从磁盘加载，跳过 VAE encode 和源图片读取。
**预期收益**：省去 ~9.2s/eval（encode_inversion 5.7s + source_load_to_device 3.5s）。首次 eval 额外花 ~10s 构建缓存。
**兼容性**：当缓存的源图集合发生变化时（如增减了评估图片），缓存会自动失效并重建（通过 `_source_latent_cache_hash` 校验）。

### 3.2 LPIPS 推理模式升级

**文件**：`src/utils/run_evaluation.py` L615, L644
**改动**：`torch.no_grad()` → `torch.inference_mode()`
**效果**：`inference_mode` 比 `no_grad` 更激进——它额外禁用了 version counting 和 view tracking，对纯推理路径来说是严格更快的。
**兼容性**：`_lpips_forward_safe` 只读输入、只写输出，不存在 in-place 修改或 view 依赖，因此 `inference_mode` 是安全的。

### 3.3 实际时间分布参考

基于 `smoe_translator_k070` epoch_0008 的 profiling 数据：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| `lancet_generation` | 141.0s | 58.5% |
| `vae_decode` | 55.2s | 22.9% |
| `eval_metrics_loop` (LPIPS+CLIP) | 23.2s | 9.6% |
| `encode_inversion` | 5.7s | 2.4% ← 已被 latent cache 消除 |
| `source_load_to_device` | 3.5s | 1.5% ← 已被 latent cache 消除 |
| 其他 (I/O, load, copy) | ~12.4s | 5.1% |
| **wall_total** | **241.0s** | 100% |