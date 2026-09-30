# Infra.md — 架构落地的深度性能优化与实现细节指南

> 本指南针对 620 阶段的理论设计，提供**极度细化到代码行级别**的工程优化方案。
> 我们将覆盖代码中“哪里必须避免循环”、“哪里必须缓存”、“哪些计算可以复用”，
> 以及针对本项目运行环境（WSL）的专属性能陷阱与跨界 I/O 调优方法。

---

## 一、 数据流与跨界 I/O 优化 (特别是 WSL 环境)

在 WSL2 (Windows Subsystem for Linux) 中，跨文件系统（如从 WSL 访问 `G:\` 或 `/mnt/g/`）的 9P 协议性能极差，尤其是读取大量小文件（图片或 `*.pt` 张量）时，会导致极度严重的 I/O 阻塞。

### 1.1 避免离散小文件，启用连续存储 (LMDB / WebDataset)
* **禁止**：训练时直接用 DataLoader 循环读取数万张小尺寸的 `.png` 和独立的 `.pt` 缓存文件（这会把 WSL 的跨界 I/O 卡死）。
* **必须**：
  * **方案 A（最佳）**：将训练数据（内容潜变量 $z_c$、目标潜变量 $z_s$、预提取的 DINO 特征 $F_s$）统一打包成 **LMDB** 格式，或者存为一个巨大的连续文件（如 HDF5 或连续的 `safetensors` block）。
  * **方案 B（次优）**：如果必须读取碎文件，**严禁将数据集放在 `/mnt/g/` 等挂载盘**。必须在 WSL 内部的 ext4 文件系统（如 `~/Latent_Style_Data/`）中创建软链接或将数据复制进去，让 I/O 完全发生在 Linux 原生 VHDX 虚拟硬盘内。

### 1.2 DataLoader 的 WSL 特有坑点
* **共享内存 (shm) 崩溃**：WSL 默认的 `/dev/shm` 极小。当 `num_workers > 0` 且开启 `pin_memory=True` 时，进程间通信极易报 `Bus error` 或 `DataLoader worker exited unexpectedly`。
* **解决**：在 WSL 中运行训练脚本前，必须先挂载扩容 shm，或者在代码中环境变量设置：
  ```bash
  export PYTHONSHAREDALLOCATOR=malloc
  # 或者避免传递极大的张量，改为传递 index 并在 worker 内读取。
  ```

---

## 二、 计算复用与缓存策略 (Caching & Reuse)

在训练循环中，任何确定性的（Deterministic）、不随梯度或 $t$ 变化的数据，都必须被**提前缓存**，绝对不允许每次 forward 都重算。

### 2.1 风格特征 $F_s$ 的彻底缓存
* **错误写法**：
  ```python
  def forward(self, batch):
      # 每一步都在实时算 DINO，白白浪费 30% 算力和几 G 显存
      with torch.no_grad():
          F_s = self.dino(batch["style_image"])
  ```
* **正确实现**：
  由离线脚本一次性计算全库的 DINO 特征并落盘保存。DataLoader 直接返回张量。在 UNet 前向时直接拿来用，DINO 模型在训练阶段**根本不需要加载进 GPU**。

### 2.2 SWD 随机投影矩阵的跨 Batch 复用
* **场景**：在计算单步假想终点 $\hat{x}_1$ 的 Fiberwise SWD 时，需要生成大量随机单位向量进行高维投影。
* **错误写法**：在每次计算 `loss_swd_ss` 时，调用 `torch.randn(dim, num_proj)` 并做 L2 归一化。
* **正确实现**：
  在 `LossComponent` 初始化时，生成固定的一组（或一个 epoch 内共享的）投影矩阵池，保存在 GPU 显存 (`register_buffer`) 中。这避免了每步高频的 RNG（随机数生成）开销和显存分配 / 垃圾回收（GC）引起的 CUDA 同步等待。

---

## 三、 绝对禁止的 Python 级循环 (No Python Loops in Hot Path)

Python 原生的 `for` 循环是深度学习的性能杀手。在 620 的架构中，我们要对计算图中的循环采取“零容忍”。

### 3.1 废除 ODE 展开循环
* 在旧版 `_terminal_swd` 中，有一个显式的 `for i in range(num_steps):` 循环来调用 `model.predict_transport_base`。
* **重构指令**：新架构直接使用单步预测 $\hat{x}_1 = x_t + (1-t)v_{\text{pred}}$。这是一个纯粹的 Tensor 标量乘加操作，完全没有任何循环。

### 3.2 废除 Batch 维度的单独处理
* **场景**：在计算 Fiberwise SWD 时，由于每个图像可能有不同的语义遮罩 (Mask)，历史代码可能会对 Batch 内的每一个样本单独执行逻辑。
* **错误写法**：
  ```python
  batch_loss = 0
  for i in range(B): # 致命的切片循环
      mask_i = mask[i]
      feat_i = feat[i, mask_i]
      batch_loss += compute_swd(feat_i, target[i])
  ```
* **正确实现 (Masked Tensor Operations)**：
  必须将其转化为**填充掩码张量 (Padded Masked Tensors)** 或利用 Scatter/Gather API 并行处理。例如，在投影到 1D 后，直接对维度为 `[B, Proj, HW]` 的张量使用 `torch.sort(dim=-1)`，排序和距离计算全部向量化。

---

## 四、 空间交叉注意力的极致优化

由于我们的 DINO 风格特征 $F_s$ 较长（例如 $16 \times 16 = 256$），且 UNet Decoder 浅层的分辨率极高（例如 $64 \times 64 = 4096$），Cross-Attention 的矩阵 $Q @ K^T$ 尺寸可达 `[B, 4096, 256]`。

### 4.1 必须使用 SDPA (Scaled Dot-Product Attention)
* 坚决抛弃手工写的 `softmax(q @ k.transpose) @ v`。
* 在 `src/lancet_blocks.py` 中强制使用：
  ```python
  import torch.nn.functional as F
  # 如果 is_causal=False 且不需要返回 attn_weights，
  # PyTorch 会底层自动切入 FlashAttention / xFormers，
  # 显存消耗从 O(N^2) 剧降到 O(N)！
  h_out = F.scaled_dot_product_attention(q, k, v)
  ```

### 4.2 消除没必要的维度重排 (Permute/Reshape)
* Tensor 的 `.transpose()` 或 `.permute()` 操作虽然只是改变了 Stride（元数据），但在传入需要连续内存（Contiguous）的底层算子（如 LayerNorm 或某些 Conv）时，会触发昂贵的隐式拷贝。
* **优化规则**：在 UNet 设计时，提前决定好通道格式是 `[B, C, H, W]` 还是 `[B, H, W, C]` (Channel Last)。尽量保持主干流格式不变。对于 Attention 块，如果使用 PyTorch 2.0，推荐尽早 `flatten` 为 `[B, L, C]`，做完 Attention 后再统一 reshape 回去，避免反复的散碎重排。

---

## 五、 显存墙防御与精度策略

### 5.1 混合精度 (AMP) 的精确控制
* **要求**：全局开启 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`。
* **例外（关键）**：
  计算 $v_{\text{pred}}$ 到 $\hat{x}_1$ 以及计算 $\mathcal{L}_{\text{FM}}$ (MSE Loss) 和 SWD 时，必须把张量转回 `torch.float32`。
  ```python
  with torch.autocast('cuda', dtype=torch.bfloat16):
      v_pred = model(z_t, t, F_s)
  
  # 离开 autocast 域，提升到 fp32 计算高精度损失，防止下溢出
  v_pred = v_pred.float()
  loss = F.mse_loss(v_pred, v_target.float())
  ```

### 5.2 激进的 Gradient Checkpointing
* 我们的目标是把显存腾出来，这样可以把 Batch Size 从被迫的 4 提高到 8 或 16。
* 识别出 UNet 中参数最重、Feature Map 最宽的 **DownBlock_2** 和 **MidBlock**。
* 在其前向传播中使用 `torch.utils.checkpoint.checkpoint(module, input, use_reentrant=False)`。这虽然会把这些层的前向时间翻倍（因为反向传播时要重算一次），但能省下至少 30% 的总显存。

---

## 结语：性能也是一种数学正确

如果因为计算速度太慢，一个实验需要跑一周才能看到结果；或者因为显存 OOM 只能使用 Batch Size=2，导致模型梯度方向完全变成了无规律的噪声，那么所有优美的数学推导都将被工程实现的粗糙所摧毁。

严格遵守本指南：
1. **能离线的决不在线 (OT配对, DINO特征)**
2. **能向量化的决不循环 (SWD, Batch维度)**
3. **能省显存的决不硬算 (SDPA, Checkpointing, bfloat16)**
4. **规避环境坑点 (WSL 数据必须内聚)**

这样我们才能在 620 阶段打一场快速、高频、正反馈的实验攻坚战。
