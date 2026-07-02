# 像素空间 SFM 对比实验计划

## 目标

实证 VAE latent 的语义优势（不只是计算节省），支撑 §3.2 "Why Latent Space" 的论点。

## 可行性结论（search agent 调研）

- **256×256 像素空间**：当前 global self-attention 在 128×128 post-DWT feature map 上单层注意力矩阵 1GB（batch 1），4 层 4GB+，**任何 batch 都 OOM**。
- **必须先实现 windowed attention**（window=8），才能在 12GB 跑 256。
- 训练时间估计：20-50 min / 5 epochs / batch 4-8（vs latent 3 min）。
- 风险：中高（主要是 windowed attention 实现 + 256 vs 512 分辨率匹配）。

## 分级方案

### 方案 A（推荐）：256×256 + windowed attention
1. 实现 `window_attn_window_size` 在 `src/blocks620.py` `SpatialBridgeBlock620`
2. 添加像素数据集 loader（`.jpg/.png` → 256×256 → `[-1,1]`）
3. 添加 `data_mode: "latent"|"pixel"` 配置
4. eval 旁路 VAE，像素输出 256→512 上采样后评测
5. 新 config `620_pixel_256.json`
6. 5 epoch 训练 + 750 对评测
7. 作为 Table 1 第 13 行 "SFM-Pixel-256"

工作量：~2-3 天代码 + 1-2 小时训练评测

### 方案 B（fallback）：128×128 像素空间
- post-DWT 64×64 = 4096 tokens，注意力矩阵 16MB/头/样本，batch 8-16 可跑
- 无需 windowed attention，只需 dataset loader + eval 旁路
- 工作量：~1 天
- 缺点：分辨率匹配更差（128 vs 512）

### 方案 C（最弱）：仅文献论证
- 当前论文已采用（§4.3 "Latent vs. pixel space" 段落引用 Rombach 2022）
- 审稿人会问"为什么不直接跑"，弱

## 推荐执行顺序

1. **先跑方案 B（128）**：1 天可出结果，先有实证
2. **如果 128 结果支持论点**（pixel < latent），再补方案 A（256）加固
3. **如果 128 结果不支持**（pixel ≥ latent），重新评估论文叙事

## 代码改动清单（方案 B - 128）

1. `src/utils/dataset.py`：添加 `_load_image_file(path, target_size=128)`，扩展 `_scan_style_files` 支持 `.jpg/.png`
2. `src/config_schema.py`：`DataConfig` 添加 `data_mode: "latent"|"pixel"` 字段
3. `src/utils/run_evaluation.py`：添加 `pixel_mode` 分支，旁路 VAE，输出 128→512 上采样后评测
4. `configs/620_pixel_128.json`：新 config，`latent_channels: 3`，`data_mode: "pixel"`，`batch_size: 8`
5. 无需改 blocks620.py（128 post-DWT 64×64，global attn 可跑）

## 代码改动清单（方案 A - 256，增量于方案 B）

6. `src/blocks620.py`：`SpatialBridgeBlock620` 实现 windowed self-attention（window=8）
7. `configs/620_pixel_256.json`：`batch_size: 4`，`use_gradient_checkpointing: true`

## 评测协议

- 同一 750 对 Distinct5-WikiArt 测试集
- 像素输出上采样到 512 后评测 CLIP-S / LPIPS（消除分辨率变量）
- 作为 Table 1 新行 "SFM-Pixel-256" 或 "SFM-Pixel-128"
- 期望结果：pixel SFM 的 CLIP-S/LPIPS 均劣于 latent SFM，实证 VAE 语义优势

## 论文影响

- §3.2 "Why Latent Space"：从纯文献论证升级为"文献 + 实证对比"
- §4.2 Main Results：Table 1 增加一行
- §4.3 Efficiency：可加一句 "pixel-space SFM at 256 trains in X min but underperforms latent by ΔCLIP-S / ΔLPIPS"
- Supplement：增加 "Pixel-Space Ablation" 小节

## 待用户确认

- 先跑方案 B（128，1 天）还是直接上方案 A（256，2-3 天）？
- 还是先提交当前论文版本，pixel 实验作为后续补充？
