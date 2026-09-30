# 620 Spatial Bridge T5多模态实验交接文档

本篇文档记录了在 620 Spatial Bridge 风格迁移实验中，将文本分支的 Encoder 替换为 T5-base (`google/t5-v1_1-base`) 的全套方案、所作修改以及后续的运行与监控指南。

---

## 1. 我们做了什么 (What We Did)

为了将默认的 CLIP 文本编码器升级为表征能力更强的 T5-base 编码器，并克服远程网络下载瓶颈，我们实现了以下方案：

### A. 实现 T5 离线 Token 缓存提取脚本
在 `SchrodingerBridge/tools/experiments/build_offline_t5_text_cache.py` 中编写了 T5 embedding 离线提取工具，支持从 ModelScope 本地缓存加载 `t5-v1_1-base` 权重。
- 该脚本读取风格数据集的 `train_style_captions.jsonl`，将 4647 个 caption 编码成 `last_hidden_state`。
- T5-base 产生的隐藏层特征维度为 **`768`**，特征长度设为 **`256`**。

### B. 解决远程下载与传输瓶颈
- 鉴于直接从远程拉取 3.5GB 缓存文件（SCP 传输）或者在远程从外网下载 T5 模型速度较慢，我们通过直接在远程 Windows 主机的缓存目录 `/mnt/c/Users/administrator/.cache/modelscope/hub/models/google/t5-v1_1-base` 下定位到已经下载好的 T5-base 权重。
- 在远程 WSL 侧安装 `sentencepiece` 后，直接执行 `remote_build_cache.sh` 脚本在远程本机快速生成了 3.5GB 的 `.pt` 缓存文件，保存在：
  `/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/t5_text_cache_wikiart_distinct5_samam_512.pt`

### C. 创建与同步训练配置
- 创建了专用的训练配置文件 `configs/620_swd16_t5_base_multimodal_vlen004_b40_remote.json`，配置如下关键参数：
  - `"style_text_encoder": "t5_v1_1_base"`
  - `"style_text_dim": 768` (直接对应 T5-base 维度)
  - `"style_text_max_length": 256`
  - `"style_caption_path": "/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/t5_text_cache_wikiart_distinct5_samam_512.pt"`
- 同步了 local 分支 `codex/620-spatial-bridge` 的更新文件（包括 `blocks620.py` 内部支持 `style_shortcut_alpha` 等 Phase 4 实验特性的代码）到远程机器上，消除了 `TypeError`。

### D. 启动远程后台训练
- 编写了 `remote_train.sh` 启动脚本，并在远程创建了持久的 `tmux` 会话 `620_t5_base_train`，成功拉起了训练任务。

---

## 2. 后续如何使用 (How to Use & Monitor)

### A. 监控训练状态
训练正在远程 WSL 的 `tmux` 后台会话中运行，日志会实时输出。

1. **查看最新日志**:
   ```bash
   # 在远程 WSL 终端中运行：
   tail -n 100 -f /mnt/i/Github/Latent_Style/exp/620_t5_base_multimodal_train.log
   ```
2. **附着到 tmux 终端进行交互**:
   ```bash
   # 在远程 WSL 终端中运行：
   tmux attach -t 620_t5_base_train
   ```
   *(退出 tmux 附着时按 `Ctrl+B` 然后按 `D` 键)*
3. **检查 GPU 显存占用情况**:
   ```bash
   nvidia-smi
   ```

### B. 主要路径清单
- **训练配置文件**: `SchrodingerBridge/configs/620_swd16_t5_base_multimodal_vlen004_b40_remote.json`
- **训练日志文件**: `/mnt/i/Github/Latent_Style/exp/620_t5_base_multimodal_train.log`
- **离线 T5 text cache**: `/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/t5_text_cache_wikiart_distinct5_samam_512.pt`
- **WSL 上的 T5 权重路径**: `/mnt/c/Users/administrator/.cache/modelscope/hub/models/google/t5-v1_1-base`
- **后台启动脚本**: `remote_train.sh`
