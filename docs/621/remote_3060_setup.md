# 621 远程3060 WSL环境与实验配置

> 建立日期: 2026-06-21

---

## 1. 远程机器信息

| 项目 | 值 |
|------|-----|
| IP | 100.115.18.62 |
| SSH端口 | 2222 |
| 用户 | administrator |
| GPU | NVIDIA RTX 3060 12GB |
| 驱动 | 581.08 |
| CUDA | 13.0 |
| VRAM | 11.5GB 可用 |
| 温度 | 44°C (空闲) |
| OS | Windows + WSL Ubuntu-26.04 |

---

## 2. 环境配置

### 2.1 项目路径

| 位置 | 路径 |
|------|------|
| Windows项目 | `C:\Users\Administrator\src\` |
| WSL挂载 | `/mnt/g/GitHub/Latent_Style/SchrodingerBridge` |
| Checkpoint | `C:\Users\Administrator\exp\` |
| Baseline | `C:\baseline\Baroque\outputs\epoch_999\` |

### 2.2 Python环境

```bash
# WSL中
python --version  # 3.12.10
pip list  # 检查torch, torchvision等
```

### 2.3 依赖安装

```bash
# 在WSL中
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
pip install -r requirements.txt
```

---

## 3. 实验运行流程

### 3.1 同步代码到远程

```bash
# 本地PowerShell
$remote = "administrator@100.115.18.62"
$port = 2222

# 同步src
scp -P $port -r "G:\GitHub\Latent_Style\SchrodingerBridge\src" "${remote}:C:\Users\Administrator\src\"

# 同步tools
scp -P $port -r "G:\GitHub\Latent_Style\SchrodingerBridge\tools" "${remote}:C:\Users\Administrator\tools\"

# 同步config
scp -P $port "G:\GitHub\Latent_Style\SchrodingerBridge\configs\620*.json" "${remote}:C:\Users\Administrator\configs\"
```

### 3.2 运行训练

```bash
# SSH到远程
ssh -p 2222 administrator@100.115.18.62

# 在WSL中运行
wsl -d Ubuntu-26.04 -- bash -c '
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
python src/run.py --config configs/620_film_v5_endpoint_film_hd512.json --epochs 5
'
```

### 3.3 运行评估

```bash
# 在WSL中
wsl -d Ubuntu-26.04 -- bash -c '
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
python run_evaluation.py \
  --checkpoint exp/620_spatial_bridge/<run>/epoch_0001.pt \
  --config exp/620_spatial_bridge/<run>/config.json \
  --output_dir exp/620_spatial_bridge/<run>/full_eval_wfi/
'
```

### 3.4 运行探针

```bash
# 在WSL中
wsl -d Ubuntu-26.04 -- bash -c '
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
python tools/probe_620_fog_whiteness_index.py \
  --eval_dir exp/620_spatial_bridge/<run>/full_eval_wfi/epoch_0001/ \
  --output probe_results/wfi.json
'
```

---

## 4. 3060优化配置

### 4.1 Batch Size优化

| 配置 | batch_size | accumulation | 有效batch | VRAM |
|------|-----------|-------------|-----------|------|
| 保守 | 2 | 16 | 32 | ~7GB |
| 平衡 | 4 | 8 | 32 | ~9GB |
| 激进 | 4 | 16 | 64 | ~10GB |

### 4.2 精度优化

```python
# 推荐配置
amp_dtype = "bf16"  # 3060支持bf16
# 或
amp_dtype = "fp16"  # 如果bf16不稳定
```

### 4.3 Compile优化

```bash
# 如果PyTorch 2.0+
torch.compile(model, mode="reduce-overhead")
```

---

## 5. 监控与调试

### 5.1 GPU监控

```bash
# 远程
ssh -p 2222 administrator@100.115.18.62 "nvidia-smi"
```

### 5.2 训练日志

```bash
# 实时查看日志
ssh -p 2222 administrator@100.115.18.62 \
  "wsl -d Ubuntu-26.04 -- tail -f /mnt/g/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/<run>/train.log"
```

### 5.3 Checkpoint管理

```bash
# 查看checkpoint
ssh -p 2222 administrator@100.115.18.62 \
  "dir C:\Users\Administrator\exp\620_spatial_bridge\<run>\ /b"

# 下载checkpoint到本地
scp -P 2222 administrator@100.115.18.62:"C:\Users\Administrator\exp\620_spatial_bridge\<run>\epoch_0001.pt" .
```

---

## 6. 常见问题

### 6.1 WSL挂载路径问题

如果WSL无法访问G盘:
```bash
wsl -d Ubuntu-26.04 -- sudo mkdir -p /mnt/g
wsl -d Ubuntu-26.04 -- sudo mount -t drvfs G: /mnt/g
```

### 6.2 CUDA OOM

```bash
# 降低batch_size
# 或使用gradient checkpointing
# 或使用更小的model (dim=96 instead of 128)
```

### 6.3 训练卡住

```bash
# 检查进程
wsl -d Ubuntu-26.04 -- ps aux | grep python

# 检查GPU使用
nvidia-smi

# 强制停止
wsl -d Ubuntu-26.04 -- kill -9 <pid>
```

### 6.4 网络断开

```bash
# 使用tmux保持会话
wsl -d Ubuntu-26.04 -- tmux new -s train

# 断开后重连
wsl -d Ubuntu-26.04 -- tmux attach -t train
```
