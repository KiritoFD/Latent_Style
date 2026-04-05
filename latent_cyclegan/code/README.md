# Latent CycleGAN

这个仓库用于在 SD1.5 VAE latent 上训练、可视化和评估 CycleGAN，并支持 few-shot 消融实验。

## 主要脚本

- [train_latent_cyclegan.py](train_latent_cyclegan.py)
  主训练脚本。当前使用 step/kimg 训练循环，支持断点续训、few-shot 截断、JSONL 日志、checkpoint 和可视化输出。

- [visualize_logs.py](visualize_logs.py)
  训练日志看板。使用 Streamlit + Plotly 读取 JSONL 日志，支持自动刷新，并按标签页查看损失与学习率曲线。

- [run_sweep.py](run_sweep.py)
  few-shot 消融扫描脚本。它会读取 [configs/example.yaml](configs/example.yaml#L4) 中的 `exp_root`，为不同的 `data.max_samples_b` 生成隔离实验目录、配置快照、日志和汇总表。

- [eval/eval_latent_cyclegan.py](eval/eval_latent_cyclegan.py)
  latent-space CycleGAN 评估脚本，用于加载 checkpoint 并输出评估结果。

## 关键配置

主要配置文件是 [configs/example.yaml](configs/example.yaml)。常用字段如下：

- `exp_root`：消融实验输出根目录，供 [run_sweep.py](run_sweep.py) 使用。
- `data.a_dir` / `data.b_dir`：两域 latent `.pt` 数据目录。
- `data.max_samples_a` / `data.max_samples_b`：few-shot 截断开关，`-1` 表示使用全部样本。
- `train.total_kimg`：总训练量，单位为千张图。
- `logging.log_dir` / `logging.log_file`：训练日志输出位置。
- `visualization.out_dir`：训练可视化图片输出位置。

## 数据格式

- 输入数据是 latent `.pt` 文件。
- 每个文件可以是 `[4,H,W]`、`[1,4,H,W]` 的 `torch.Tensor`，或包含 `latent` 键的字典。
- 训练时 A/B 两域独立 shuffle，并通过 infinite dataloader 持续供数。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 运行方式

训练：

```bash
python train_latent_cyclegan.py --config configs/example.yaml
```

训练日志看板：

```bash
python -m streamlit run visualize_logs.py
```

消融扫描 dry-run：

```bash
python run_sweep.py --dry-run
```

消融扫描正式运行：

```bash
python run_sweep.py
```

评估：

```bash
python utils/eval_latent_cyclegan.py --config configs/example.yaml
```

## 输出位置

- 单次训练输出由 `train.checkpoint_dir`、`logging.log_dir`、`visualization.out_dir` 控制。
- sweep 输出会写入 `exp_root/run_all`、`exp_root/run_3000` 等目录。
- sweep 汇总表会保存为 `exp_root/sweep_summary.csv`。
