# 620 本地探针 Runbook

> 本文件规定如何在本地（Windows / WSL 均可）对任意 620 checkpoint 运行内部探针、WFI 评估与 endpoint alpha 诊断。

## 1. 环境准备

- 仓库根目录：`G:\GitHub\Latent_Style\SchrodingerBridge`
- Python 环境需已安装仓库依赖（`torch`、`PIL`、`numpy`、`tqdm` 等）。
- 确保 VAE、DINO cache、latent cache 路径在本地可解析。

## 2. 内部探针：定位白化首次出现的位置

### 2.1 端到端 fog path 探针

使用 `tools/probe_620_fog_path.py` 对真实 checkpoint 进行 source / target / endpoint / integrate 各阶段统计：

```powershell
python tools/probe_620_fog_path.py `
  --config src/exp/620_spatial_bridge/<run_name>/config.json `
  --checkpoint src/exp/620_spatial_bridge/<run_name>/epoch_0008.pt `
  --output-dir docs/620/fog/probe_system/runs/<run_name>_fog_path `
  --device cuda `
  --sample-count 8 `
  --steps 1 4 8 16 `
  --vae-model ema
```

输出：
- `summary.json`：含 `stage_summary`（source_latent / target_latent / predict_endpoint_t0 / integrate_nfe_*）与 `headline` 比率。
- `fog_stage_metrics.csv`、`fog_sample_summary.csv`：逐样本、逐阶段指标。

### 2.2 读取训练时内部探针

训练过程中 `src/model620.py` 与 `src/blocks620.py` 会把统计量写入 `model.last_debug`。训练日志通过 `src/trainer.py` 的 `_bridge_probe_stats()` 自动抽取并写入：

- 训练 CSV：`src/exp/.../logs/training_*.csv`
- 检查点 `metrics` 字段：`epoch_*.pt["metrics"]`
- `logger.info` 控制台输出（每 `log_interval` 步）

关键字段前缀：
- `bridge_latent_input_*`：输入 latent 统计
- `block{N}_output_*`：第 N 层 block 输出统计
- `bridge_cross_attn_entropy` / `bridge_actual_attn_entropy` / `bridge_gate_mean` / `bridge_gate_std` / `bridge_style_gate_value`
- `bridge_film_gamma_abs` / `bridge_film_beta_abs` / `bridge_pre_film_gamma_abs` / `bridge_pre_film_beta_abs` / `bridge_style_bias_abs`
- `bridge_sa_input_std` / `bridge_sa_output_std` / `bridge_ca_input_std` / `bridge_ca_output_std`
- `bridge_endpoint_output_*` / `bridge_endpoint_low_*` / `bridge_endpoint_high_*`
- `bridge_velocity_*`
- `bridge_endpoint_alpha` / `bridge_endpoint_high_alpha`

### 2.3 手动在 Python 中读取探针

```python
import torch
from model620 import SpatialBridge620
from config_schema import load_config

cfg = load_config("src/exp/620_spatial_bridge/<run_name>/config.json")
model = SpatialBridge620(cfg.model, cfg.bridge)
model.load_state_dict(torch.load("src/exp/.../epoch_0008.pt", map_location="cpu")["model_state_dict"])
model.eval()

# 运行一次前向
with torch.no_grad():
    x = torch.randn(1, cfg.model.latent_channels, 64, 64)
    source = torch.randn_like(x)
    target = torch.randn_like(x) + 0.3
    v = model(x, source=source, target_latent=target, t=0.5, style_id=0)

print(model.last_debug.keys())
print(model.last_debug["endpoint_alpha"])
print(model.last_debug["cross_attn_entropy"])
```

## 3. WFI 评估

### 3.1 跑完整 eval + WFI

```powershell
python tools/run_eval_with_wfi.py `
  --checkpoint src/exp/620_spatial_bridge/<run_name>/epoch_0008.pt `
  --output docs/620/fog/probe_system/runs/<run_name>_wfi_eval `
  --test-dir G:/wikiart_distinct5_samam_512_classview/test `
  --cache-dir G:/Latent_Style/eval_cache `
  --clip-hf-cache-dir G:/Latent_Style/eval_cache/hf `
  --source-dir G:/wikiart_distinct5_samam_512_classview/test `
  --batch-size 8 `
  --target-chunk-size 2 `
  --vae-decode-batch-size 16 `
  --eval-lpips-chunk-size 4
```

产物：
- `<output>/summary.json`：包含 `appearance_deltas`（生成图与 source/target 的亮度/对比度/动态范围/饱和度/WFI 差异）和 `wfi_benchmark`。
- `<output>/wfi_benchmark.json`：完整 WFI 统计。
- `<output>/wfi_eval_report.json`：汇总关键指标。

### 3.2 仅对已有生成图跑 WFI

```powershell
python -m src.utils.wfi docs/620/fog/probe_system/runs/<run_name>_wfi_eval --source-dir G:/wikiart_distinct5_samam_512_classview/test
```

注意：`src/utils/wfi.py` 会读取 `<eval_dir>/images/*.png` 并写入 `<eval_dir>/wfi_benchmark.json`，同时追加到 `<eval_dir>/summary.json`。

## 4. Endpoint Alpha 诊断

### 4.1 使用 `tools/probe_620_endpoint_time_sweep.py`

查看不同 `t` 下 endpoint 质量：

```powershell
python tools/probe_620_endpoint_time_sweep.py `
  --config src/exp/620_spatial_bridge/<run_name>/config.json `
  --checkpoint src/exp/620_spatial_bridge/<run_name>/epoch_0008.pt `
  --output-dir docs/620/fog/probe_system/runs/<run_name>_endpoint_sweep `
  --device cuda `
  --times 0.0 0.125 0.25 0.5 0.75 0.875
```

### 4.2 手动计算 endpoint alpha

`SpatialBridge620.compute_endpoint_alpha(endpoint, source, target)` 已内置：

\[
\alpha = \frac{\|\text{endpoint} - \text{source}\|_2}{\|\text{target} - \text{source}\|_2 + \epsilon}
\]

在训练或推理脚本中：

```python
endpoint = model.predict_endpoint(x, t=0.0, style_id=target_id)
alpha = model.compute_endpoint_alpha(endpoint, source, target)
```

或在训练时直接读取 `model.last_debug["endpoint_alpha"]`（当 forward 收到 `source` 与 `target_latent` 时）或 `bridge_endpoint_alpha_trainer`（由 trainer 从 `last_endpoint` 计算）。

## 5. 需要记录的 Checkpoint 集合

每次诊断必须覆盖以下三类 checkpoint：

| 类别 | 示例路径 | 用途 |
|---|---|---|
| 当前最优 | `src/exp/620_spatial_bridge/<best>/epoch_*.pt` | 基线，判断修复是否退化 |
| 最近失败 | `src/exp/620_spatial_bridge/<failed>/epoch_*.pt` | 复现白化症状，验证指标是否敏感 |
| 候选修复 | `src/exp/620_spatial_bridge/<fix>/epoch_*.pt` | 验证修复是否把白化指标压下去 |

记录方式：
- 在 `docs/620/fog/probe_system/runs/<run_name>_<probe_type>/` 下保存所有 probe 输出。
- 每次运行后把 `<run_name>`、checkpoint 路径、关键指标写入 `docs/620/fog/probe_system/index.md`（手动维护）。

## 6. 输出目录规范

```
docs/620/fog/probe_system/
├── metrics_catalog.md          # 指标定义
├── local_probe_runbook.md      # 本文件
├── remote_probe_runbook.md     # 远程 3060 占位
├── index.md                    # 每次探针运行的索引（手动更新）
└── runs/
    ├── <run_name>_fog_path/         # probe_620_fog_path.py 输出
    ├── <run_name>_endpoint_sweep/   # probe_620_endpoint_time_sweep.py 输出
    ├── <run_name>_wfi_eval/         # run_eval_with_wfi.py 输出
    └── ...
```

## 7. 快速检查清单

- [ ] 已确认 config.json 与 checkpoint 属于同一训练 run。
- [ ] 已确认 `--test-dir` 与训练/评估使用的一致。
- [ ] 已保存生成图（`run_eval_with_wfi.py` 默认 `--save_generated_images`）。
- [ ] 已检查 `summary.json` 中 `appearance_deltas` 与 `wfi_benchmark` 是否存在。
- [ ] 已检查训练 CSV 中是否出现 `bridge_*` 字段。
- [ ] 已将关键结果摘要写入 `docs/620/fog/probe_system/index.md`。
