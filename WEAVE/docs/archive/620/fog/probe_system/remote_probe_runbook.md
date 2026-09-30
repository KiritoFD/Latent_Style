# 620 远程 3060 探针 Runbook（占位）

> 本文件为远程 RTX 3060 WSL 环境预留。当前远程环境暂时不可用，待恢复后按本 runbook 执行并回收产物。

## 1. 远程环境信息（待确认）

- SSH：`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`
- 仓库路径：待远程确认（建议 `/mnt/i/Github/Latent_Style/SchrodingerBridge` 或等效路径）
- 数据路径：待远程确认（建议 `/mnt/i/wikiart_distinct5_samam_512_classview`）
- 评估缓存：待远程确认（建议 `/mnt/i/Github/Latent_Style/eval_cache`）
- Python：待远程确认（建议 `python3` 或指定的 conda 环境）

## 2. 远程探针命令（固定后使用）

### 2.1 内部 fog path 探针

```bash
REMOTE_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
RUN_NAME="<run_name>"
CKPT="${REMOTE_ROOT}/src/exp/620_spatial_bridge/${RUN_NAME}/epoch_0008.pt"
CFG="${REMOTE_ROOT}/src/exp/620_spatial_bridge/${RUN_NAME}/config.json"
OUT="${REMOTE_ROOT}/docs/620/fog/probe_system/runs/${RUN_NAME}_fog_path"

cd "${REMOTE_ROOT}"
python3 tools/probe_620_fog_path.py \
  --config "${CFG}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}" \
  --device cuda \
  --sample-count 16 \
  --steps 1 4 8 16 \
  --vae-model ema
```

### 2.2 WFI 评估

```bash
REMOTE_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
RUN_NAME="<run_name>"
CKPT="${REMOTE_ROOT}/src/exp/620_spatial_bridge/${RUN_NAME}/epoch_0008.pt"
OUT="${REMOTE_ROOT}/docs/620/fog/probe_system/runs/${RUN_NAME}_wfi_eval"
TEST_DIR="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
HF_CACHE="/mnt/i/Github/Latent_Style/eval_cache/hf"

cd "${REMOTE_ROOT}"
python3 tools/run_eval_with_wfi.py \
  --checkpoint "${CKPT}" \
  --output "${OUT}" \
  --test-dir "${TEST_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --clip-hf-cache-dir "${HF_CACHE}" \
  --source-dir "${TEST_DIR}" \
  --batch-size 8 \
  --target-chunk-size 2 \
  --vae-decode-batch-size 16 \
  --eval-lpips-chunk-size 4
```

### 2.3 Endpoint Alpha 诊断

```bash
REMOTE_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
RUN_NAME="<run_name>"
CKPT="${REMOTE_ROOT}/src/exp/620_spatial_bridge/${RUN_NAME}/epoch_0008.pt"
CFG="${REMOTE_ROOT}/src/exp/620_spatial_bridge/${RUN_NAME}/config.json"
OUT="${REMOTE_ROOT}/docs/620/fog/probe_system/runs/${RUN_NAME}_endpoint_sweep"

cd "${REMOTE_ROOT}"
python3 tools/probe_620_endpoint_time_sweep.py \
  --config "${CFG}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}" \
  --device cuda \
  --times 0.0 0.125 0.25 0.5 0.75 0.875
```

## 3. 产物回收

远程执行完成后，必须将以下产物同步回本仓库 `docs/620/fog/probe_system/runs/`：

- `*_fog_path/summary.json`
- `*_fog_path/fog_stage_metrics.csv`
- `*_fog_path/fog_sample_summary.csv`
- `*_wfi_eval/summary.json`
- `*_wfi_eval/wfi_benchmark.json`
- `*_wfi_eval/wfi_eval_report.json`
- `*_endpoint_sweep/` 下的 JSON/CSV

同步方式（待远程确认）：
- `scp -P 2222 administrator@100.115.18.62:/mnt/i/... docs/620/fog/probe_system/runs/`
- 或直接在 WSL 中挂载同一磁盘后 `cp -r`。

## 4. 需要记录的远程 Checkpoint 集合

待远程恢复后核对：

| 类别 | 远程路径示例 | 状态 |
|---|---|---|
| 当前最优 | `/mnt/i/.../620_swd16_*/epoch_*.pt` | 待确认 |
| 最近失败 | `/mnt/i/.../620_*_failed/epoch_*.pt` | 待确认 |
| 候选修复 | `/mnt/i/.../620_*_fix/epoch_*.pt` | 待确认 |

## 5. 远程恢复后的第一件事

1. 登录远程 WSL。
2. 确认仓库路径、conda/python 环境、最近训练 run 目录。
3. 将本 runbook 中的 `<run_name>`、路径占位符替换为真实值。
4. 跑一个最小 smoke（1-2 样本）验证探针链路通畅。
5. 将 smoke 结果同步回 `docs/620/fog/probe_system/runs/`。
