# Remote Sidecar Recovery Status

Date: 2026-06-05

Scope: sidecar-only recovery snapshot for `same-cost` K-longer eval closure and `img2img-turbo` smoke readiness. This note does not touch the main paper text.

## Post-check Update

The snapshot below was taken before the next local audit. A later direct check from the
main rollout found that the tmux holder had died, with no surviving `k_longer`
writer processes and only `epoch_0005/images + metrics.csv` retained. The existing
helper was relaunched at `2026-06-05T01:58+08:00`:

```bash
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 \
  "wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/run_k_longer_eval_5_8_artfid_tmux.sh"
```

Immediate verification after relaunch:

- tmux helper status: `LIVE`
- pane pid: `459`
- free space after helper cleanup/restart: about `709 MB`
- log restarted from `Phase 1: Generation` on `epoch_0005`

## 1. K-Longer `full_eval_artfid` Status

Remote root:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8\full_eval_artfid`

Snapshot command family:

- `ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "dir /b I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8\full_eval_artfid\epoch_0005"`
- `ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader"`
- `ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/k_longer_tmux_status.sh"`
- `ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "wsl -d Ubuntu-26.04 -- ps -ef"`

Status table:

| epoch | `summary.json` | `metrics.csv` | `aggregate_targetwise_artfid.json` | current note |
| --- | --- | --- | --- | --- |
| `0005` | `N` | `Y` | `N` | active tmux-held eval is still in `Generating Summary...` according to `remote_k_longer_eval_5_8_artfid.log` |
| `0006` | `N` | `N` | `N` | active foreground WSL eval is running from checkpoint `epoch_0006.pt` |
| `0007` | `N` | `N` | `N` | not started in the current clean rerun |
| `0008` | `N` | `N` | `N` | not started in the current clean rerun |

Current process/session snapshot:

- GPU snapshot during inspection: `3565 MiB / 12288 MiB`, `0% util`, `11.61 W`
- tmux helper status: `LIVE`, pane pid `1405`
- active WSL foreground holder:
  - `bash /mnt/c/Users/Administrator/k_longer_eval_5_8_artfid_clean_20260605.sh`
  - child `bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_longer_eval_5_8_artfid.sh`
  - active python on `epoch_0006.pt`
- active tmux-held holder:
  - `tmux new-session -d -s k_longer_eval_5_8_artfid ...`
  - active python on `epoch_0005.pt`

Interpretation:

- the run is not dead; do **not** launch another recovery holder right now
- the current state is two concurrent holders, one on `epoch_0005` and one on `epoch_0006`
- no `full_eval_artfid` epoch is paper-safe yet because none has both `summary.json` and `aggregate_targetwise_artfid.json`

## 2. Minimal Recovery Commands

Use only if the current session is dead:

1. Check tmux/session state:

```bash
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 \
  "wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/k_longer_tmux_status.sh"
```

2. If that returns `DEAD`, restart the existing helper that clears `full_eval_artfid/epoch_0005..0008` and relaunches tmux:

```bash
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 \
  "wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/run_k_longer_eval_5_8_artfid_tmux.sh"
```

3. Tail the holder log:

```bash
ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 \
  "type I:\Github\Latent_Style\SchrodingerBridge\_codex_tmp\remote_k_longer_eval_5_8_artfid.log"
```

Important:

- if tmux is `LIVE`, do not relaunch the helper again
- if you intentionally want a hard reset, first stop existing WSL eval holders for `aaai2027_longer_train_k_seed42_b44_e8`; otherwise you will stack duplicate writers again

## 3. `img2img-turbo` Smoke Readiness

Repo root:

- `G:\GitHub\Latent_Style\Related_Works\repos\cyclegan_turbo\img2img-turbo`

Closed entrypoints:

- train entry: `src/train_cyclegan_turbo.py`
- unpaired dataset contract: `src/my_utils/training_utils.py`
- single-image inference entry: `src/inference_unpaired.py`

Required dataset layout for one Distinct5 target-specific adapter:

- `train_A =` union of the other 4 domains
- `train_B =` target domain
- `test_A =` fixed source-side test images under the same target-specific split
- `test_B =` target-domain held-out references
- `fixed_prompt_a.txt`
- `fixed_prompt_b.txt`

The repo expects:

```text
<dataset_root>/
  train_A/
  train_B/
  test_A/
  test_B/
  fixed_prompt_a.txt
  fixed_prompt_b.txt
```

Dependency surface to preinstall:

- `accelerate`
- `diffusers`
- `transformers`
- `torch` / `torchvision` / `torchaudio`
- `xformers`
- `peft`
- `lpips`
- `clean-fid`
- `vision_aided_loss`
- `wandb`

Source files:

- `requirements.txt`
- `environment.yaml`

## 4. Minimal Local-4070 Smoke Launch Template

Assumption:

- run in a local Linux/WSL env rooted at `G:\GitHub\Latent_Style\Related_Works\repos\cyclegan_turbo\img2img-turbo`
- dataset already materialized for one target, for example `target=Hayao`

Template:

```bash
cd /mnt/g/GitHub/Latent_Style/Related_Works/repos/cyclegan_turbo/img2img-turbo
export NCCL_P2P_DISABLE=1

accelerate launch --main_process_port 29531 src/train_cyclegan_turbo.py \
  --pretrained_model_name_or_path "stabilityai/sd-turbo" \
  --dataset_folder "/mnt/g/GitHub/Latent_Style/Related_Works/datasets/distinct5_turbo_to_<TARGET>" \
  --output_dir "/mnt/g/GitHub/Latent_Style/Related_Works/runs/cyclegan_turbo_distinct5_smoke/<TARGET>" \
  --train_img_prep "resize_512" \
  --val_img_prep "resize_512" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 20 \
  --validation_steps 10 \
  --validation_num_images 8 \
  --checkpointing_steps 10 \
  --dataloader_num_workers 0 \
  --tracker_project_name "distinct5_turbo_smoke_<TARGET>" \
  --report_to "none" \
  --enable_xformers_memory_efficient_attention
```

Smoke intent:

- verifies the target-specific unpaired dataset contract
- verifies that `sd-turbo + LoRA + discriminator + LPIPS + clean-fid` all import and start
- keeps batch at `1` and validation small enough for a local `4070`

What this smoke does **not** close:

- 5-target adapter sweep
- Distinct5 `5x5 / 750` merged evaluator packet
- any paper-safe LoRA anchor claim
