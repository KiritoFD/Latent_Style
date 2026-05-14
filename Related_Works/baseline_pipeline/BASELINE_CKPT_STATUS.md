# Baseline Checkpoint And Training Status

Updated: 2026-05-11

This file separates baselines into four categories:

- already has local checkpoint/results, do not retrain for the current protocol
- has code but needs official checkpoint download before inference
- has code but needs local training to create a fair artist/domain checkpoint
- not ready because the repo/adapter is incomplete

Current engineering protocol:

- reference images: `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`
- size: `5 source styles x 5 target styles x 30 images = 750 outputs`
- styles: `photo / monet / vangogh / cezanne / Hayao`
- `ukiyoe` is intentionally not used

## Current Local Checkpoints

| Baseline | Local checkpoint status | Need training now? | Action |
| --- | --- | --- | --- |
| `S2WAT` | `baseline_pipeline/checkpoints/s2wat/{photo,monet,vangogh,cezanne,Hayao}/checkpoint_2000_epoch.pkl` exists | No | Use existing checkpoints for inference/eval |
| `SaMST` | `baseline_pipeline/checkpoints/samst/{monet,vangogh,cezanne,Hayao}/epoch_100.model` exists; `photo` missing; `ukiyoe` exists but excluded | No for current artist targets; only train `photo` if photo target is required | Use migrated/evaluated results for current table |
| `CUT` | No model checkpoint under `baseline_pipeline`; complete reusable output images already migrated | No | Do not retrain for current protocol; evaluate migrated outputs |
| `Ours` | Ours reference outputs exist in `SchrodingerBridge/exp/.../images` | No | Use as reference/current Ours row |

## New AAAI Baselines

| Baseline | Repo present? | Local official weights found? | Need training now? | Notes |
| --- | --- | --- | --- | --- |
| `StyleID` | Yes | Uses HF Stable Diffusion via diffusers cache, not a local method ckpt | No | Full 750-image inference completed and evaluated |
| `StyTR-2` | Yes | No `experiments/*.pth` weights found | No, prefer official weights | Needs four official Google Drive weights before fair inference |
| `AesFA` | Yes | Only `vgg_normalised.pth` exists; `ckpt/main/main.pth` missing | No, prefer official `main.pth` | Training from scratch is possible but expensive and not needed before trying official model |
| `AesPA-Net` | Yes | No VGG/decoder/transformer weights found | No, prefer official weights | Needs `vgg_normalised_conv5_1.t7`, `dec_model.pth`, `transformer_model.pth` |
| `ArtBank` | Yes | No `sd-v1-4.ckpt`, `embeddings.pt`, or `Mapper.pt` found | Not first; needs SD1.4 plus ArtBank prompt-bank weights or a dedicated artist training run | Training is diffusion/prompt-bank finetuning, heavier than S2WAT/SaMST |
| `CycleGAN` | Yes | No local trained artist-domain checkpoints found | Yes if we want a fair domain baseline and no reusable output exists | Use local `runs/cut_5x5/datasets/to_*` style datasets for training |
| `cyclegan_turbo` | Partial repo present | No checkpoints found | No until adapter/repo is validated | Treat as separate optional baseline |
| `AdaIN` | Clone is incomplete from interrupted download | No | No, prefer official pretrained decoder/vgg | Clean incomplete clone before retrying |

## What Actually Needs Training

For the current 750-image table:

- `S2WAT`: no training needed, checkpoint exists.
- `SaMST`: no training needed for current evaluated rows; `photo` checkpoint is missing but not needed if using migrated current outputs.
- `CUT`: no training needed because complete generated outputs were migrated.
- `StyleID`: no training needed; inference already done.

Training candidates only if we expand the table:

- `CycleGAN`: train locally per target domain if no reusable outputs/checkpoints are found.
- `ArtBank`: train only after confirming we cannot obtain official ArtBank prompt-bank weights; it also needs SD1.4 checkpoint.
- `AesFA / AesPA-Net / StyTR-2 / AdaIN`: do not train first; these should use official pretrained weights for fair paper baselines.

## Immediate Recommendation

1. Do not retrain `S2WAT`, `SaMST`, or `CUT` for the current protocol.
2. Fix or download official weights for `StyTR-2`, `AesFA`, `AesPA-Net`, and `AdaIN` before any inference/eval.
3. Treat `CycleGAN` as the first true local-training baseline to test, because it is an artist-domain baseline and local paired trainA/trainB folders already exist under `Related_Works/runs/cut_5x5/datasets/to_*`.
4. Treat `ArtBank` as heavyweight: first search/download official prompt-bank weights; train only if official weights are unavailable and time allows.

## Local Training Jobs

CycleGAN is now the active local-training target for the time-to-quality figure.

Script:

- `Related_Works/baseline_pipeline/scripts/train_cyclegan_targets.py`

Smoke test completed:

- command: `python Related_Works/baseline_pipeline/scripts/train_cyclegan_targets.py --targets monet --run_root Related_Works/runs/cyclegan_5x5_smoke --n_epochs 1 --n_epochs_decay 0 --max_dataset_size 8 --load_size 128 --crop_size 128 --save_epoch_freq 1 --print_freq 4`
- status: `ok`
- checkpoint dir: `Related_Works/runs/cyclegan_5x5_smoke/checkpoints/cyclegan_to_monet`
- timing CSV: `Related_Works/runs/cyclegan_5x5_smoke/train_timing.csv`

Planned background run:

- targets: `monet / vangogh / cezanne / Hayao`
- run root: `Related_Works/runs/cyclegan_5x5`
- purpose: first CycleGAN checkpoints for `time-to-quality` against `FastCUT / SaMST / Ours`
- note: this is serial by design, not four parallel GPU jobs

## Server-Copy Training Launchers

Unified launch directory:

- `Related_Works/baseline_pipeline/server_train_launchers/`

Resource profiles:

- `VRAM_PROFILE=4g`: batch `1`, resize/crop `128`, `16` images/style
- `VRAM_PROFILE=7g`: batch `1`, resize/crop `192`, `32` images/style
- `VRAM_PROFILE=11g`: batch `2`, resize/crop `256`, `64` images/style

Entrypoints:

- `train_aesfa.sh`: actual conservative AesFA training if `Related_Works/AesFA/vgg_normalised.pth` exists
- `train_stytr2.sh`: actual conservative StyTR-2 training; copies AesFA VGG into StyTR-2 experiments if needed
- `train_aespa.sh`: prepares data and starts AesPA-Net only if `baseline_checkpoints/vgg_normalised_conv5_1.t7` exists
- `train_artbank_preflight.sh`: checks ArtBank required assets only; does not start heavy diffusion training
- `train_new_baselines_serial.sh`: serial wrapper for `AesFA / StyTR-2 / AesPA-Net preflight / ArtBank preflight`

Example:

```bash
export VRAM_PROFILE=7g
bash Related_Works/baseline_pipeline/server_train_launchers/train_new_baselines_serial.sh
```

Outputs:

- `Related_Works/runs/server_new_baselines/<profile>/train_status.csv`
- `Related_Works/runs/server_new_baselines/<profile>/logs/`
- `Related_Works/runs/server_new_baselines/<profile>/checkpoints/`
