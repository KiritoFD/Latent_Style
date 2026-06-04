# img2img-turbo Distinct5 Preflight

Date: 2026-06-05

Scope: local preflight for the `img2img-turbo` / CycleGAN-Turbo comparison arm on
Distinct5-512. This note records dataset readiness and environment readiness
before the first GPU smoke run.

## 1. Dataset Root

Materialized target-specific unpaired datasets:

- `F:\wikiart_distinct5_img2img_turbo_datasets`

Builder:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\prepare_distinct5_img2img_turbo_datasets.py`

Source roots actually used:

- train:
  - `F:\wikiart_distinct5_512_images\train`
- test:
  - `F:\wikiart_distinct5_512_images\test`

Reason:

- Windows Python cannot reliably read the `0-byte + ReparsePoint` test images
  under `F:\wikiart_distinct5_samam_512_classview\test`.
- The builder now resolves to the real image roots and records the resolved
  paths in each dataset manifest.

## 2. Closed Dataset Counts

All five Distinct5 targets were materialized successfully:

| target | train_A | train_B | test_A | test_B |
| --- | ---: | ---: | ---: | ---: |
| `Early_Renaissance` | 4000 | 1000 | 150 | 30 |
| `Impressionism` | 4000 | 1000 | 150 | 30 |
| `Minimalism` | 4000 | 1000 | 150 | 30 |
| `Rococo` | 4000 | 1000 | 150 | 30 |
| `Ukiyo_e` | 4000 | 1000 | 150 | 30 |

Each target folder contains:

- `train_A/`
- `train_B/`
- `test_A/`
- `test_B/`
- `fixed_prompt_a.txt`
- `fixed_prompt_b.txt`
- `dataset_manifest.json`

Global manifest:

- `F:\wikiart_distinct5_img2img_turbo_datasets\manifest.json`

## 3. Training-Entry Preflight

Repo root:

- `G:\GitHub\Latent_Style\Related_Works\repos\cyclegan_turbo\img2img-turbo`

Verified on the current local Python environment:

```powershell
C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe `
  G:\GitHub\Latent_Style\Related_Works\repos\cyclegan_turbo\img2img-turbo\src\train_cyclegan_turbo.py --help
```

Result:

- the training entry imported successfully and printed the full CLI help
- currently installed local modules were sufficient for import-time preflight:
  - `torch`
  - `diffusers`
  - `transformers`
  - `peft`
  - `lpips`
  - `clean-fid`
  - `vision_aided_loss`
  - `wandb`
  - `accelerate`
- `xformers` is not installed in the current local Python environment, so the
  first smoke should run without `--enable_xformers_memory_efficient_attention`
  unless a dedicated env is prepared first

## 4. Smoke Launcher

Reusable launcher:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\run_img2img_turbo_distinct5_smoke.py`

Dry-run evidence:

- `G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\img2img_turbo_smoke_runs\Early_Renaissance\launch_manifest.json`
- `G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\img2img_turbo_smoke_runs\Early_Renaissance\launch_command.txt`

The launcher currently:

- points to the target-specific dataset root
- writes a launch manifest before execution
- keeps `batch_size=1`
- keeps `NCCL_P2P_DISABLE=1`
- makes `xformers` optional

## 5. Current Blocker

No GPU smoke was launched in this preflight turn because the local `RTX 4070
Laptop GPU` is still occupied by the Distinct5-512 `SDXL-fix` run:

- observed local state during this preflight:
  - `7643 MiB / 8188 MiB`
  - `100% util`

The next aligned action after the local SDXL run releases the GPU is:

1. launch a single-target `img2img-turbo` smoke at `batch=1`
2. capture memory usage, startup success, and first checkpoint/validation output
3. decide whether to fan out to the full 5-target comparison
