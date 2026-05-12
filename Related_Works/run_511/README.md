# run_511

Self-contained style-transfer baseline train + inference packages.

Each baseline follows the same contract: train on `style_data/train`, infer
750 images (5 source x 5 target x 30) matching the SB reference manifest,
write `summary.json` + `summary.csv`.

## Baselines

| Baseline | Script | Repo Source | Notes |
|----------|--------|-------------|-------|
| **StyTR-2** | `run_stytr2_750.py` | `run_511/repos/StyTR-2` (self-contained) | Transformer-based |
| **AdaIN** | `run_adain_750.py` | `run_511/repos/adain` (self-contained) | Feedforward, VGG decoder only |
| **AesFA** | `run_aesfa_750.py` | `run_511/repos/AesFA` (self-contained) | Frequency-aware, needs VGG |
| **AesPA-Net** | `run_aespa_750.py` | `run_511/repos/AesPA-Net` (self-contained) | Contextual attention + GAN |
| **StyleID** | `run_styleid_750.py` | Diffusers (SD1.5) | Training-free, diffusion-based |
| **SaMST** | `run_samst_750.py` | `run_511/repos/SaMST-main` (self-contained) | Lightweight TransformerNet |
| **CAST** | `run_cast_750.py` | `run_511/repos/cast` (self-contained) | Contrastive learning |

## Smoke Test (all 4 baselines)

```bat
run_511\smoke_all_511.bat
```

Runs 1 training iteration + 1 content image per target style for each baseline.
Expected output per baseline:

```text
run_511/outputs/<baseline>_smoke/summary.json
run_511/outputs/<baseline>_smoke/infer_750/images/*.jpg
```

## Full 750 Run (single baseline)

```bat
set PROFILE=7g
set MODE=all
run_511\run_stytr2_750.bat
run_511\run_adain_750.bat
run_511\run_aesfa_750.bat
run_511\run_aespa_750.bat
```

## Serial Run (all 4 baselines, one after another)

```bat
set PROFILE=7g
set MODE=all
run_511\run_all_511.bat
```

This runs StyTR-2 -> AdaIN -> AesFA -> AesPA-Net serially. If one fails, the
script stops (unless it's `blocked`, e.g. missing checkpoint for AesPA-Net).

To run specific baselines only:

```bat
python run_511\run_all_511.py --baselines adain aesfa --mode all --profile 7g
```

## Inference Only (after training)

```bat
set MODE=infer
run_511\run_adain_750.bat
```

## GPU Profiles

| Profile | Batch Size | Train Images/Style | Max Iter |
|---------|-----------|-------------------|----------|
| `4g` | 1-4 | 16 | 200 |
| `7g` | 1-8 | 32 | 500 |
| `11g` | 2-16 | 64 | 1000 |

## Output Structure

```text
run_511/outputs/<baseline>_750/
  checkpoints/<baseline>/    # trained weights
  infer_750/images/          # 750 stylized images (5x5x30 naming)
  logs/                      # training + inference logs
  work/                      # temporary data prep
  summary.json               # run metadata
  summary.csv                # flat status table
```

## Requirements

- `style_data/train/` with `photo/`, `monet/`, `vangogh/`, `cezanne/`, `Hayao/`
- `style_data/overfit50/` for inference content images
- `vgg_normalised.pth` in `run_511/repos/AesFA/` or `run_511/repos/StyTR-2/experiments/`
- AesPA-Net VGG weights in `run_511/repos/AesPA-Net/baseline_checkpoints/`
- `SchrodingerBridge/exp/pareto_probe_4/.../images` for the 750-image filename manifest
