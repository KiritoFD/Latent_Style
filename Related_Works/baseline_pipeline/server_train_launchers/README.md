# Server Training Launchers

Copy this directory together with the repo to a server, then launch training from
the repository root.

The scripts use one shared resource profile:

```bash
export VRAM_PROFILE=4g   # conservative laptop / small GPU
export VRAM_PROFILE=7g   # default single 8GB-class GPU
export VRAM_PROFILE=11g  # 12GB-class GPU
```

Profiles map to conservative settings:

| Profile | Batch | Train resize/crop | Images/style | AesFA iters | StyTR2 iters | AesPA iters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4g` | 1 | `128/128` | 16 | 200 | 200 | 200 |
| `7g` | 1 | `192/192` | 32 | 500 | 500 | 500 |
| `11g` | 2 | `256/256` | 64 | 1000 | 1000 | 1000 |

## Entrypoints

```bash
bash Related_Works/baseline_pipeline/server_train_launchers/train_aesfa.sh
bash Related_Works/baseline_pipeline/server_train_launchers/train_stytr2.sh
bash Related_Works/baseline_pipeline/server_train_launchers/train_aespa.sh
bash Related_Works/baseline_pipeline/server_train_launchers/train_artbank_preflight.sh
```

Serial all-in-one:

```bash
bash Related_Works/baseline_pipeline/server_train_launchers/train_new_baselines_serial.sh
```

Outputs go to:

```text
Related_Works/runs/server_new_baselines/<profile>/
```

Notes:

- `AesFA` requires `Related_Works/AesFA/vgg_normalised.pth`.
- `StyTR-2` requires `Related_Works/StyTR-2/experiments/vgg_normalised.pth`; if missing, the launcher copies the AesFA VGG file when available.
- `AesPA-Net` requires `Related_Works/AesPA-Net/baseline_checkpoints/vgg_normalised_conv5_1.t7`.
- `ArtBank` is heavyweight. The provided script only checks required assets and does not start diffusion training.
