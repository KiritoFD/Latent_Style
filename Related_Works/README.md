# Related Works

Baseline methods and evaluation infrastructure for style transfer comparison.

## Directory Structure

```
Related_Works/
  repos/                  # Baseline method repositories
    AdaIN-style-official/ # AdaIN (Huang & Belongie 2017)
    AesFA/                # AesFA (Kim et al. 2024)
    AesPA-Net/            # AesPA-Net (Kim et al. 2023)
    ArtBank/              # ArtBank (Liu et al. 2024)
    blora/                # BLORA baseline
    cut/                  # CUT (Park et al. 2020)
    Cycle-NCE/            # CycleNCE variant
    cyclegan_turbo/       # CycleGAN-Turbo
    Dreambooth-Stable-Diffusion-main/
    external/             # External eval assets
    pytorch-CycleGAN-and-pix2pix/
    S2WAT-main/           # S2WAT (Mao et al. 2023)
    SaMST-main/           # SaMST (Deng et al. 2024)
    style_aligned/        # StyleAligned
    styleid/              # StyleID (Hertz et al. 2024)
    StyTR-2/              # StyTR-2 (Deng et al. 2022)
    s2wat/                # S2WAT (alternate)
  run_511/                # Protocol-750 evaluation runner & results
    complete_750/         # Aggregated eval results (AdaIN/SaMST/StyleID/Ours)
    outputs/              # Inference outputs
    repos/                # Working copies of repos used by run_511
    eval_*.py             # Evaluation scripts
    run_*.py / run_*.bat  # Experiment launchers
    timing_metrics_combined.json  # Ours vs SaMST full comparison
  baseline_pipeline/      # Older baseline pipeline infrastructure
  runs/                   # Experiment run history (tracked)
  summary/                # Aggregated experiment summaries (tracked)
  scripts/                # Utility scripts (export, plot, convert, etc.)
  results/                # CSV/JSON result files and scatter plots
  docs/                   # Documentation (README_TOOLS.md, jobs.md)
```

## Completed Evaluations (Protocol-750)

See `run_511/complete_750/` for full eval suites. Key results in `Plan_Docs/RESULTS_SUMMARY.md`.

| Method | LPIPS↓ | CLIP-style↑ | CLIP-content↑ | SSIM-Y↑ | Status |
|--------|--------|-------------|---------------|---------|--------|
| Ours 7ep | 0.451 | 0.716 | 0.809 | 0.455 | complete |
| SaMST 100ep | 0.466 | 0.719 | 0.819 | 0.652 | complete |
| StyleID | 0.750 | 0.760 | 0.552 | 0.147 | complete |
| AdaIN v32k | 0.630 | 0.713 | 0.699 | 0.325 | complete |
| AdaIN vgg19 | 0.687 | 0.693 | 0.599 | 0.290 | complete |
