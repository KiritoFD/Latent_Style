# Project Status & Plan

Updated: 2026-05-12

## Paper Scope (AAI)

- **Table 1** (quality): AdaIN / StyTR-2 / AesPA-Net / AesFA / CAST / StyleID / SaMST / Ours
- **Table 2** (efficiency): SaMST / CAST / StyleID / Ours
- **Figure 1** (time-to-quality): CycleGAN / FastCUT / SaMST / Ours
- **Table 3** (ablation): six key SB variants
- **Table 4** (user study): Ours vs CAST / StyleID / SaMST / StyTR-2

Do not add extra baselines to the main table. Put AdaAttN / EFDM / ArtBank / InST / DiffuseIT / DiffStyle in supplement.

## Experiment Status

### Ours (Schrodinger Bridge)

| Variant | Train | Infer (750) | Full Eval | Notes |
|---------|-------|-------------|-----------|-------|
| 1ep (S-add__K-1_C-0_W-20_Col-0) | done (52.5s) | done (85.4s) | summary.json only | No guard/antihf/artifact/kid |
| 7ep (same) | done (309.9s) | done (85.4s) | complete | All eval suites run |
| Grid search (3ep x 108 configs) | done | done | done | In SchrodingerBridge/grid_search_3epoch/ |
| Pareto probe (4 configs) | done | done | done | In SchrodingerBridge/pareto_probe_4/ |

### Baselines (in Related_Works/run_511/)

| Method | Train | Infer (750) | Full Eval | Status |
|--------|-------|-------------|-----------|--------|
| AdaIN v32k | done (9220.4s, 32k iter) | done (9.3s) | complete | Best AdaIN run |
| AdaIN vgg19 | done (262.8s, 2k iter) | done (9.1s) | complete | Weaker than v32k |
| SaMST (100ep) | extrapolated 6768.7s | done (39.8s) | complete | epoch_100 for 4 styles, epoch_30 for photo |
| StyleID | training-free | done (603.3s) | complete | Weak content preservation |
| StyTR-2 | smoke only | smoke only | none | Needs re-run with realistic profile |
| AesFA | timing probe only | timing probe only | none | Pending |
| AesPA-Net | timing probe only | timing probe only | none | Pending |
| CAST | smoke failed | none | none | Needs weight/script fix |

### Pending Work

1. Run AesFA full 750 inference + eval
2. Run AesPA-Net full 750 inference + eval
3. Fix CAST and run 750 inference + eval
4. Re-run StyTR-2 with realistic profile or official weights
5. Run guard/antihf/artifact/kid eval for Ours 1ep
6. DINO/CFSD metrics for structure sensitivity
7. User study

## Directory Structure (after reorganization)

```
Latent_Style/
  SchrodingerBridge/          # Our method: code, configs, experiments, full_eval
  Related_Works/
    run_511/                  # Baseline evaluation runner (moved from root)
      repos/                  # Cloned baseline repos (AesFA, AesPA-Net, SaMST, StyTR-2, adain, cast)
      outputs/                # Inference outputs for all methods
      complete_750/           # Aggregated 750-image eval results
      scripts/                # Helper scripts
      eval_*.py               # Evaluation scripts
      run_*.py / run_*.bat    # Experiment runners
    baseline_pipeline/        # Older baseline pipeline infrastructure
    AdaIN-style-official/     # Individual repo checkouts
    AesFA/
    AesPA-Net/
    SaMST-main/
    StyTR-2/
    ...
  Plan_Docs/                  # Project documentation & status
    PROJECT_STATUS.md         # This file
    RESULTS_SUMMARY.md        # All quantitative results
  experiments/                # Archived experiment configs
  style_data/                 # Style image datasets
  clip-feats-vitb32/          # CLIP feature cache
```

## Evaluation Protocol

- **Protocol-750**: 5 source styles x 5 target styles x 30 images = 750 images
- **Reference manifest**: SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images
- **Metrics family**:
  - Main: LPIPS↓, CLIP-style↑, CLIP-content↑
  - Guard: SSIM-Y↑, Edge-F1↑, Blockiness↓, HF-gen↓
  - AntiHF: hf_z↓, hf_artifact_index↓
  - Artifact: MUSIQ↑, MANIQA↑, DISTS↓, FFT-KL↓, FFT-slope↓
  - KID: KID↓, HF-Patch-KID↓
