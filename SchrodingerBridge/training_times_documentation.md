# Training Times Documentation — All Baselines

## Measurement Protocol
All timings measured on: **NVIDIA RTX 4070 Laptop GPU (8GB VRAM)**
Training = end-to-end wall clock. All measurements for 5-domain multi-style model (photo, Hayao, monet, vangogh, cezanne).

---

## Ours (Latent Bridge Matching / LBM)

| Item | Value | Source |
|------|-------|--------|
| Training epochs | 7 (primary), 8 total | `config.json` |
| **Total train time (7 ep)** | **309.9 s** (~5.2 min) | `timing_filled.json` |
| Avg epoch train | 44.3 s/epoch | `timing_filled.json` |
| 1-epoch train | 52.5 s | `timing_metrics_combined.json` |
| Batch size | 108 (latent) | `config.json` |
| Samples/sec (epoch 7) | 239.9 | `timing_filled.json` |
| **Inference (750 img)** | **85.4 s** (0.114 s/img) | `timing_filled.json` |
| Parameters | 3.91 M | `paper_aaai2026.tex` Table 3 |
| Peak VRAM (inference) | 33.4 MB | `paper_aaai2026.tex` Table 3 |
| Micro-benchmark throughput | 102.16 img/s | `review_baseline_suite_full4g` |

### Destructive Ablations (all 7-epoch, comparable timing)

| Ablation | Train sec | Source |
|----------|-----------|--------|
| D0 full control | 290.65 | `summary_all_tested_metrics_with_ablations.csv` |
| D1 no terminal SWD | 295.94 | same |
| D2 no kinetic | 303.31 | same |
| D3 no SWD + no kinetic | 306.55 | same |
| D4 conv body | 295.50 | same |
| D5 no skip routing | 294.61 | same |
| D6 no spatial prior | 305.78 | same |
| D7 no residual path | 304.25 | same |
| D8 strong color loss | 308.59 | same |
| D9 L2 cost | 311.10 | same |
| D10 micro HF SWD | 302.12 | same |
| D11 single terminal step | 298.57 | same |
| **Range** | **290–311 s** | consistent across all 12 |

---

## SaMST (SaMST strict, used in complete_750)

| Item | Value | Source |
|------|-------|--------|
| Model | epoch_20.model (from external/SaMST) | `SaMST-main/checkpoint/` |
| **Training type** | **1 training covers all 5 styles** | `run_511/docs/timing_filled.json` |
| 1-epoch probe (all 5 styles) | 67.687 s | `timing_summary.csv` (samst_timing_probe) |
| Batch size | 1 | probe config |
| Training samples/style | 16 (probe), full data for main | same |
| **Target epochs** | **100** (paper default) | 👆 user confirmed |
| **Estimated total train (100 ep)** | **6,768.7 s** (~1.9 h) | 67.687 × 100 |
| Alternative estimate (30 ep) | 2,030.6 s | `timing_filled.json` |
| **Inference (750 img)** | **39.8 s** (0.053 s/img) | `timing_summary.csv` |
| Parameters | ~6 M | SaMST paper |

**Note:** Training time for the strict model was NOT preserved in original `summary.json`. The 1-epoch probe was measured specifically to fill this gap.

---

## S2WAT (S2WAT strict, used in complete_750)

| Item | Value | Source |
|------|-------|--------|
| Model | checkpoint from `bs1_safe` training | `repos/S2WAT-main/pre_trained_models/` |
| **Training type** | **1 training covers all 5 styles** (arbitrary style transfer) | confirmed via `run_generate_5x5.py` |
| 1-epoch (1 iteration) time | **~5.3 s** | Measured 2026-05-14, batch_size=1, bf16, grad_checkpoint |
| Batch size | 1 (safe mode) | `run_train_chunk_resume.bat` |
| **Total iterations** | **2,000** (e2000 naming convention) | `runs/s2wat_bs1_safe_e2000_5x5` |
| **Estimated total train** | **~10,600 s** (~2.9 h) | 5.3 × 2,000 |
| Inference (750 img) | Not separately measured | — |
| Parameters | ~7 M | S2WAT paper |
| **Per-iteration loss (1 epoch)** | 71.40 (start) / 58.82 (batch=2) | Measured |

**Note:** S2WAT's "epoch" is defined as 1 iteration (1 batch) in the codebase. Total 2,000 iterations was inferred from the `e2000` run directory naming. Training was done in chunks of 200 epochs per round (`run_train_chunk_resume.bat` default).

---

## Other Baselines (from run_511/outputs/review_baseline_suite_full4g)

| Method | Train sec | Infer sec (750 img) | Parameters | FLOPs | Peak VRAM |
|--------|-----------|---------------------|------------|-------|-----------|
| StyTr2 | 143.46 | 567.37 (0.76 s/img) | 48.34 M | 603.15 G | 408.7 MB |
| CAST | 1,759.80 | 75.47 (0.10 s/img) | 7.01 M | 94.90 G | 145.6 MB |
| AesFA | 6,607.60 | 40.26 (0.054 s/img) | 3.22 M | 25.29 G | 89.0 MB |
| AesPA-Net | 366.30 | 345.28 (0.46 s/img) | 24.20 M | 246.11 G | 575.0 MB |
| **StyleID** | **Training-free** | 603.32 (0.80 s/img, 750 est) | — | — | — |
| **AdaIN v32k** | 9,220.39 (32k iters) | 9.28 (0.012 s/img) | — | — | — |
| **AdaIN vgg19** | 262.78 (2k iters) | 9.10 (0.012 s/img) | — | — | — |

Sources: `review_baseline_suite_full4g/summary.csv`, `timing_summary.csv`

---

## Timing Comparison Summary

| Method | Train Time | Inference (750 img) | Train Speedup vs Ours |
|--------|-----------|---------------------|----------------------|
| **Ours (7ep)** | **310 s** | **85.4 s** | 1.0× (baseline) |
| SaMST (100ep) | 6,769 s (est.) | 39.8 s | **0.046×** (21.8× slower) |
| S2WAT (2000it) | 10,600 s (est.) | — | **0.029×** (34.2× slower) |
| StyTr2 | 143 s | 567.4 s | 2.2× faster train |
| CAST | 1,760 s | 75.5 s | 0.18× |
| AesFA | 6,608 s | 40.3 s | 0.047× |
| AesPA-Net | 366 s | 345.3 s | 0.85× |
| AdaIN v32k | 9,220 s | 9.3 s | 0.034× |
| AdaIN vgg19 | 263 s | 9.1 s | 1.18× |
| StyleID | 0 (training-free) | 603 s (est.) | ∞ |

**Key takeaway:** Ours is 21.8× faster to train than SaMST and 34.2× faster than S2WAT, while achieving comparable or better metrics.
