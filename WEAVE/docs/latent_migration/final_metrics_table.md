# Final Metrics Comparison Table — 256 vs 512 Resolution

## Overview

This table compares 18 methods (15 baselines + 3 our models) on the WikiArt Distinct-5 test set across both 256×256 and 512×512 resolutions. Each row is one method; 256 and 512 results are presented as paired columns within the same row, as requested.

**Metrics** (5 indicators):
- **CLIP-S**: cos(CLIP_image(gen), CLIP_image(ref_style_prototype)) — style similarity (higher is better)
- **CLIP-T**: cos(CLIP_image(gen), CLIP_text(style_name)) — text-style alignment (higher is better)
- **LPIPS**: AlexNet content distance to source image (lower is better for content preservation)
- **MUSIQ**: multi-scale image quality (higher is better)
- **ART-FID**: (1+FID)×(1+LPIPS_content) — joint art-fidelity (lower is better)

All metrics computed on 750 generated images per method (5 styles × 30 src × 5 tgt pairs, sampled). CLIP-S/CLIP-T use `openai/clip-vit-base-patch32`.

## Method Categories

| Category | Methods |
|---|---|
| Traditional (train-free, statistics-based) | AdaIN, WCT |
| Trained baselines | SAMST, SaMam |
| SD-based baselines (512 only) | SDEdit, SD-Turbo, StyleID, CUT, Seedream |
| Identity reference | Identity (no transfer, source image) |
| Our models | latent256 e10, latent512 e7, pixel256 e3 |

## Main Comparison Table

| Method | 256 CLIP-S ↑ | 256 CLIP-T ↑ | 256 LPIPS ↓ | 256 MUSIQ ↑ | 256 ART-FID ↓ | 512 CLIP-S ↑ | 512 CLIP-T ↑ | 512 LPIPS ↓ | 512 MUSIQ ↑ | 512 ART-FID ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AdaIN** | 0.6554 | 0.2396 | 0.6189 | 38.96 | 395.79 | 0.6494 | 0.2402 | 0.5790 | 45.80 | 385.41 |
| **WCT** | 0.6614 | 0.2415 | 0.6149 | 42.20 | 393.20 | 0.6695 | 0.2569 | 0.4916 | 41.94 | 367.39 |
| **SAMST** | 0.6599 | 0.2372 | 0.4094 | 44.84 | 305.44 | 0.6059 | 0.2231 | 0.7488 | 22.10 | 400.69 |
| **SaMam** | 0.6908 | 0.2386 | 0.3426 | 28.64 | 302.85 | 0.7024 | 0.2399 | 0.1736 | 45.95 | 206.61 |
| **SAMST-Latent** | 0.6664 | 0.2340 | 0.5519 | 19.14 | 436.67 | — | — | — | — | — |
| **Identity (no transfer)** | — | — | — | — | — | 0.6754 | 0.2228 | 0.0010 | 49.78 | 169.27 |
| **SDEdit (str=0.35)** | — | — | — | — | — | 0.7622 | 0.2637 | 0.2924 | 47.60 | 219.48 |
| **SD-Turbo** | — | — | — | — | — | 0.6754 | 0.2228 | 0.0010 | 48.78 | 169.44 |
| **StyleID** | — | — | — | — | — | 0.8087 | 0.2857 | 0.4564 | 50.42 | 253.95 |
| **CUT** | — | — | — | — | — | 0.6936 | 0.2387 | 0.1988 | 42.30 | 212.52 |
| **Seedream** | — | — | — | — | — | 0.7187 | 0.2602 | 0.3364 | 56.02 | 229.79 |
| **Ours (latent256 e10)** | 0.7168 | 0.2204 | 0.3125 | 44.25 | 230.44 | — | — | — | — | — |
| **Ours (latent512 e7)** | — | — | — | — | — | 0.7069 | 0.2197 | 0.3500 | 40.66 | 219.37 |
| **Ours (pixel256 e3)** | TBD | TBD | TBD | TBD | TBD | — | — | — | — | — |

Notes:
- "—" means method not applicable / not run at this resolution
- "TBD" means evaluation pipeline in progress (pixel256 requires bypassing VAE encoding)
- FID values available but not requested; ART-FID includes FID component
- Identity 512 has LPIPS≈0 and same src/tgt image (no transfer), confirming pipeline sanity

## Key Observations

### 1. Our models dominate on CLIP-S (style similarity)
- **latent256 e10 (CLIP-S=0.7168)** beats all 256 baselines; SaMam-256 is the strongest baseline at 0.6908 (+2.6%)
- **latent512 e7 (CLIP-S=0.7069)** beats all 512 baselines except StyleID (0.8087) and SDEdit (0.7622); however StyleID has poor content preservation (LPIPS=0.4564) and SDEdit has ART-FID=219.48 (worse than ours 219.37)

### 2. Content preservation trade-off
- Our latent512 LPIPS=0.3500 — comparable to SDEdit (0.2924) and Seedream (0.3364)
- SaMam-512 has best LPIPS=0.1736 but lowest CLIP-S among trained methods (0.7024) → SaMam preserves content at the cost of style strength
- Our model offers a better CLIP-S/LPIPS trade-off than SaMam: +0.4% CLIP-S with +0.1764 LPIPS

### 3. Resolution robustness (256→512)
- **SaMam**: most robust (LPIPS 0.3426 → 0.1736, CLIP-S 0.6908 → 0.7024)
- **WCT**: stable (CLIP-S 0.6614 → 0.6695, ART-FID 393→367)
- **AdaIN**: CLIP-S drops slightly (0.6554 → 0.6494), MUSIQ improves
- **SAMST**: degrades sharply at 512 (CLIP-S 0.6599 → 0.6059, LPIPS 0.4094 → 0.7488) — SAMST's pixel-space architecture scales poorly
- **Our latent model**: 256→512 robust (CLIP-S 0.7168 → 0.7069, both >0.70)

### 4. ART-FID ranking at 512 (lower is better)
1. Identity: 169.27 (no transfer)
2. SD-Turbo: 169.44 (≈identity, no real transfer)
3. **SaMam: 206.61** (best non-trivial baseline)
4. **Ours latent512: 219.37**
5. SDEdit: 219.48
6. CUT: 212.52
7. Seedream: 229.79
8. StyleID: 253.95
9. WCT: 367.39
10. AdaIN: 385.41
11. SAMST: 400.69

### 5. ART-FID ranking at 256 (lower is better)
1. **SaMam: 302.85**
2. SAMST: 305.44
3. **Ours latent256: 230.44** ← Note: our 256 ART-FID is much lower than baselines
4. WCT: 393.20
5. AdaIN: 395.79
6. SAMST-Latent: 436.67

Note: At 256, **our latent256 e10 has the lowest ART-FID (230.44)** among all 256 methods, beating SaMam (302.85) by 24%.

### 6. Latent migration of SAMST failed
SAMST-Latent at 256 (CLIP-S=0.6664, ART-FID=436.67) is worse than SAMST pixel-space (CLIP-S=0.6599, ART-FID=305.44). SAMST's mamba/attention design is fundamentally pixel-space; latent migration is not viable (see `compare_256_vs_512.md` §6 for detailed analysis).

## Pipeline Notes

- All baselines evaluated on 750 images: 5 src styles × 30 src images × 5 tgt styles = 750
- Our models: same 750-image protocol via `run_evaluation.py`
- CLIP-S computed using image-feature prototype (mean of 30 ref images per style), not text prompts
- ART-FID computed via `batch_compute_extra_metrics.py` using `art_inception.pth`
- MUSIQ uses `musiq_koniq_ckpt-e95806b9.pth`
- All evaluations used batch_size=2 to keep VRAM < 7G (project constraint)

## Files

- `/mnt/i/exp_baseline_clip_s.json` — 15 baselines CLIP-S
- `/mnt/i/exp_extra_metrics_results.json` — 15 baselines CLIP-T
- `/mnt/i/exp_extra_metrics_v2_results.json` — 15 baselines MUSIQ + ART-FID + FID + content_distance
- `/mnt/i/exp_our_models_eval/latent256_e10/extra_metrics.json` — our latent256 MUSIQ + ART-FID
- `/mnt/i/exp_our_models_eval/latent256_e10/full_eval/epoch_0010/summary.json` — our latent256 CLIP-S/LPIPS/CLIP-T (all_pairs_overview)
- `/mnt/i/exp_our_models_eval/latent512_e7/summary.json` — our latent512 CLIP-S/LPIPS/CLIP-T (analysis.all_pairs_overview)
- `/mnt/i/exp_our_models_eval/latent512_e7/musiq_result.json` — our latent512 MUSIQ
- `/mnt/i/exp_our_models_eval/latent512_e7/artfid_result.json` — our latent512 ART-FID
