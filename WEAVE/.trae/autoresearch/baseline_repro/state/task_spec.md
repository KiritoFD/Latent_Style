# Task: Baseline Reproduction & Unified Evaluation (AutoResearch)

## Goal
Complete reproduction and unified evaluation of all baseline methods for AAAI 2027 paper on distinct5_512 dataset, recording training/inference times, forming supporting conclusions.

## Milestones
1. CUT training (5 styles × 2+2 epochs, 250 imgs/style subsample) — IN PROGRESS on remote 3060
2. SaMam training+inference in WSL (mamba-ssm env, 5000 iters, batch=4, size=256)
3. Unified evaluation: LPIPS-VGG + CLIP-ViT-L/14, clip_style_idt_baseline=0.6399
4. Aggregated training/inference time table + supporting conclusions
5. AMiner query for additional baselines requiring reproduction

## Success Criteria
- All non-structurally-failed methods produce 750 images in standard naming format
- Unified evaluation JSON contains all completed methods with clip_style + content_lpips
- Training/inference time recorded for each method (excluding inference-only: Identity)
- Supporting conclusions document with ranking table

## Hard Constraints (from project memory)
- Local GPU only (no remote GPU for training)
- WSL mamba-ssm env for SaMAM only
- GPU 12GB VRAM ceiling (4070 Laptop 8GB + remote 3060 12GB)
- Sequential GPU tasks (no parallel GPU jobs)
- 30s timeout on all commands

## Methods Status (as of 2026-06-30 20:50)
| Method | Train | Infer | Eval | Notes |
|--------|-------|-------|------|-------|
| identity | N/A | 750 ✅ | 0.6933/0.0 | Source copy |
| adain | N/A | 750 ✅ | 0.6679/0.7425 | Pretrained decoder |
| sdedit_str0.10 | N/A | 750 ✅ | 0.7188/0.3183 | |
| sdedit_str0.20 | N/A | 750 ✅ | 0.7340/0.3492 | |
| sdedit_str0.35 | N/A | 750 ✅ | 0.7797/0.4508 | |
| sdedit_str0.40 | N/A | 750 ✅ | 0.7934/0.4826 | |
| sdturbo | N/A | 750 ✅ | 0.6933/0.0033 | Near-identity |
| styleid | N/A | 750 ✅ | 0.8223/0.5523 | |
| samst | 39.5 min | ~10 min | PENDING | 2 epochs |
| s2wat | FAILED | — | — | OOM 13.6GB at 128px |
| cut | IN PROGRESS | PENDING | PENDING | 5 styles, 2+2 epochs |
| samam | PENDING (WSL) | PENDING | PENDING | 5000 iters |

## Iteration Cap
- Per-work session: 15 rounds or 30 minutes
- Stall threshold: 2h no progress update
- Pivot threshold: stale_count >= 2
