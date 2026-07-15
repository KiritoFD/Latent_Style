# Task: Semantic SWD Deep Exploration

## Goal
Deeply explore the semantic region SWD mechanism, understand why it works, find its limits, and write theoretical documentation同步进行.

## Background
- Semantic region SWD: k-means on content latent partitions into K regions, each matched independently via deterministic quantile interpolation.
- Current best: K=8, β=0.7, SWD w=12 → MUSIQ=51.86 (base), 54.50 (with EOTA τ=0.08).
- Key finding: deterministic F.interpolate-based quantile matching is ESSENTIAL. Stochastic multinomial sampling regresses MUSIQ by ~10 points.
- Governing pattern: adding HF energy hurts MUSIQ; redistributing/cleaning HELPS.

## Milestones
1. M1: K-value sweep (K=4, 16, 32 vs baseline 8) — understand region granularity tradeoff.
2. M2: β-value fine sweep (0.5, 0.6, 0.8, 0.9 vs 0.7) — find optimal global/region blend.
3. M3: Theoretical analysis of deterministic vs stochastic matching — why 10-point gap.
4. M4: Region matching alternatives (softmax weighting, hierarchical) — explore mechanisms.
5. M5: Cross-dataset generalization check — does D5 optimal transfer to R5?
6. M6: Write theory document (docs/SWD/semantic_swd_theory.md) with formal analysis.

## Success Criteria
- At least 4 new experimental configurations trained to convergence (5 epochs, batch=24).
- Theoretical document with formal analysis of why region-wise matching is tighter bound.
- Identify whether MUSIQ can be pushed beyond 54.50 via parameter adjustment.
- All findings logged to findings.jsonl with quantitative metrics.

## Constraints
- Local GPU RTX 3060 12GB, VRAM 9-11G training, ≤7G eval.
- D5 dataset: F:/wikiart_distinct5_samam_512_classview (latents at F:/wikiart_distinct5_samam_512_latents_ema).
- Configs in configs/semantic_swd_musiq/, base = swd_cm_sem_r8.json.
- Each experiment: 5 epochs, batch_size=24, Patience=2.
