# Task: AAAI 2027 v2 Paper Rewrite

## Goal
Rewrite `aaai2027_v2/paper.tex` following the "Affordable Real Style Transfer" narrative:
- Title: "Affordable Real Style Transfer: Training Spectral Flow Matching on an RTX 3060 in Minutes"
- Core narrative: 揭露骗局 → 发现物理定律 → 降维打击
- Emphasis: §3 Method (math theory) + §4 Experiments
- Language: concise, clear, objective, no internal codenames

## Milestones
- M1: Survey complete (math theory + exp data + baseline + existing paper.tex)
- M2: Initialize task state
- M3: Rewrite paper.tex (title + abstract + intro + related + method + exp + conclusion)
- M4: Build PDF + verify
- M5: Git commit

## Success Criteria
- All numbers verified against train.log / summary.json / unified_results.json
- SaMam = 0.5816 / 0.2434 (NOT 0.7175 / 0.2423)
- Params = 903,248 (NOT 4.2M)
- Training time = ~3 min on RTX 3060 (NOT 310 sec on RTX 4070)
- Dataset = distinct5 (Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
- 6 SOTA ckpts preserved: T11, T10, 4F.1, 4I.2b, 4I.7b, 4J.1
- 12 baselines with unified protocol (HF CLIP ViT-B/32 + LPIPS Alex + 750 pairs)
- Narrative covers: Pareto Deadlock, Degeneration Attractor, Base Locking, Fiber Flow, EOTA, Stochastic DWT, 645 ablations

## Hard Constraints
- VRAM ≤ 7G during evaluation
- batch_size=2 for eval
- Dataset path: I:\wikiart_distinct5_samam_512_classview\test
- No fabricated data; all numbers from real logs

## Verification Data (verified 2026-07-03)
- T11 train.log: 2026-07-02 00:39:44 → 00:42:49 = 3 min 5 sec total
- T11 model params: 903,248 (from train.log line 33)
- T11 CLIP-S: 0.7212809419631957 (from summary.json analysis.all_pairs_overview.clip_style)
- T11 LPIPS: 0.2868271142013333 (from summary.json analysis.all_pairs_overview.content_lpips)
- Eval wall_total: 97.25 sec
- SaMam: 0.5816 / 0.2434 (from docs/baseline/README.md, verified v5)
