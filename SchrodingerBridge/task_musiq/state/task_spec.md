# Task: Push MUSIQ and Update Main Table

## Goal
Push WEAVE's MUSIQ higher via model architecture improvements, then run metrics on three datasets (Distinct5-512, Photo2Art-256, Random5-WikiArt) and fill back into the main table. Copy aaai2027_v4 to v5 and update narrative.

## Current Baseline (aaai2027_v4 main table, WEAVE row)
- D5-512:  CLIP-S=0.7213, LPIPS=0.2868, MUSIQ=35.31
- P256:    CLIP-S=0.6826, LPIPS=0.2031, MUSIQ=45.68
- R5:      CLIP-S=0.7434, LPIPS=0.2904, MUSIQ=31.72

## Trade-off Reference (Seedream 4.5)
- D5-512:  CLIP-S=0.7198, LPIPS=0.4767, MUSIQ=69.51
- P256:    CLIP-S=0.7515, LPIPS=0.2270, MUSIQ=64.00
- R5:      -- (cost prohibitive)
Trade-off orientation: accept higher LPIPS (up to ~0.48) to push MUSIQ toward 60+.

## Latest Experiment (dwt_route + attn-SWD, distinct5 only)
- CLIP-S=0.7275, LPIPS=0.4347, MUSIQ=41.11 (+5.8 vs baseline)

## Milestones
1. Analyze current results, plan >=4 architecture directions
2. Implement and validate on distinct5 (fast iteration, batch=48, 10ep)
3. Pick best config, run on all 3 datasets
4. Copy v4->v5, update main table + narrative

## Constraints
- Local GPU only (RTX 3060 12GB), VRAM 9-11G train, <=7G eval
- batch_size=24-48 for 12GB safety
- Training: Patience=2, max=10, at least 5 epochs to convergence
- Eval: batch_size=2, full_eval_batch_size=2, ref_feature_batch_size=2
- Architecture improvements first, parameter tuning last
