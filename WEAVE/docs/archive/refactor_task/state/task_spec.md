# Refactor Task: Codebase Subtraction (减法重构)

## Goal
通过逐个删除无效死代码和未消费参数，得到最简洁最干净的codebase，保持或超过baseline性能。

## Baseline (must match or exceed)
- Config: t1_asg_5ep (adaptive_style_gate=true, 5 epochs)
- CLIP-S:  0.7261
- LPIPS:   0.3354
- DINO-S:  0.4843
- DINO-C:  0.7692
- Checkpoint: exp/t1_asg_5ep/epoch_0005.pt

## Known Dead Components (from prior diagnosis)
1. `lowpass_mode` parameter — not consumed in 620_spectral_ode model
2. `spectral_ode_enabled` parameter — not consumed in 620_spectral_ode model
3. ASG (Adaptive Style Gate) — minimal impact (delta < 0.005)
4. SWD loss — only 4.3% of total loss, redundant with FM
5. Edge loss — minimal contribution
6. Endpoint AdaIN — main style injection channel (KEEP, but verify)
7. Wavelet is structurally hardcoded — parameters exist but unused

## Effective Components (KEEP)
1. Wavelet (Haar DWT) — structural, CLIP-S -0.016 when removed
2. Flow Matching — core mechanism, Flow-Only ≈ Full performance
3. Endpoint AdaIN — main style injection, largest CLIP-S drop when disabled

## Success Criteria
- Code line count reduced by >= 30%
- All 4 metrics within ±0.005 of baseline (or better)
- No dead parameters (every config field is consumed)
- Complete experiment report with scatter plot

## Constraints
- Remote RTX 3060 12GB for all training/eval
- Eval: batch_size=2, VRAM < 7GB
- Train: VRAM < 11.2GB, Patience=2, max=10
- Dataset: D5 (wikiart_distinct5_samam_512_classview)
- No external pretrained models (DINO/CLIP) in training

## Milestones
1. Audit complete: list of all dead code with evidence
2. Each deletion verified by eval (CLIP-S/LPIPS/DINO-S/DINO-C)
3. Param ablation complete (scatter data collected)
4. Final report + scatter plot
