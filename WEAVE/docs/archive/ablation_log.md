# Phase 2 A/B Test Log

## Target
- **clip_style > 0.72** (transfer pairs)
- **content_lpips < 0.45** (transfer pairs)
- Training: 8 epochs per experiment
- Eval: run_evaluation.py on checkpoint dir

## Baseline (Phase 1 cleaned code)
Config: skip_routing_mode=none, w_curvature=0.0, output_moment_match=false
- clip_style (transfer): 0.635
- clip_content (transfer): 0.900
- content_lpips (transfer): 0.304

## Experiment Order (by priority)

### Exp 1: skip_routing_mode = "normalized"
**Theory:** U-Net skip connections with style modulation may improve style transfer by injecting encoder features into decoder with style-aware routing. Current "none" mode discards skip info entirely.
**Hypothesis:** Enabling normalized skip routing should improve clip_style by allowing the decoder to access multi-scale encoder features modulated by style.
**Risk:** May hurt content preservation (LPIPS increase) if skip features fight with the flow matching objective.

### Exp 2: w_curvature = 1.0
**Theory:** Flow matching already tends toward straight trajectories, but explicit curvature regularization (penalizing velocity difference at t vs t+dt) may further straighten the flow field.
**Hypothesis:** Should improve 1-step inference quality without affecting multi-step results.
**Risk:** May add gradient noise and slow convergence.

### Exp 3: output_moment_match = true
**Theory:** AdaIN-style mean/std alignment of output to target style latent. Acts as a statistical safety net against color drift.
**Hypothesis:** May help clip_style by ensuring global color statistics match the target style.
**Risk:** May reduce local variance and make images look "flat" or "washed out".

## Results Table

| Exp | skip_routing | w_curv | moment_match | clip_style | clip_content | lpips | notes |
|-----|-------------|--------|--------------|------------|--------------|-------|-------|
| 0 (baseline) | none | 0.0 | false | 0.635 | 0.900 | 0.304 | Phase 1 cleaned |
| 1 | normalized | 0.0 | false | 0.635 | 0.900 | 0.303 | 与baseline几乎无差异，skip routing在当前框架下无贡献 |
| 2 | none | 1.0 | false | | | | |
| 3 | none | 0.0 | true | | | | |
