# 620 Final Experiment Summary

## Best Results (Pareto Frontier)

### Configuration 1: swd16 (Best Style Transfer)
- **Config**: swd_weight=16, vlen=0.04, epoch=5
- **clip_style**: 0.7051
- **content_lpips**: 0.2935
- **delta_idt**: +0.065
- **delta_samam**: +0.090
- **Use case**: When style transfer quality is priority

### Configuration 2: swd20 (Best Pareto)
- **Config**: swd_weight=20, vlen=0.04, epoch=7
- **clip_style**: 0.7037
- **content_lpips**: 0.2674
- **Pareto score**: 0.4363
- **Use case**: When content preservation matters

## Comparison Table

| Method | clip_style | content_lpips | delta_idt | Notes |
|--------|-----------|--------------|-----------|-------|
| IDT baseline | 0.6399 | - | - | Reference |
| SaMAM step2250 | 0.6146 | 0.396 | -0.025 | Below IDT |
| **620 swd16 e5** | **0.7051** | 0.2935 | **+0.065** | Best style |
| **620 swd20 e7** | 0.7037 | **0.2674** | +0.064 | Best Pareto |

## Key Findings

1. **SWD weight sweet spot**: 16-20 (not higher)
2. **Convergence diagnosis essential**: vlen=0.04 reveals true peak
3. **Early epochs powerful**: e5-e7 often optimal
4. **Trade-off exists**: Higher clip_style ↔ Higher lpips

## Remaining Gap

- **Target**: clip_style ≥ 0.72
- **Current best**: 0.7051
- **Gap**: +0.015 needed

## Next Experiments

### Immediate (High ROI)
1. **Batch size scan**: {32, 128} - affects gradient noise
2. **NFE scan**: {4, 16} - affects inference quality
3. **Sigma scan**: {0.01, 0.03} - noise schedule

### Medium Priority
4. **Style gate init**: {0.1, 0.15} - style influence strength
5. **Learning rate**: {5e-5, 5e-4} - convergence speed

### Architecture (If needed)
6. **MoE cross-attention**: Specialized style handling
7. **Multi-scale SWD**: Coarse-to-fine style matching

## Time Investment

- Experiments completed: 12
- Time spent: ~4 hours
- Best improvement: +0.065 over IDT
- ROI: 0.0054 clip_style per hour

## Recommendations

1. **For production**: Use swd20 e7 (better content preservation)
2. **For research**: Continue hyperparameter scan
3. **For demo**: Use swd16 e5 (higher style transfer)
