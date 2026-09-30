# 620 Phase 3-4 Exploration Plan (3 Days)

## Current Status
- **Best result**: swd16, vlen=0.04, e5 → clip=0.7051, lpips=0.2935
- **Gap to target**: 0.7051 → 0.72 (need +0.015)

## Day 1: DINO Conditioning Upgrade (Phase 3)

### Hypothesis
Current DINO conditioning uses representative samples per style. Per-instance DINO may provide better style guidance.

### Experiments
1. **Per-instance DINO** (Priority: HIGH)
   - Modify: `style_condition_source = "target_dino_patches_per_instance"`
   - Config: swd16, vlen=0.04
   - Expected: Better style specificity → higher clip_style

2. **DINO bank size scan**
   - Current: topk=8
   - Test: topk={4, 12, 16}
   - Find optimal bank size

3. **DINO layer selection**
   - Current: layer=-1 (last)
   - Test: layer={-2, -3} (earlier layers)
   - Earlier layers may capture more style info

## Day 2: Architecture Upgrade (Phase 4)

### 4.1 MoE Cross-Attention
- **Hypothesis**: Mixture of Experts can specialize on different style aspects
- **Config**: `num_experts={4, 8}`, `topk_experts={1, 2}`
- **Base**: swd16, vlen=0.04

### 4.2 Multi-scale SWD
- **Hypothesis**: SWD at multiple scales captures both fine and coarse style
- **Config**: `swd_scales={[64, 32, 16]}`
- **Weight distribution**: coarse-to-fine weighting

### 4.3 Style Gate Tuning
- Current: `style_cross_attn_gate_init=0.05`, observed gate=0.12
- Test: `gate_init={0.1, 0.2, 0.3}`
- Hypothesis: Higher gate init → stronger style influence

## Day 3: Integration & Optimization

### 5.1 Best Combination
- Combine top findings from Day 1-2
- Example: per-instance DINO + MoE + multi-scale SWD

### 5.2 Hyperparameter Fine-tuning
- Learning rate scan: {1e-4, 2e-4, 5e-4}
- Batch size scan: {32, 64, 128}
- NFE scan: {4, 8, 16}

### 5.3 Final Validation
- Run best config with formal training (8 epochs)
- Full evaluation on all test sets
- Document final results

## Execution Protocol

### For Each Experiment:
1. **Quick test**: 1 epoch smoke test
2. **Convergence check**: If e1 optimal → refine vlen
3. **Formal run**: 10 epochs with optimal vlen
4. **Analysis**: Record clip_style, lpips, delta_idt
5. **Decision**: Keep if improvement > 0.005

### Time Budget per Experiment:
- Smoke test: 5 min
- Convergence refinement: 15 min
- Formal run: 30 min
- Total per experiment: ~1 hour

### Total Capacity:
- Day 1: 8 hours → 8 experiments
- Day 2: 8 hours → 8 experiments
- Day 3: 8 hours → 8 experiments
- **Total**: 24 experiments

## Success Criteria
- clip_style ≥ 0.72 (primary)
- content_lpips ≤ 0.30 (secondary)
- Pareto improvement over current best

## Fallback Plan
If Phase 3-4 fail to reach 0.72:
- Analyze failure modes
- Consider: data augmentation, longer training, ensemble methods
- Document insights for future work
