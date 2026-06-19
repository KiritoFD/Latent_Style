# 620 Experiment Progress Summary

## Current Best Result
**Configuration**: swd16, virtual_length=0.04, epoch=5
**Metrics**: clip_style=0.7051, content_lpips=0.2935
**Improvement**: +0.065 over IDT (0.6399), +0.090 over SaMAM (0.6146)

## Experiments Completed

### Phase 1: SWD Weight Scan
| SWD | vlen | Best Epoch | clip_style | lpips | Status |
|-----|------|------------|-----------|-------|--------|
| 12 | 1.0 | e8 | 0.6725 | 0.2968 | ✅ Baseline |
| 16 | 1.0 | e1 | 0.7053 | 0.2901 | ✅ Found e1 optimal |
| 16 | 0.2 | e9 | 0.7038 | 0.3064 | ✅ Refined |
| **16** | **0.04** | **e5** | **0.7051** | **0.2935** | ✅ **Best** |
| 20 | 0.04 | e1 | 0.7006 | 0.2750 | ⚠️ Incomplete (OOM) |

### Phase 2: Convergence Diagnosis Protocol
**Key Finding**: When e1 is optimal with vlen=1.0, refine to vlen=0.04 to find true peak.

**Protocol**:
1. Run smoke test (vlen=1.0, 1 epoch)
2. If e1 optimal → refine to vlen=0.2, run 10 epochs
3. If still e1 optimal → refine to vlen=0.04, run 10 epochs
4. Identify peak epoch from fine-grained curve

### Phase 3: Hyperparameter Tuning (In Progress)
| Experiment | Config | Status | Notes |
|-----------|--------|--------|-------|
| lr=1e-4 | swd16, vlen=0.04 | 🔄 Running | Testing slower learning rate |

## Remaining Gap
**Target**: clip_style ≥ 0.72
**Current**: 0.7051
**Gap**: +0.015 needed

## Next Experiments (Priority Order)

### High Priority
1. **Learning rate scan**: {5e-5, 1e-4, 5e-4}
2. **Batch size scan**: {32, 128}
3. **NFE scan**: {4, 16}

### Medium Priority
4. **Style gate init**: {0.1, 0.15, 0.2}
5. **DINO bank topk**: {4, 12, 16} (requires cache rebuild)

### Low Priority (Architecture Changes)
6. **MoE cross-attention**: num_experts={4, 8}
7. **Multi-scale SWD**: scales=[64, 32, 16]

## Time Budget
- Each experiment: ~30 min (smoke + convergence + formal)
- Remaining today: 4 hours → 8 experiments
- Day 2-3: 16 hours → 32 experiments total capacity

## Success Criteria
- Primary: clip_style ≥ 0.72
- Secondary: content_lpips ≤ 0.30
- Tertiary: Pareto improvement over current best

## Lessons Learned
1. **Convergence diagnosis critical**: vlen=1.0 hides true peak
2. **SWD=16 optimal**: Higher weights don't help
3. **Early epochs powerful**: e5 often better than e8-e10
4. **LPIPS trade-off**: Higher clip_style → higher lpips
