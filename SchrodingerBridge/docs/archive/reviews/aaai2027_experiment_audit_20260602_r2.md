# AAAI 2027 Experiment Audit

Date: 2026-06-02  
Round: `R20260602B`

Scope:

- `aaai_submission/paper_aaai2026.tex`
- `docs/experiments/aaai2027_master_experiment_log.csv`
- `docs/experiments/comparison_20260602/comparison_report.md`
- `docs/reviews/aaai2027_review_consensus_20260603.md`

## Directly supported claims

1. On historical strict-750, LBM is a strong content-preserving operating
   point, not the raw style maximum.
2. On Distinct5-512, the `idt` / no-op baseline is high enough that raw
   `CLIP-S` alone is misleading.
3. On Distinct5-512, the currently evaluated LBM-F point is the strongest
   content-preserving frontier point in the present evidence bundle.
4. Increasing tokenizer capacity by itself is not the main breakthrough path.
5. On historical strict-750, the current narrow claim that LBM is close to
   SaMST in style while preserving content slightly better is directly auditable
   from the current merged comparison table.

## Indirectly supported claims

1. The main bottleneck is style execution through the renderer rather than
   tokenizer size.
2. Endpoint-side `W1` / SA-SWD is an important correction.
3. LBM has a real efficiency advantage at the research-decision level.
4. The renderer-execution bottleneck story is plausible, but not yet isolated
   as a closed causal result.

## Claims that should not be written now

1. `Huber/L1` flow residual is already proven as the driver of the headline
   result.
2. LBM is universally or comprehensively better than all baselines.
3. Semantic SA-SWD axis superiority has already been decisively proven.
4. A strict normalized `22x` or universal speedup claim.
5. Non-audited bootstrap or subset statistics as if they were already external
   reviewer-ready artifacts.

## Next experiments and minimum acceptable protocols

### 1. Flow-loss metric ablation

- Dataset: `Distinct5-512`
- Hardware: same remote `3060`
- Base: one fixed LBM config; only switch the flow residual
- Variants: `MSE`, `Huber`, `L1`
- Budget: same seed, batch, and epoch budget
- Eval: strict-750 `full + transfer`
- Required outputs:
  - `CLIP-S`
  - `LPIPS`
  - `Δ_idt` on full and transfer
  - `ArtFID`
  - train wall time

### 2. SA-SWD axis ablation

- Dataset / hardware: same as above
- Base: one fixed strong baseline, preferably `F` or `H`
- Only change projection-axis selection
- Variants: `semantic-axis` vs `random-axis`
- Eval: strict-750 `full + transfer`
- Required outputs:
  - `CLIP-S`
  - `LPIPS`
  - `Δ_idt`
  - `ArtFID`

This reviewer considers the SA-SWD isolation experiment the most direct
novelty-closing test, even though the adversarial and scorecard lanes still put
the flow-loss metric ablation first for overall paper risk reduction.

### 3. Normalized time-to-parity curve

- Dataset: `Distinct5-512`
- Methods: `LBM`, `SaMAM`, `SaMST`
- Scope: same hardware if possible, otherwise explicitly split by hardware
- X-axis: wall-clock training time
- Y-axes:
  - `CLIP-S`
  - `LPIPS`
  - preferably `Δ_idt`
- Required outputs:
  - pre-registered stop criterion
  - one vector figure
- one source CSV
