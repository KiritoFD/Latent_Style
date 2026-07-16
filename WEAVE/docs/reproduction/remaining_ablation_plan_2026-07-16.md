# Remaining Ablation Plan

Date: 2026-07-16

## Current evidence is sufficient for the main method

The main table should use only the six controls evaluated with the current 1.04M model and
the canonical 750-pair D5 protocol: full WEAVE, AdaIN removal, high-frequency conditioning
removal, `lambda_LL=1.0`, direct target endpoint, and a learned HH head. Together they cover
the active style path, source-aligned endpoint, LL weighting, target conditioning, and the
decision not to learn HH in the submitted model.

No additional `lambda_LL` sweep is needed. The current matched `0.3` versus `1.0` result
already shows robustness to nonzero LL weights. Older `lambda_LL=0` results come from a
different architecture and must not be mixed into the main table. They may be retained as
historical evidence only.

The frozen SD1.5 extension is also complete: it has before/after DINO-S, CLIP-S, LPIPS, and
DINO-C on 600 cross-style and 750 all-pair protocols, paired significance tests, and measured
operator overhead.

## Conditional experiments

1. **HH seed replication.** Required only if the learned HH head is promoted to the final
   method or if the paper claims that learning HH is generally harmful. Repeat the matched
   15-epoch run with seed 7 and use the frozen checkpoint-selection rule. Under the current
   wording, HH is an unresolved small style-content trade-off and no new run is required.

2. **Uncensored stopping-rule validation.** Required only if internal early stopping is
   presented as a general convergence criterion. Complete uncensored 15-epoch trajectories
   beyond the current architecture and three-seed check, then apply the fixed relative rule
   retrospectively. The current evidence supports an architecture-specific training-dynamics
   criterion, not an architecture-independent guarantee.

3. **Multi-level or alternative wavelets.** Required only if the paper claims that one-level
   Haar is optimal. The current claim is narrower: Haar is orthonormal, local, inexpensive,
   and effective in the tested setting. Multi-scale styles remain a stated limitation.

## Experiments not needed

- Further AdaIN-strength tuning.
- Further `lambda_LL`, endpoint-blend, or learning-rate sweeps.
- WCT controls, which test a destructive alternative rather than a core component.
- A human study for the current submission.
- Additional architecture variants that do not answer one of the conditional claims above.
