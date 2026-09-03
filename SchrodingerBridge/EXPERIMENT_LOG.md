# Experiment Log - SchrodingerBridge

Reviewed on `2026-05-16`.

This file is the main narrative log for what we currently believe, based only on runs and documents that are already present in the repository.

## 2026-05-16: Theory Reset Around the D0 -> D2 Frontier

### Objective

Refocus the project around the smallest evidence-backed path to higher `clip_style`, while preserving the already acceptable training speed.

### Trusted Anchors

From `ablation_destructive_7epoch/combined_summary/combined_750_with_destructive_ablations.csv`:

- `D0_full_correct_7ep`: `clip_style = 0.7014`, `clip_content = 0.8022`, `LPIPS = 0.4593`, `train_sec = 290.650`
- `D1_no_terminal_swd`: `clip_style = 0.6708`
- `D2_no_kinetic`: `clip_style = 0.7159`, `clip_content = 0.6624`, `LPIPS = 0.6375`, `train_sec = 303.312`
- `D10_micro_hf_swd_trap`: `clip_style = 0.6989`
- `SaMST strict`: `clip_style = 0.7194`, `clip_content = 0.8193`, `LPIPS = 0.4664`

### What These Numbers Mean

1. Terminal SWD is essential.
2. Kinetic regularization is the main style-content tradeoff knob.
3. Removing kinetic entirely raises style but causes unacceptable collapse.
4. Pushing SWD toward micro high-frequency detail is not the right style path.
5. Speed is already acceptable in the baseline family at about `41 - 44 s / epoch`.

### Theory Decision

The active theory is now:

`min_theta lambda_swd * SWD(z_1, Z_style) + lambda_kin * E ||v_theta||^2`

The main experimental question is:

Can we move from the `D0` point toward the `D2` style frontier without falling into the `D2` collapse regime?

### Current Recommended Restart Point

- checkpoint: `S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`
- family: D0/K1-style OMF branch
- first variable to change: `w_kinetic`
- second variable to change: `terminal_swd_weight`
- speed variable: `semantic_swd_num_projections`

## 2026-05-16: Full CSV-Based Evidence Pass

Canonical writeup:

- `maths/FULL_EVIDENCE_ANALYSIS_2026-05-16.md`
- `maths/CSV_STATISTICAL_APPENDIX_2026-05-16.md`
- `maths/DECISION_TREE_AND_EXPERIMENT_PLAN.md`

What was added in this pass:

1. code-faithful interpretation of `SemanticCrossAttn`, skip routing, and endpoint losses
2. grouped analysis of destructive ablations
3. grouped analysis of `experiments_root`
4. grouped analysis of `weight_sweep_40`
5. grouped analysis of step-size and residual-scale sweeps
6. grouped analysis of theory-switch validation
7. grouped analysis of high-tension and orthogonal phase sweeps

Main repository-level conclusion:

the bottleneck is no longer missing style capacity. The bottleneck is controlling where style amplification enters so that style can rise without the skip path and endpoint update collapsing content.

## 2026-05-16: Aborted Fast K0.5 Attempt

See:

- `docs/experiments/2026-05-16-aborted-fast-k05-attempt.md`

Summary:

- a proposed `K0.5 + P32` continuation was prepared
- the run did not produce valid experimental evidence
- initial failure was a path issue
- later failure was a Windows dataloader worker instability issue
- no metric result from that attempt should be cited as evidence
