# Theory Verification Experiments

## Overview

This directory contains experiment logs for verifying the propositions in
the theory development draft (`../theory_development.tex`).

## Key Finding (pre-experiment)

**ALL configs use `objective_mode: "omf"`**, not `"flow_matching"`.
The training uses one-step endpoint matching (t=1), not time-sampled bridge states.
This has major implications for the theory:
- Proposition 1 (conditional velocity estimation) must be adapted to the OMF setting
- The "flow matching" in the paper describes an idealized version not reflected in code
- The actual training couples endpoint prediction, kinetic regularization, and terminal SWD

## Experiment Directory

| # | Name | Goal | Propositions |
|---|------|------|-------------|
| 001 | Step-count endpoint error | Measure ||z_K - z_256|| for K=1,4,8,12,16 | Prop 3 |
| 002 | Velocity field t-dependence | Measure ||v_θ||^2 across t ∈ [0,1] | Prop 2 |
| 003 | OT vs random coupling | Compare velocity variance | Prop 5 |
| 004 | Terminal SWD across steps | SWD vs integration steps | Prop 4 |

## Checkpoints Used

- D0: `ablation_destructive_7epoch/D0_full_correct_7ep/epoch_0007.pt` (full control, w_kinetic=1.0, terminal_swd=20.0)
- K1: `manual_k1_k2_8epoch/K1_manual_weighted_8ep/epoch_0008.pt` (w_kinetic=1.0, terminal_swd=20.0)
- K2: `manual_k1_k2_8epoch/K2_manual_weighted_8ep/epoch_0008.pt` (w_kinetic=2.0, terminal_swd=20.0)
- D1: `ablation_destructive_7epoch/D1_no_terminal_swd/epoch_0007.pt` (no terminal SWD)
- D2: `ablation_destructive_7epoch/D2_no_kinetic/epoch_0007.pt` (no kinetic regularization)
