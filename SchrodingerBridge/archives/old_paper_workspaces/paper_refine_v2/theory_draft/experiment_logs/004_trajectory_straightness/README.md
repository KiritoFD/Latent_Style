# Experiment 004: Trajectory Straightness Analysis

## Goal
Characterize the straightness of learned transport trajectories and compare across ablations.

## Method
Compute per-trajectory metrics for 128-step Euler integration:
1. **Path Length Ratio (PLR)** = ||z_K - z_0|| / Σ||Δz_k|| — 1.0 = perfectly straight
2. **Max Normalized Deviation** = max perpendicular distance from chord / chord length
3. **Energy Ratio** = first-half displacement / total displacement
4. **Directional Consistency (DC)** = mean cos(v_k, v_{k+1}) — 1.0 = no direction change

Compare: D0 (full), D1 (no terminal SWD), D2 (no kinetic). 5 batches × 32 samples each.

## Results

| Metric | D0 (full) | D1 (no SWD) | D2 (no kinetic) |
|--------|-----------|-------------|-----------------|
| **Path Length Ratio ↑** | **0.9811** | **0.9995** | **0.9425** |
| Directional Consistency | 1.0000 | 1.0000 | 0.9999 |
| Max Normalized Dev ↓ | 0.0859 | 0.0132 | 0.1483 |
| Energy Ratio (1st half) | 0.5614 | 0.5009 | 0.5834 |

## Analysis

### The three losses produce distinct trajectory signatures

1. **D1 (no terminal SWD) ≈ perfect straight line (PLR 0.9995)**
   - Without endpoint distribution matching, the model learns the simplest possible transport: linear interpolation from content to OT-matched target
   - Symmetric energy split (50/50) confirms equal movement throughout
   - Max deviation only 1.3% — effectively a straight line

2. **D0 (full control) = slightly curved but very straight (PLR 0.9811)**
   - Terminal SWD adds a correction: trajectories bend slightly to match target patch statistics
   - The bending is small (max dev 8.6%) — style transfer happens as a perturbation of linear transport
   - Slight front-loading (56/44 split): initial movement toward target, then correction at the end

3. **D2 (no kinetic) = noticeably curved (PLR 0.9425)**
   - Without displacement regularization, the velocity field produces winding trajectories
   - Max deviation 14.8% — nearly double D0's curvature
   - Energy front-loaded (58/42) — large initial movement, then corrections
   - **This validates Proposition 2: kinetic regularization controls path straightness**

### Implications for theory

- The learned transport is **nearly linear**: all models have PLR > 0.94
- Kinetic regularization improves straightness by **4%** (D0 vs D2 PLR difference)
- Terminal SWD introduces **controlled curvature** for style matching (D0 vs D1: PLR 0.98 vs 1.00)
- The flow matching + kinetic combo ensures efficient transport; terminal SWD makes it style-correct

### Connection to Proposition 5 (OT directional coherence)

The very high directional consistency (DC ≈ 1.0) across all models suggests that OT coupling produces well-aligned velocity fields regardless of the loss configuration. This is consistent with Experiment 3's finding that OT improves directional coherence.
