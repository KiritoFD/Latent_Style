# Experiment 003: OT vs Random Coupling — Velocity Variance

## Goal
Test Proposition 5: Does OT coupling reduce endpoint supervision variance?

## Method
- Model: D0 (full control), epoch 7
- Data: 10 batches × 64 samples = 640 content latents
- Compare:
  - **OT coupling**: Sinkhorn on SWD cost matrix
  - **Random coupling**: uniform permutation
- Metrics:
  - Transport cost (SWD × OT plan)
  - Velocity displacement magnitude ||z̃₁ - z₀||²
  - Directional coherence (cosine similarity to mean direction)

## Results

| Metric | OT | Random | Ratio |
|--------|-----|--------|-------|
| Transport cost | 0.1185 ± 0.003 | 0.1238 ± 0.004 | **0.958** |
| Displacement² mean | 5069.7 ± 158.5 | 5109.1 ± 121.9 | 0.992 |
| Displacement² CV | 0.237 ± 0.042 | 0.211 ± 0.013 | 1.124 |
| Directional cos sim | **0.150 ± 0.010** | **0.124 ± 0.011** | **1.206** |

## Key Findings

1. **Cost reduction confirmed**: OT reduces transport cost by ~4.2%
2. **Directional coherence significantly improved**: OT produces ~21% higher cosine similarity to mean direction
3. **Magnitude variance NOT reduced**: OT has slightly higher displacement variance (CV 0.237 vs 0.211)
4. **Overlap between OT and random**: The cost difference is modest because batch size (64) is large and random matches are often reasonable

## Revised Interpretation

The primary benefit of OT coupling is NOT variance reduction but **directional alignment**.
OT-matched displacements point in more consistent directions, which means:
- The model receives more coherent supervision about which direction to move content
- This likely produces straighter, more efficient transport trajectories
- The kinetic loss penalizes the magnitude regardless of coupling

## Implication for Proposition 5

Proposition 5 should be reformulated from:
> "OT coupling reduces endpoint supervision variance"

To:
> "OT coupling produces more directionally coherent endpoint supervision, improving the geometric consistency of the learned transport field"

This is a more accurate and still meaningful theoretical contribution.
