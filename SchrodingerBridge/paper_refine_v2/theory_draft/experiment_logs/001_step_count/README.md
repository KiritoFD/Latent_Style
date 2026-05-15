# Experiment 001: Step-Count Endpoint Error Analysis

## Goal
Verify Proposition 3: Euler integration endpoint error ||z(1) - z_K|| = O(Δt) = O(1/K).

## Method
- Model: D0 (full control, w_kinetic=1.0, terminal_swd=20.0), epoch 7 checkpoint
- Data: 5 batches × 32 samples = 160 content latents
- Reference: 256-step Euler integration as ground truth
- Test: K ∈ {1, 2, 4, 8, 12, 16, 32, 64, 128, 256}
- Metric: ||z_K - z_256||_2 mean over batch

## Results

| Steps | Abs Error (mean±std) | Scaling (Err(K)/Err(2K)) |
|-------|---------------------|-------------------------|
| 1     | 10.67 ± 1.03        | -                       |
| 2     | 3.80 ± 0.29         | 1/2.8                   |
| 4     | 1.74 ± 0.13         | 1/2.2                   |
| 8     | 0.835 ± 0.065       | 1/2.1                   |
| 12    | 0.546 ± 0.045       | 1/1.8                   |
| 16    | 0.403 ± 0.035       | 1/1.4                   |
| 32    | 0.190 ± 0.021       | 1/2.1                   |
| 64    | 0.084 ± 0.013       | 1/2.3                   |
| 128   | 0.027 ± 0.002       | 1/3.1                   |
| 256   | 0 (reference)       | -                       |

## Analysis
- Clear O(1/K) scaling: doubling steps halves error (ratio ~2.0-2.3 for mid-range K)
- At 4 steps: error is 1.74, which in latent space corresponds to ~0.087 VAE units per channel
  - This is small relative to VAE latent range (typically ±3-4)
- At 12 steps (paper default): error is 0.55, very small
- Scaling slightly better than O(1/K) at high K (ratio 3.1 at 128→256) suggesting near-convergence
- Paper's observation of flat quality across 4-16 steps is consistent: even at 4 steps, error is small relative to style effect size

## Conclusion
Proposition 3 (O(Δt) Euler error bound) is validated. The paper's step-count flatness has a clear numerical basis.
