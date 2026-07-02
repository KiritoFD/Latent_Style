# Distinct5-512 Latent Baseline Convergence Closure

Date: 2026-06-07

Scope:

- `latent SaMST` on `Distinct5-512`
- `latent SaMAM` on `Distinct5-512`
- goal: run both baselines far enough to replace "did not run" or "same-cost only" with a defensible convergence reading

## Bottom line

- `SaMST-latent` did converge, but into a bad-solution plateau.
- `SaMAM-latent` did not stop at the first same-cost lift. After the corrected latent patching fix, it continued improving past `step1000`, then settled into a late trade-off band around `step1200-1500`.

The final paper-facing convergence picks for the latent baselines are collected in:

- [2026-06-07-distinct5_latent_baseline_convergence.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-distinct5_latent_baseline_convergence.csv)

## Selected convergence points

### 1. `SaMST-latent`

Run roots:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_convergence_20260606_180529`
- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_convergence_20260606_214051`

Observed retained points:

| label | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `batch300_fast` | `0.6893` | `0.8382` |
| `batch950_fast` | `0.6944` | `0.8409` |
| `batch1050_fast` | `0.6820` | `0.8318` |

Reading:

- the run is no longer a smoke failure or NaN case
- later checkpoints do not move toward a useful frontier
- the entire late band remains far above the no-op line on style but catastrophically weak on LPIPS

We keep `batch1050_fast` as the final convergence artifact because it is the latest retained point and the least-damaging LPIPS point inside the late plateau.

## 2. `SaMAM-latent`

Run roots:

- corrected same-cost seed:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_samecost_20260606_162933`
- first convergence continuation:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_convergence_20260606_222608`
- late continuation from `step1300`:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_convergence_20260607_011328`

Observed retained points:

| label | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `step0020_fast` | `0.6297` | `0.7823` |
| `step0110_fast` | `0.6388` | `0.7042` |
| `step0300_fast` | `0.6223` | `0.5650` |
| `step0600_fast` | `0.6541` | `0.5468` |
| `step1000_fast` | `0.6667` | `0.2744` |
| `step1200_fast` | `0.6550` | `0.1739` |
| `step1300_fast` | `0.6533` | `0.2198` |
| `step1500_fast` | `0.6547` | `0.1635` |

Reading:

- the corrected latent adaptation clearly crosses the Distinct5 `idt` line after `step600`
- `step1000` is the raw style peak of the late run family
- `step1200`, `step1300`, and `step1500` form a narrow late style band around `0.654-0.655`
- within that band, `step1500` gives the best LPIPS of the retained late checkpoints
- the cumulative training wall to `step1500` is about `140.6 min`
  - compared with the reviewed compact `LANCET` anchors on the same split:
    - `F e1`: about `1.2 min`
    - `H e2`: about `2.3 min`
    - `K e1`: about `1.2 min`

That is sufficient to treat `step1500_fast` as the final convergence pick, with `step1200-1500` as the late plateau band.

## Claim boundary

The late latent baselines support the following bounded reading on `Distinct5-512`:

- `SaMST-latent` is a negative baseline that converges to a bad trade-off region.
- `SaMAM-latent` is not "below idt forever"; it eventually rises above `idt`, but still does not stably beat the compact `LANCET` frontier on transfer style.
- `SaMAM-latent` does beat the compact `LANCET` anchors on LPIPS at late convergence, but only after roughly `140.6 min` of training, so this does not erase the compact model's efficiency advantage.
- the main value of the latent-baseline packet is therefore diagnostic:
  - these adapted baselines can be made operational and can converge,
  - but their converged operating regions remain weak relative to the reviewed compact transport frontier.

## Figures

- `SaMAM` latent curve with `idt`:
  - [fig_samam_latent_distinct5_curve.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai_submission/figures/fig_samam_latent_distinct5_curve.png)
- `SaMAM` latent vs. compact `LANCET` points:
  - [fig_samam_latent_vs_lancet.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai_submission/figures/fig_samam_latent_vs_lancet.png)
