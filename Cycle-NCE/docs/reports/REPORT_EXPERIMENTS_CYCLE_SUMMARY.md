# Experiments-Cycle Summary (Detailed Pass)

- Generated: 2026-02-13 15:56:59
- Total runs: `53`
- Strict comparable runs: `25`
- Per-run records: `docs/reports/experiments_cycle_records/*.md`

## Key Conclusions

- Best strict score is `0.551533` from `full_300_distill_low_only_v1`.
- The evaluation pipeline should keep strict-comparable outputs (`summary.json` + non-zero LPIPS + complete matrix) for every run.
- Main optimization tradeoff remains style gain vs content retention (LPIPS drift).

## Top Strict Runs

| run | path | best_style | cls | lpips |
|---|---|---:|---:|---:|
| full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | 0.551533 | 1.000000 | 0.697327 |
| overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | 0.549101 | 0.890000 | 0.764001 |
| overfit50-style-distill-struct-v1-20 | `experiments/overfit50-style-distill-struct-v1-20` | 0.547901 | 0.970000 | 0.670641 |
| overfit50-style-force-schedule | `experiments/overfit50-style-force-schedule` | 0.543730 | 0.260000 | 0.582351 |
| overfit50-style-distill-struct-v2 | `experiments/overfit50-style-distill-struct-v2` | 0.534868 | 0.880000 | 0.645370 |
| overfit50-strong_structure | `overfit50-strong_structure` | 0.534638 | 0.120000 | 0.581129 |
| overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 0.528368 | 0.850000 | 0.551213 |
| overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | 0.526751 | 1.000000 | 0.606507 |
| overfit50-style-force-balance-v1-cycle4 | `experiments/overfit50-style-force-balance-v1-cycle4` | 0.521914 | 0.520000 | 0.675383 |
| overfit50-distill_low_only | `overfit50-distill_low_only` | 0.518469 | 0.930000 | 0.545553 |

## Recommended Next Steps

1. Use strict-comparable protocol for all new runs to keep ranking fair.
2. Continue from balanced candidates first, then tune style strength in small increments.
3. Keep `summary_history` and `src_snapshot` coverage to improve reproducibility and post-mortem quality.
