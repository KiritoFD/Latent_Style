# Experiments-Cycle Report

- Generated: 2026-02-13 15:27:53
- Total runs indexed: 53
- Runs with parseable best style metric: 35
- Strict comparable runs: 25

## Top Runs (Strict)

| run | path | best_style | cls_acc | content_lpips | eval_count | history_rounds |
|---|---|---:|---:|---:|---:|---:|
| full_300_distill_low_only_v1 | full_300_distill_low_only_v1 | 0.551533 | 1.000 | 0.697327 | 30 | 0 |
| overfit50-style-force-balance-v1 | experiments/overfit50-style-force-balance-v1 | 0.549101 | 0.890 | 0.764001 | 50 | 0 |
| overfit50-style-distill-struct-v1-20 | experiments/overfit50-style-distill-struct-v1-20 | 0.547901 | 0.970 | 0.670641 | 50 | 0 |
| overfit50-style-force-schedule | experiments/overfit50-style-force-schedule | 0.543730 | 0.260 | 0.582351 | 50 | 0 |
| overfit50-style-distill-struct-v2 | experiments/overfit50-style-distill-struct-v2 | 0.534868 | 0.880 | 0.645370 | 50 | 0 |
| overfit50-strong_structure | overfit50-strong_structure | 0.534638 | 0.120 | 0.581129 | 50 | 0 |
| overfit50-style-distill-struct-v2 | overfit50-style-distill-struct-v2 | 0.528368 | 0.850 | 0.551213 | 50 | 0 |
| overfit50-v5-mse-sharp-style_back | overfit50-v5-mse-sharp-style_back | 0.526751 | 1.000 | 0.606507 | 50 | 1 |
| overfit50-style-force-balance-v1-cycle4 | experiments/overfit50-style-force-balance-v1-cycle4 | 0.521914 | 0.520 | 0.675383 | 50 | 0 |
| overfit50-distill_low_only | overfit50-distill_low_only | 0.518469 | 0.930 | 0.545553 | 50 | 1 |
| overfit50-style-distill-struct-v3 | overfit50-style-distill-struct-v3 | 0.516456 | 0.670 | 0.510532 | 50 | 0 |
| overfit50-strok-style | overfit50-strok-style | 0.516445 | 0.930 | 0.386163 | 50 | 2 |
| full_300-map16+32 | full_300-map16+32 | 0.509921 | 0.780 | 0.424217 | 50 | 6 |
| overfit50-upscale-balance-v2-re-0.1cycle | experiments/overfit50-upscale-balance-v2-re-0.1cycle | 0.506272 | 0.230 | 0.434735 | 50 | 0 |
| overfit50-upscale-balance-v2 | experiments/overfit50-upscale-balance-v2 | 0.504675 | 0.500 | 0.441309 | 50 | 0 |

## Stability Snapshot

- Runs with `summary_history`: 8
- Example long-history baseline: `full_300-map16+32` (6 rounds).

## Notes

- Older runs with `content_lpips=0` are excluded from strict ranking.
- Some runs have image artifacts but no parseable `summary.json`; they are audit-incomplete.
