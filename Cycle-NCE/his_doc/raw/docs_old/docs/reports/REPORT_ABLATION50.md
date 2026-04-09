# Ablation50 Report

- Generated: 2026-02-13 15:27:53
- Ablation roots found: 3
- Variants indexed: 3

## Root Summary

| root | variants | completed | failed | interrupted | not_started |
|---|---:|---:|---:|---:|---:|
| ablation50_repro | 1 | 0 | 1 | 0 | 0 |
| ablation50_repro_compile | 1 | 0 | 1 | 0 | 0 |
| ablation50_repro_cwdfix | 1 | 0 | 0 | 1 | 0 |

## Variant Details

| root | variant | status | train_epoch | eval_epoch | style | error_hint |
|---|---|---|---:|---:|---:|---|
| ablation50_repro | baseline_50e | failed | - | - | - | dataset_path_mismatch |
| ablation50_repro_compile | baseline_50e | failed | - | - | - | dataset_path_mismatch |
| ablation50_repro_cwdfix | baseline_50e | interrupted | - | - | - | - |

## Key Findings

- Dataset path mismatch failures: 2
- `ablation50_repro` and `ablation50_repro_compile` baseline failed before training due dataset root.
- `ablation50_repro_cwdfix` baseline started training but has no completed epoch/eval output yet.
