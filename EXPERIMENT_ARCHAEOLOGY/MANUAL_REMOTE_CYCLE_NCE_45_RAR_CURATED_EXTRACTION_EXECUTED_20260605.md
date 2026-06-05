# Remote Cycle-NCE 45.rar Curated Extraction Executed - 2026-06-05

This pass executes the previously defined curated extraction path for:

```text
I:\Github\Latent_Style\Cycle-NCE\45.rar
```

No deletion was performed on the original archive. The archive still exists at
`507.452 MB`.

## Remote Package

Created remote package:

```text
I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605
```

Execution path:

1. Extracted all `6096` archive entries into staging.
2. Removed only the `12` `.pt` weight files from the staging package.
3. Left `6084` nonweight payload files in the curated package.
4. Wrote remote package manifest and removed-weight ledger inside the package.
5. Pulled the manifest and ledgers back into `EXPERIMENT_ARCHAEOLOGY`.

The final curated payload is `144.730 MB` and has `0` payload files with weight
extensions (`.pt`, `.pth`, `.ckpt`, `.safetensors`). The directory has `6086`
files including the two package-local manifest/ledger CSVs. A loose filename
scan also flags four retained `model.py` files; those were manually opened and
are source snapshots, not model weights.

## Verification

Local verification compared the pulled remote manifest against
`manual_remote_cycle_nce_45_rar_entry_classes_20260605.csv`.

| check | result |
| --- | ---: |
| expected nonweight entries | `6084 / 6084` matched |
| nonweight missing paths | `0` |
| nonweight size mismatches | `0` |
| manifest extra paths | `0` |
| expected weight entries removed from staging | `12 / 12` matched |
| weight size mismatches | `0` |
| removed-weight extra paths | `0` |

Class-level verification:

| class | entries | MB | status |
| --- | ---: | ---: | --- |
| `config` | 4 | 0.014 | pass |
| `summary_json` | 8 | 0.156 | pass |
| `metrics_csv` | 8 | 0.745 | pass |
| `training_csv` | 5 | 0.040 | pass |
| `source_or_structured` | 42 | 26.704 | pass |
| `other` | 9 | 0.117 | pass |
| `generated_or_eval_image` | 6008 | 116.955 | pass |

## Local Files

- `manual_remote_cycle_nce_45_rar_curated_extraction_execution_20260605.csv`
- `manual_remote_cycle_nce_45_rar_curated_extraction_manifest_20260605.csv`
- `manual_remote_cycle_nce_45_rar_curated_extraction_removed_weights_20260605.csv`
- `manual_remote_cycle_nce_45_rar_curated_extraction_verify_20260605.csv`
- `manual_remote_cycle_nce_45_rar_curated_extraction_class_counts_20260605.csv`

## Current Decision

Current decision: keep `45.rar` for now.

The blocking condition changed. It is no longer blocked by missing nonweight
extraction; that package now exists and is verified. The remaining decision is
whether the old archived weights are disposable and whether deleting the
original compressed archive is acceptable after preserving the extracted
nonweight package.

If owner approves archive deletion, the next step must be a new delete
whitelist plus post-delete verification. Do not delete `45.rar` from this report
alone.
