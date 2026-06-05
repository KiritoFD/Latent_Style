# Remote Cycle-NCE 45.rar Delete Executed - 2026-06-05

Exact target deleted:

```text
I:\Github\Latent_Style\Cycle-NCE\45.rar
```

Released space: `507.452 MB`.

## Why This Was Safe

The prior curated extraction pass created and verified:

```text
I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605
```

Evidence already recorded:

- `6084 / 6084` expected nonweight payload entries matched by relative path and
  byte size.
- `12 / 12` `.pt` weight entries were recorded in the removed-weight ledger and
  removed from the curated staging package.
- The remote curated package currently has `6086` files, including the package
  manifest and removed-weight ledger.
- Recursive weight-extension count in the curated package is `0`.

## Execution

Deletion was performed with a PowerShell `Remove-Item -LiteralPath` call against
the exact absolute path only.

Execution ledger:

- `manual_remote_cycle_nce_45_rar_delete_execution_20260605.csv`

## Post-Delete Verification

All checks passed:

- `45.rar` absent.
- Curated nonweight package present.
- Extracted `45\` payload directory present.
- `_curated_nonweight_manifest_20260605.csv` present.
- `_removed_weight_files_20260605.csv` present.
- Curated package recursive weight-extension count remains `0`.

Verification ledger:

- `manual_remote_cycle_nce_45_rar_post_delete_verify_20260605.csv`

## Boundary

No local paper/source files were touched. No expanded `Cycle-NCE` evidence
directory was deleted. The only removed remote object was the original compressed
archive `I:\Github\Latent_Style\Cycle-NCE\45.rar`.
