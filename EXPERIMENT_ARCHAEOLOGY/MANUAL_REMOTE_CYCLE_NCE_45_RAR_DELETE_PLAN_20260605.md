# Remote Cycle-NCE 45.rar Delete Plan - 2026-06-05

Target:

```text
I:\Github\Latent_Style\Cycle-NCE\45.rar
```

This is an exact-path delete whitelist for the original `45.rar` archive after
the curated nonweight extraction pass.

## Pre-Delete Evidence

- Original archive exists remotely: `507.452 MB`, last write
  `2026-04-06 00:11:44`.
- Curated nonweight package exists remotely:
  `I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605`.
- Curated package current remote recursive count: `6086` files,
  `145.512 MB`, `0` files with weight extensions.
- Package-local root contains:
  `_curated_nonweight_manifest_20260605.csv`,
  `_removed_weight_files_20260605.csv`, and extracted `45\`.
- Local verification already matched `6084 / 6084` expected nonweight payload
  entries by relative path and byte length.
- Local verification already matched `12 / 12` expected `.pt` weight entries in
  the removed-weight ledger.

## Delete Boundary

Delete only:

```text
I:\Github\Latent_Style\Cycle-NCE\45.rar
```

Do not delete:

- `I:\Github\Latent_Style\Cycle-NCE\_curated_45_nonweight_20260605`
- `I:\Github\Latent_Style\Cycle-NCE\45`
- any expanded `Cycle-NCE` evidence directories
- any local files outside `EXPERIMENT_ARCHAEOLOGY`

## Required Post-Delete Verification

- `45.rar` absent.
- Curated nonweight package present.
- Package manifest present.
- Removed-weight ledger present.
- Recursive package weight-extension count remains `0`.

Execution and post-delete verification must be recorded in separate CSVs before
commit.
