# Remote experiments.rar Resolved Duplicate Policy - 2026-06-05

Remote root:
`I:\Github\Latent_Style`

Archive:
`experiments.rar`

## Manual Check Result

This was not accepted from a raw scan alone. The RAR mismatch block was opened and checked in three steps:

1. Deep RAR provenance showed `343177` file entries already existed in the expanded `experiments` tree with the same byte size.
2. The remaining `9` mismatches were opened individually and identified as HuggingFace CLIP snapshot `SymbolicLink` entries under `experiments\eval_cache\hf\models--openai--clip-vit-base-patch32`.
3. Each of the `9` snapshot links was resolved to its `..\..\blobs\...` target; every target exists and its byte size equals the RAR entry byte size.

## Decision

`experiments.rar` is now a delete whitelist target.

Reason: the archive has no remaining unique payload after resolving HF snapshot symlinks to their blob targets. The expanded `experiments` directory remains the retained evidence package, including configs, metrics, logs, generated outputs, eval summaries, snapshot links, and blob payloads.

## Source Files

- `manual_remote_rar_provenance_deep_20260605.csv`
- `manual_remote_experiments_rar_cache_mismatch_20260605.csv`
- `manual_remote_experiments_rar_symlink_targets_20260605.csv`
- `manual_remote_experiments_rar_resolved_policy_20260605.csv`

## Required Post-Delete Verification

- `experiments.rar` absent.
- `experiments` directory present.
- All 9 CLIP snapshot symlink target blobs still present and same-size as the original RAR entry sizes.
