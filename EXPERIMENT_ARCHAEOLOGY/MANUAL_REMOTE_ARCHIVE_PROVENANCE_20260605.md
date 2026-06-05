# Remote Archive Provenance and Cleanup - 2026-06-05

Remote root:

`I:\Github\Latent_Style`

This block is the follow-up to the remote main data/cache/archive residue pass. It checks large archives one by one and deletes only archives with enough provenance evidence.

## Probe Method

Probe file:

`manual_remote_archive_provenance_probe_20260605.csv`

Read-only checks performed:

- Checked whether archive tools are available on remote: `7z`, `7za`, `rar`, `unrar`, `WinRAR` were not found.
- Listed archive file size and last-write time.
- Counted matching directories and sizes.
- Computed SHA256 for selected duplicate candidates.
- Used .NET ZipFile to inspect `.zip` entries without extracting.

Because no RAR tool is available, `.rar` and multipart `.rar` archives cannot be content-listed in this pass. Those remain pending unless exact duplicate hash proof exists.

## Deleted Archives

Policy CSV:

`manual_remote_archive_provenance_policy_20260605.csv`

Delete whitelist:

`manual_remote_archive_delete_candidates_20260605.csv`

Cleanup ledger:

`cleanup/manual_remote_duplicate_archive_cleanup_20260605.csv`

Post-delete verification:

`manual_remote_archive_post_delete_verify_20260605.csv`

Deleted files:

| path | MB | evidence |
| --- | ---: | --- |
| `eval_cache.zip` | 704.467 | Zip had 28 entries; 27 same-size entries still exist in `eval_cache`. The only missing entry is the failed CLIP `.incomplete` already deleted as invalid residue. |
| `Cycle-NCE\1-decoder-patch5-15_eAzEC.zip` | 2078.795 | Zip had 1514 entries; 1510 same-size nonweight outputs exist. The four missing entries are old epoch checkpoint weights: `epoch_0020.pt`, `epoch_0030.pt`, `epoch_0040.pt`, `epoch_0060.pt`. |
| `Cycle-NCE\src\45.rar` | 507.452 | SHA256 exactly matches retained `Cycle-NCE\45.rar`: `8810DA5CBA3E158B2F28A2E07225E568E88E6C4DF4B7326D702B984C9F2F2D9E`. |

Result:

- Deleted 3 exact archive targets.
- Freed `3290.714 MB`.
- Post-delete verification: all three deleted targets have `exists=False`.
- Retained evidence still exists: `eval_cache`, `experiments\1-decoder-patch5-15`, root `Cycle-NCE\45.rar`, and `Cycle-NCE\src`.

## Retained Archives

Retained because content could not be proven disposable:

- `experiments.rar` - `8091.026 MB`: matching `experiments` directory exists with 343186 files / 8715.036 MB, but RAR content cannot be listed without a RAR tool.
- `Cycle-NCE\Gate.rar` - `3384.032 MB`: matching `Cycle-NCE\Gate` directory exists, but RAR content cannot be listed.
- `Cycle-NCE\Attn_48.part1.rar`, `part2.rar`, `part3.rar`: matching `Cycle-NCE\Attn_48` directory exists, but multipart RAR content cannot be listed.
- `Cycle-NCE\chess.part1.rar`, `part2.rar`: matching `Cycle-NCE\chess` directory exists, but multipart RAR content cannot be listed.
- `Cycle-NCE\45.rar`: retained as the primary copy after deleting exact duplicate `Cycle-NCE\src\45.rar`.
- `Cycle-NCE\src_BGmRM.7z`: tiny archive, no extractor/provenance decision needed for disk recovery.
- `Cycle-NCE\summary_fhJh7.zip`: retained because it contains 37 summary JSON entries not found in matching directories and is only 0.024 MB.

## Remaining Archive Gaps

- Install/use a RAR-capable tool if owner wants to prove `experiments.rar`, `Gate.rar`, `Attn_48.part*.rar`, and `chess.part*.rar` duplicate existing directories.
- Do not delete those RAR archives by size alone; this pass intentionally stops at proven duplicates/stale archives.
- Cross-cache dedup still remains separate from archive provenance.
