# Local Archive / Tmp / Paper Scratch Provenance - 2026-06-05

Scope:

```text
G:\GitHub\Latent_Style\archive
G:\GitHub\Latent_Style\tmp
G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission_snapshot_9a4b99dfa_page1_scatter_artfid
G:\GitHub\Latent_Style\SchrodingerBridge\configs\archive\20260605_local_distinct5_ema
G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission
```

This is an exact-path provenance pass for local archive/tmp/paper scratch
surfaces. No deletion was performed.

## Findings

| path | files | MB | git state | decision |
| --- | ---: | ---: | --- | --- |
| `archive` | 3450 | 52.856 | ignored archive tree, 3299 ignored-untracked files | keep_pending_owner |
| `tmp` | 371 | 176.618 | ignored paper/PDF scratch, 371 ignored-untracked files | keep_pending_owner |
| `SchrodingerBridge\aaai_submission_snapshot_9a4b99dfa_page1_scatter_artfid` | 103 | 29.142 | untracked paper snapshot | keep_pending_owner |
| `SchrodingerBridge\configs\archive\20260605_local_distinct5_ema` | 6 | 0.005 | 1 tracked config, 5 untracked configs | keep |
| `SchrodingerBridge\aaai_submission` | 251 | 109.545 | tracked paper workspace with current dirty files | keep_no_touch |

## Manual Interpretation

- `archive` is old cleanup history. It may be removable later, but it is not an
  experiment checkpoint target and must not be swept without owner approval.
- `tmp` is paper/PDF/review scratch from 2026-06-04. It includes PDFs, TeX
  recovery files, page PNGs, PDF review folders, and visual review folders.
  Because the current task explicitly avoids paper TeX/PDF, no cleanup is
  whitelisted here.
- `aaai_submission_snapshot_9a4b99dfa_page1_scatter_artfid` is a full untracked
  paper snapshot. It is not safe to delete as experiment cleanup.
- `configs/archive/20260605_local_distinct5_ema` contains experiment config
  records. They are tiny and useful for provenance.
- `aaai_submission` is the active tracked paper workspace and has unrelated
  dirty files. It is out of cleanup scope.

Row-level CSV:

- `manual_local_archive_tmp_paper_scratch_provenance_20260605.csv`

## Cleanup Boundary

All rows remain `delete_whitelist=no`.

If the owner later wants scratch cleanup, it should be a separate paper-scratch
cleanup task with an exact whitelist and post-delete verification. It should not
be mixed with experiment checkpoint/media cleanup.
