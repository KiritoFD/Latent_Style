# Archives Index

Updated: 2026-06-03

This directory stores retired source snapshots, old paper workspaces, and
experiment bundles that should remain auditable but no longer belong in the
active working surface.

## Top-level groups

- `code_backups/`
  - source snapshots taken before major refactors or restorations
- `exp_archive_20260526_051536/`
  - early local experiment archive
- `old_experiment_dirs/`
  - preserved historical experiment trees and recovered eval artifacts
- `old_paper_workspaces/`
  - superseded paper workspaces and figure-generation environments
- `old_root_files/`
  - retired root-level scripts, manifests, and historical reports
- `root_level_snapshots/`
  - coarse repository snapshots kept for recovery

## High-value preserved references

These are the archived paths most likely to matter again during paper writing
or provenance recovery.

### Historical strict-750 timing / table support

- `old_root_files/training_times_documentation.md`
  - retained timing note for the historical strict-750 paper point
- `old_root_files/combined_750_with_destructive_ablations.*`
  - older comparison exports kept for auditability

### Historical recovered eval artifacts

- `old_experiment_dirs/grid_search_3epoch/`
  - recovered historical evaluation bundles, including aggregate ArtFID payloads
    used by later comparison notes

### Pre-refactor source recovery

- `code_backups/src_backup_pre_sadd_restore_20260528_231430/`
  - source snapshot before the S-add restore path
- `code_backups/src_before_concept_atoms_20260529_003635/`
  - source snapshot before the concept-atom tokenizer branch

### Superseded paper workspaces

- `old_paper_workspaces/paper_cn/`
- `old_paper_workspaces/paper_orchestra_workspace/`
- `old_paper_workspaces/paper_refine_v2/`

These are draft-history surfaces only. They are useful for recovering figure or
text provenance, but they are not the current paper source of truth.

## Active-vs-archived rule

An artifact belongs here when it is:

- needed for provenance or recovery,
- cited by a historical note,
- or useful for comparison recovery,

but is no longer part of the active code path, current paper build path, or
main experiment queue.

For the current active paper/evidence tree, start from:

- `docs/aaai2027_working_index_20260602.md`
- `docs/experiments/README.md`
- `docs/reviews/README.md`

For the current cleanup and `exp/` retention policy, also read:

- `docs/experiments/2026-06-03-repo-cleanup-and-archive-pass.md`
- `docs/experiments/2026-06-03-exp-surface-classification.md`

## Citation / promotion rule

Do not cite an archived path directly in the manuscript unless one of the
following is true:

1. the archived artifact is explicitly named in a current `docs/experiments/`
   note or the working index; or
2. the archived artifact has been re-promoted into the current evidence graph
   through:
   - a dated experiment note,
   - a ledger row,
   - and a clear paper-facing explanation of why the archive path still matters.

This keeps the archive auditable without letting it become a shadow active
workspace.
