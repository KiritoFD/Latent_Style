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
