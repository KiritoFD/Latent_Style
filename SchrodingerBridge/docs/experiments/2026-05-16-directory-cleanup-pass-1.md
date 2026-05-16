# 2026-05-16 Directory Cleanup Pass 1

## Scope

This pass covers documentation alignment and root-directory hygiene only.

## Decisions Made

1. Declared `paper_orchestra_workspace/aaai_submission/` as the canonical paper workspace.
2. Declared `maths/` as the canonical theory workspace.
3. Declared `EXPERIMENT_LOG.md` as the empirical source of truth.
4. Marked `paper_refine_v2/` as a legacy refinement workspace rather than the active paper root.

## Why This Matters

Recent work had produced a split between the actual paper location and the older refinement workspace. Without clearing that up first, later theory/model edits would continue landing in the wrong place.

## Files Updated In This Pass

- `PROJECT_OVERVIEW.md`
- `README.md`
- `DIRECTORY_CLEANUP_LOG.md`
- `docs/experiments/2026-05-16-directory-cleanup-pass-1.md`

## Next Cleanup Step

Move clearly non-active packaged snapshots out of the root and keep the working surface narrow.

## Pass 2 Completed

After this note was created, a second cleanup step was executed:

- moved `grid_search_3epoch.rar` into `archives/root_level_snapshots/`
- moved `review_additional_experiments.rar` into `archives/root_level_snapshots/`
- moved `src.zip` into `archives/root_level_snapshots/`
- removed root `__pycache__/`

This was intentionally limited to archive-like and cache-like files only.
