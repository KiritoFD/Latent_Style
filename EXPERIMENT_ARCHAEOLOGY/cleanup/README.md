# Checkpoint Cleanup Candidate Manifest

No checkpoints were deleted.

The CSV in this directory lists 38355 checkpoint-like files totaling 27819.6 MB. The `cleanup_class` column is conservative:

- `likely_mainline_keep`: path contains known mainline/AAAI/Distinct5/LBM evidence markers.
- `likely_non_mainline_delete_candidate`: path contains smoke, tmp, archive, old experiment, or run_511 output markers.
- `review_delete_candidate`: not recognized as mainline, requires manual review before deletion.

Use this file as the next-step review gate before any destructive cleanup.
