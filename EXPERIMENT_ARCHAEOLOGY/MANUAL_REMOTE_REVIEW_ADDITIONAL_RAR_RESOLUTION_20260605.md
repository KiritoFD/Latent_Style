# Manual Remote Review Additional RAR Resolution - 2026-06-05

Remote target:

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
I:\Github\Latent_Style\SchrodingerBridge\review_additional_experiments.rar
```

This pass resolved the earlier uncertain state for `review_additional_experiments.rar`. The first remote surface recheck kept it because no reliable RAR listing proof had been recorded. This pass reopened the exact path and used Windows `tar.exe` on the remote host.

## Fixed-Path Proof

| item | value |
| --- | ---: |
| RAR pre-delete size | 3,136,734,074 bytes |
| RAR pre-delete MB | 2991.423 MB |
| `tar.exe -tf` exit | 0 |
| archive entries listed | 58421 |
| expanded directory files | 58151 |
| expanded directory MB | 1270.619 MB |
| expanded weights | 9 |
| expanded `summary.json` | 77 |
| expanded training CSV | 9 |
| expanded metric-like CSV/JSON | 183 |

The archive listing and expanded directory were compared by normalized relative path.

After including expanded directory entries as well as files, archive-only entries were:

- 63 `.pt` checkpoint files under `review_additional_experiments/lambda_grid/runs/*/epoch_0001.pt` through `epoch_0007.pt`.
- 1 archive root directory entry: `review_additional_experiments`.

Expanded directory extra entries were `0`.

## Decision

`review_additional_experiments.rar` was whitelisted for deletion because:

- the same-name expanded directory retained all non-weight evidence;
- summaries, metrics, status files, images, configs, logs, and training CSVs were covered by the expanded directory;
- the only substantive archive-only payload was 63 non-mainline intermediate `.pt` checkpoints;
- the owner objective is to release space by removing non-mainline checkpoint payloads after indexing.

This was not a broad archive cleanup. It was one exact RAR file, with a policy row and post-delete verification.

## Execution

Deleted exact path:

```text
I:\Github\Latent_Style\SchrodingerBridge\review_additional_experiments.rar
```

Deletion ledger:

- `cleanup/manual_remote_review_additional_rar_delete_execution_20260605.csv`

Policy:

- `manual_remote_review_additional_rar_resolution_policy_20260605.csv`

Post-delete verification:

- `manual_remote_review_additional_rar_post_delete_verify_20260605.csv`

## Post-Delete Verification

| check | status |
| --- | --- |
| RAR absent | pass |
| expanded directory present | pass |
| expanded files = 58151 | pass |
| expanded weights = 9 | pass |
| expanded `summary.json` = 77 | pass |
| expanded training CSV = 9 | pass |
| expanded metric-like CSV/JSON = 183 | pass |

## Cleanup Boundary

No other file or directory was deleted in this pass.

Remaining review-additional cleanup, if desired, must be separate:

- thinning any of the 9 retained expanded weights;
- pruning generated images;
- pruning tiny step-count or lambda-grid directories;
- migrating owner-selected representative samples.
