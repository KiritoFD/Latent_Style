# Remote Main Surface Recheck - 2026-06-05

Remote:

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
I:\Github\Latent_Style
```

This is a fixed-path recheck block. It is not a one-shot recursive scan and no
remote deletion was performed.

## Paths Opened

| path | current finding | decision |
| --- | --- | --- |
| `I:\Github\Latent_Style` | exists, 23 top-level dirs, 53 top-level files | keep / continue exact-path review |
| `SchrodingerBridge` | exists, 32 top-level dirs, 175 top-level files | mixed evidence surface |
| `SchrodingerBridge\exp` | exists, 123 dirs, 1 file after previous epoch thinning | keep anchors |
| `SchrodingerBridge\review_additional_experiments.rar` | exists, 3136734074 bytes | keep pending archive provenance |
| `SchrodingerBridge\review_additional_experiments` | exists, 58151 files, 1270.619 MB, 9 weights, 77 summaries, 9 training CSVs | keep |
| `Related_Works` | exists, 5 top-level dirs | baseline/source/dependency surface |
| `Related_Works\runs\cut_5x5\infer_5x5` | 2427 JPG images and one 1531-byte fake checkpoint; no summary/metrics/meta | retain pending owner |
| `Related_Works\baseline_pipeline\results` | 16 result dirs opened | mixed baseline evidence |
| `Cycle-NCE\45.rar` | absent | already deleted and verified |
| `Cycle-NCE\_curated_45_nonweight_20260605` | present, 6086 files, 145.512 MB, 0 weight-extension files | keep |

Row-level CSV:

- `manual_remote_main_surface_recheck_20260605.csv`

## Manual Conclusions

- Remote main is not empty. The earlier empty-remote concern is false on live
  check.
- `review_additional_experiments.rar` is a real cleanup candidate by size, but
  it cannot be deleted yet. The remote host has no `7z`, `rar`, or `unrar`, so
  the RAR contents were not listed. The same-name directory exists and contains
  evidence, but that is not enough proof to delete the archive.
- `Related_Works\runs` on remote main is cleaner than local: only `cut_5x5`
  remains. Its `infer_5x5` directory lacks summary/metrics/meta and is
  qualitative media only; deletion still needs owner approval and representative
  sample retention.
- `baseline_pipeline\results` contains many tiny probe directories and one large
  retained SaMAM diagnostic anchor:
  `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`
  has 12128 files, 7232.586 MB, 12 ckpt files, and 24 metric-like files. It was
  not deleted because those checkpoints are retained SaMAM curve anchors.
- `Cycle-NCE\45.rar` remains absent and the curated nonweight package is still
  present with no weight-extension files.

## Cleanup Boundary

All rows remain `delete_whitelist=no`.

Next cleanup candidates require separate proof:

- `review_additional_experiments.rar`: needs archive listing/extraction proof.
- Remote CUT qualitative media: needs owner-approved representative-sample
  retention or migration.
- Tiny/empty SaMAM/Flux/ZImage/SaMST probe dirs: only worth pruning if owner
  wants cosmetic cleanup; they release negligible space.
- SaMAM diagnostic checkpoints: require a separate checkpoint-thinning policy,
  not broad deletion.
