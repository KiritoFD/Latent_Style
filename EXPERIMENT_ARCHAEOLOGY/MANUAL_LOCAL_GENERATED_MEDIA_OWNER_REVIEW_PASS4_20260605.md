# Local Generated Media Owner Review Pass 4 - 2026-06-05

This pass continues local generated-media archaeology below the pass3 cluster.
The candidate list was only used to pick exact paths. Each row in the CSV was
then checked by opening the exact path plus nearby summaries, metrics, docs,
scripts, prior policies, or duplicate peers.

## Decisions

### Retained Evidence

- `SchrodingerBridge\exp\local_wsl_distinct5_512_ema_k_b16_step2min_v350\full_eval\step_000350`
- `SchrodingerBridge\external_eval\samst_wikiart5_target_e05_750`
- `SchrodingerBridge\external_eval\samst_wikiart5_target_e10_750`
- `SchrodingerBridge\external_eval\samst_wikiart5_target_e15_750`
- `exp\highres_eval_local\same_test_eval\samst\images`
- `exp\highres_eval_local\same_test_eval\lbm\images`
- `exp\highres_eval_local\samst\images`

The v350 directory is a closed Distinct5 same-cost packet. It has
`summary.json`, `metrics.csv` with 750 rows, `summary_grid.png`, 750 generated
images, and direct docs/timing references. The SaMST external-eval directories
are quality packets with summaries, metrics, grids, and images; their summaries
do not contain inference timing and should not be promoted as timing rows. The
highres directories remain qualitative evidence because they have no in-dir
summary/metrics and prior policies already kept the highres surface while
deleting only proven duplicate archives.

### Retained Pending Owner Decision

- `exp\highres_eval_local\samst_same_test_v2\images`
- `SchrodingerBridge\exp\diagnostics\seedream_gap_inputs\styleemb_m02_highpass`
- `SchrodingerBridge\exp\diagnostics\seedream_gap_inputs\ema_transport_adain_w34_e6`

`samst_same_test_v2` is not byte-identical to the retained paired SaMST output
directory, so it cannot be deleted as a duplicate. The two `seedream_gap_inputs`
directories are diagnostic input candidates under the diagnostics parent, backed
by source CLIP geometry and candidate style separability notes. They remain
pending owner review because they have no in-dir summary but also are not proven
duplicate or orphan media.

### Delete Whitelist

- `SchrodingerBridge\external_eval\samst_wikiart5_target_e15_750_images.zip`
- `exp\highres_eval_local\samst_same_test`

The zip has 750 image entries and all entry basenames plus byte lengths match
the retained `samst_wikiart5_target_e15_750\images` directory. The standalone
`samst_same_test` directory contains only an `images` child; all 750 files
SHA256-match the retained paired directory
`exp\highres_eval_local\same_test_eval\samst\images`.

## Cleanup Boundary

No paper TeX/PDF files, source code, tracked dirty files, datasets, metric
summaries, timing docs, retained highres paired evidence, checkpoint tar, or
Seedream diagnostic inputs are deleted by this pass. Deletion is limited to the
two explicit whitelist targets above and must be followed by post-delete
verification.

## Post-Delete Verification

Deletion was executed only for the two whitelist targets. The cleanup ledger is
`cleanup/manual_local_generated_media_pass4_cleanup_20260605.csv`, totaling
`101.913 MB` released. Post-delete verification passed all 11 checks in
`manual_local_generated_media_pass4_post_delete_verify_20260605.csv`: both
deleted targets are absent, while the retained e15 summary/metrics/grid/images,
paired highres SaMST/LBM images, non-identical highres v2 variant, and both
Seedream diagnostic input dirs remain present.
