# Experiment Archaeology

This directory contains the curated local G and remote I experiment archaeology outputs.

## Main Outputs

- `../EXPERIMENT_ARCHAEOLOGY_MASTER.csv`: final root-level master CSV.
- `final_master_experiments.csv`: same final master CSV inside this directory.
- `final_by_dataset/*.csv`: one CSV per dataset/setting family.
- `final_timeline.csv`: chronological experiment event index.
- `EXPERIMENT_TIMELINE.md`: narrative timeline and experiment lineage.
- `remote_i_curated/`: remote-side curated outputs generated after filtering and checkpoint cleanup.
- `cleanup/local_deleted_checkpoints.csv`: local per-file deletion audit.
- `remote_i_curated/remote_i_deleted_checkpoints.csv`: remote per-file deletion audit.
- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`: timing-focused subset.

`final_by_dataset/` is the authoritative per-dataset/per-setting split after merging and reclassifying both G: and I: evidence. `remote_i_curated/` is kept as the remote-side audit bundle generated on I: before final local reclassification.

## Counts

- Final experiment rows: 22629
- Timing rows: 416
- Timeline events: 7829
- Source roots: {'G:/GitHub/Latent_Style': 17134, 'I:\\': 5495}
- Local deleted checkpoints: 329, MB=11575.67
- Remote deleted checkpoints: 246, MB=9074.318

## Dataset Counts

- `cycle_nce`: 11794
- `schrodingerbridge_exp_general`: 4051
- `schrodingerbridge_weight_sweep`: 1285
- `legacy_style_transfer_experiments`: 1120
- `schrodingerbridge_grid_search`: 1013
- `schrodingerbridge_vae_backend`: 699
- `schrodingerbridge_frontier`: 692
- `schrodingerbridge_representation_probe`: 567
- `distinct5_512`: 417
- `wikiart512_5style`: 200
- `schrodingerbridge_root_legacy`: 197
- `legacy256_overfit50`: 131
- `run511_5domain`: 120
- `schrodingerbridge_aaai2027`: 87
- `strict_protocol_750`: 79
- `path_family_final_works`: 75
- `photo_monet_5x5`: 42
- `schrodingerbridge_docs_experiments`: 13
- `schrodingerbridge_destructive_ablation`: 12
- `unclassified_curated_experiments`: 11
- `schrodingerbridge_review_additional`: 10
- `related_works_baselines`: 5
- `path_family_run_summary.csv`: 4
- `s2wat`: 3
- `path_family_step_count_sweep`: 2

## Method Counts

- `unknown`: 11977
- `LANCET/LBM`: 8962
- `IDT`: 554
- `AdaIN`: 512
- `SaMAM`: 190
- `LANCET`: 146
- `No-op`: 39
- `SaMST`: 35
- `LBM`: 29
- `StyTr2`: 28
- `CAST`: 25
- `CUT`: 18
- `idt`: 14
- `AesPA-Net`: 13
- `AesFA`: 12
- `StyleID`: 10
- `S2WAT`: 9
- `SDEdit`: 9
- `CycleGAN`: 3
- `Ours`: 3
- `SD-Turbo`: 2
- `L2 matching cost`: 2
- `Ours D0 full`: 2
- `conv body, no global attention`: 2
- `disable routed skip path`: 2
- `disable spatial style prior`: 2
- `micro high-frequency SWD`: 2
- `no residual path`: 2
- `single terminal step`: 2
- `strong color loss`: 2
- `w/o SWD and kinetic`: 2
- `w/o kinetic`: 2
- `w/o terminal SWD`: 2
- `AdaIN v32k`: 2
- `AdaIN vgg19`: 2
- `LBM-F e1`: 1
- `LBM-H e1`: 1
- `LBM-H e2`: 1
- `LBM-K e1`: 1
- `SaMST e15`: 1

## Validity Classes

- `metric_evidence`: 17053
- `summary_evidence`: 2964
- `log_evidence`: 2250
- `timing_evidence`: 257
- `indexed_curated_evidence`: 105

## Cleanup Rule

Only explicitly non-mainline checkpoint candidates were deleted. Ambiguous `review_delete_candidate` files were retained.


## Conclusion Reports

- `ARCHAEOLOGY_REPORT.md`: full local + remote experiment archaeology conclusion report.
- `LOCAL_G_CONCLUSIONS.md`: local G: repository conclusions and cleanup state.
- `REMOTE_I_CONCLUSIONS.md`: remote I: curated experiment conclusions and cleanup state.
- `EXPERIMENT_LINEAGE_SUMMARY.md`: chronological experiment lineage and reusable-result map.
- `conclusions_by_dataset.csv`: one-row-per-dataset conclusion, timing coverage, source examples, and gaps.
- `cleanup/CLEANUP_AUDIT_SUMMARY.md`: checkpoint and directory cleanup audit summary.

## Manual Evidence Layer

The broad CSVs above are navigation indexes. For checked timing and cleanup evidence, use these manual files first:

- `GRAND_EXPERIMENT_ARCHAEOLOGY_20260605.md`: hand-checked local/remote grand synthesis, lineage, cleanup boundary, and remaining gaps.
- `LOCAL_REMOTE_ARCHAEOLOGY_CONCLUSIONS_CN_20260605.md`: readable Chinese conclusion report for local state, remote state, lineage, timing, cleanup boundary, and 8-hour continuation plan.
- `manual_top_level_directory_index_20260605.csv`: top-level directory-by-directory manual classification for local G:, remote `I:\Github\Latent_Style`, and remote `I:\Github\Latent_Style_TokenizerClean`.
- `manual_family_walkthrough_20260605.csv`: family-level walkthrough for local `SchrodingerBridge/exp`, local `Related_Works`, local `Cycle-NCE`, remote main experiment families, and remote TokenizerClean packets.
- `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv`: every current local `SchrodingerBridge/exp` top-level directory/file opened and classified with weight counts and cleanup decision.
- `manual_related_works_directory_ledger_20260605.csv`: local `Related_Works` top-level plus `baseline_pipeline/results`, `runs`, `run_511`, `repos`, and `final_works` directory ledger.
- `manual_cleanup_retention_and_next_candidates_20260605.csv`: remaining weight/cache classes, keep reasons, and next deletion candidates.
- `MANUAL_EXPERIMENT_AUDIT_20260605.md`: current hand-checked audit narrative and gap list.
- `manual_directory_audit_20260605.csv`: checked directory-level evidence rows.
- `manual_timing_evidence_20260605.csv`: checked training/inference timing rows with source paths.
- `manual_remote_phase_space_sweep_20260605.csv`: per-run remote phase-space sweep audit.
- `manual_remaining_weight_classes_20260605.csv`: post-cleanup remaining weight classes and keep reasons.
- `MANUAL_REMOTE_PHASE_SPACE_SWEEP_20260605.md`: detailed notes for the remote phase-space sweep cleanup.
- `cleanup/manual_deleted_checkpoints_20260605.csv`: local manual deletion ledger.
- `cleanup/manual_empty_directory_cleanup_20260605.csv`: local empty-directory cleanup ledger for non-evidence zero-file probe trees.
- `cleanup/remote_manual_deleted_checkpoints_20260605.csv`: remote manual deletion ledger.
