# Experiment Archaeology

This directory contains the curated local G and remote I experiment archaeology outputs.

## Main Outputs

- `AUTHORITATIVE_ARCHAEOLOGY_REPORT_ZH_20260605.md`: current readable Chinese authority report for local state, remote state, TokenizerClean, lineage, timing, cleanup totals, gaps, and the 8-hour continuation plan.
- `authoritative_report_evidence_map_20260605.csv`: conclusion-to-evidence map for the authority report.
- `MANUAL_REMOTE_TOKENIZERCLEAN_RETAINED_NO_SUMMARY_OWNER_REVIEW_20260605.md`: second-pass owner review for the 10 retained TokenizerClean no-summary directories.
- `manual_remote_tokenizerclean_retained_no_summary_owner_policy_20260605.csv`: policy table splitting 3 pure orphan probe dirs from 7 trained no-summary payload dirs.
- `cleanup/manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv`: deletion ledger for the 3 pure orphan probe dirs, totaling `170.017 MB`.
- `manual_remote_tokenizerclean_orphan_probe_post_delete_verify_20260605.csv`: post-delete verification for orphan probe cleanup and retained diagnostics/payload evidence.
- `manual_remote_tokenizerclean_remaining_weight_classes_after_owner_review_cleanup_20260605.csv`: current post-owner-review TokenizerClean remaining weight classes.
- `MANUAL_REMOTE_RAR_DEEP_PROVENANCE_20260605.md`: deep RAR provenance pass using temporary UnRAR listing and same-size expanded-file comparison.
- `manual_remote_rar_deep_provenance_policy_20260605.csv`: per-RAR keep/delete policy for `experiments.rar`, `Gate.rar`, `Attn_48.part*.rar`, `chess.part*.rar`, and `45.rar`.
- `cleanup/manual_remote_rar_weight_only_archive_cleanup_20260605.csv`: deletion ledger for weight-only RAR archives, totaling `6553.384 MB`.
- `manual_remote_rar_weight_only_archive_post_delete_verify_20260605.csv`: post-delete verification for RAR cleanup and retained expanded evidence.
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
- Remote deleted checkpoints: 405, MB=14535.700
- Remote deleted generated media: 43008, MB=11883.246
- Remote main data/cache/archive residue deleted: 11, MB=381.807
- Remote TokenizerClean checkpoint/probe files deleted across citation/no-summary/owner-review passes: 170, MB=5731.399
- Remote TokenizerClean remaining weight classes after owner-review cleanup: 29 dirs, 156 files, MB=5643.952
- Remote RAR weight-only archives deleted: 6 files, MB=6553.384

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

- `MASTER_ARCHAEOLOGY_CONCLUSIONS_AND_8H_PLAN_CN_20260605.md`: current readable master report for local state, remote main state, remote TokenizerClean state, experiment lineage, cleanup already done, remaining gaps, and the 8-hour execution plan.
- `ARCHAEOLOGY_EXECUTIVE_CONCLUSIONS_CN_20260605.md`: clean current entry report for local conclusions, remote conclusions, experiment lineage, timing state, cleanup principles, and remaining 8-hour-plan gaps.
- `manual_conclusion_index_20260605.csv`: one-row-per-area conclusion index for global/local/remote-main/remote-tokenizerclean/timing/lineage status.
- `manual_goal_completion_audit_20260605.csv`: requirement-by-requirement completion audit; records why the overall archaeology goal is still active.
- `manual_8h_execution_plan_20260605.csv`: concrete 8-hour block plan for continuing no-summary checkpoint review, generated-image policy, remote cache/archive policy, timing quality pass, and final consistency audit.
- `AUTHORITATIVE_ARCHAEOLOGY_SYNTHESIS_CN_20260605.md`: current authoritative Chinese synthesis for local state, remote state, experiment lineage, cleanup already performed, gaps, and the 8-hour continuation plan.
- `GRAND_EXPERIMENT_ARCHAEOLOGY_20260605.md`: hand-checked local/remote grand synthesis, lineage, cleanup boundary, and remaining gaps.
- `LOCAL_REMOTE_ARCHAEOLOGY_CONCLUSIONS_CN_20260605.md`: readable Chinese conclusion report for local state, remote state, lineage, timing, cleanup boundary, and 8-hour continuation plan.
- `CONSOLIDATED_EXPERIMENT_ARCHAEOLOGY_REPORT_CN_20260605.md`: consolidated Chinese executive report answering local state, remote state, experiment lineage, cleanup boundary, and 8-hour policy-driven cleanup plan.
- `manual_coverage_matrix_20260605.csv`: coverage matrix showing which local/remote roots were deeply opened, top-level counted, or only classified, with next action per root.
- `manual_top_level_directory_index_20260605.csv`: top-level directory-by-directory manual classification for local G:, remote `I:\Github\Latent_Style`, and remote `I:\Github\Latent_Style_TokenizerClean`.
- `manual_family_walkthrough_20260605.csv`: family-level walkthrough for local `SchrodingerBridge/exp`, local `Related_Works`, local `Cycle-NCE`, remote main experiment families, and remote TokenizerClean packets.
- `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv`: every current local `SchrodingerBridge/exp` top-level directory/file opened and classified with weight counts and cleanup decision.
- `manual_related_works_directory_ledger_20260605.csv`: local `Related_Works` top-level plus `baseline_pipeline/results`, `runs`, `run_511`, `repos`, and `final_works` directory ledger.
- `manual_cycle_nce_directory_ledger_20260605.csv`: local `Cycle-NCE` top-level and main family ledger, with checked summaries/logs/CSV evidence, timing fields, weight/cache classification, and cleanup decision.
- `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md`: narrative manual walkthrough for local `Cycle-NCE`, including timing evidence, metric anchors, cleanup boundary, and remaining gaps.
- `manual_local_eval_cache_policy_20260605.csv`: file-level local root `eval_cache` retention/cleanup policy for ArtFID, HF/ModelScope, manual CLIP, DINO offline pairing, reference features, VAE compile, and VAE ONNX caches.
- `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md`: narrative local `eval_cache` manual walkthrough; explains why these files are cache/dependency/speed artifacts rather than training checkpoints.
- `manual_local_root_misc_policy_20260605.csv`: local root misc/archive/tmp policy for `archive`, `tmp`, root `exp`, `final_works`, `seedream45_api`, root tracked files, and legacy code/data roots.
- `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md`: narrative local root misc walkthrough; records deletion of duplicate `Cycle-NCE.tar` and stale launcher residue, and why paper tmp/tex/pdf/png surfaces were retained.
- `manual_local_dataset_cache_policy_20260605.csv`: local dataset/latent/feature cache policy for `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale`, `horse2zebra`, and `wikiart_fewshot`.
- `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md`: narrative dataset/cache walkthrough; records tensor-shape checks and deletion of failed `wikiart_81k` HF cache residue.
- `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`: remote `I:\Github\Latent_Style\SchrodingerBridge\exp` top-level inventory with file/log/summary/weight counts and sample evidence paths.
- `MANUAL_REMOTE_SCHRODINGERBRIDGE_EXP_20260605.md`: manual remote `SchrodingerBridge/exp` walkthrough with opened README/config/log/summary evidence, timing examples, weight retention, and cleanup boundary.
- `manual_remote_samam_checkpoint_thinning_policy_20260605.csv`: remote SaMAM central `step_checkpoints` keep/delete policy for 19 checkpoint files.
- `manual_remote_samam_hash_pairs_20260605.csv`: SHA256 comparison of `last*.ckpt` aliases against corresponding `step-step=*.ckpt` files; all differed, so aliases were not deleted as duplicates.
- `manual_remote_samam_checkpoint_metadata_20260605.csv`: PyTorch metadata readout for the 19 remote SaMAM checkpoints.
- `manual_remote_samam_state_dict_hashes_20260605.csv`: state-dict SHA256 comparison showing `last*.ckpt` aliases are model-duplicates of paired step checkpoints.
- `cleanup/manual_remote_samam_alias_cleanup_20260605.csv`: deletion ledger for 7 redundant SaMAM `last*.ckpt` aliases, totaling `1931.291 MB`.
- `manual_remote_samam_remaining_step_checkpoints_after_alias_cleanup_20260605.csv`: post-delete verification list of 12 retained SaMAM step checkpoints, totaling `3310.776 MB`.
- `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md`: narrative remote SaMAM checkpoint thinning audit with opened `eval_curve`, `convergence_recovered.md`, ArtFID reuse, checkpoint metadata, state-dict hashes, and alias cleanup.
- `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv`: remote `SchrodingerBridge/exp` epoch-level evidence for the 101 pre-cleanup checkpoint files, with config/log/summary/timing/metric fields.
- `manual_remote_schrodingerbridge_epoch_thinning_policy_20260605.csv`: per-checkpoint keep/delete policy for remote `SchrodingerBridge/exp` after manual retained-epoch review.
- `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv`: deletion ledger for 84 remote `SchrodingerBridge/exp` checkpoint files, totaling `4961.604 MB`.
- `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv`: post-delete verification list of the 17 remaining remote `SchrodingerBridge/exp` checkpoints.
- `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md`: narrative manual epoch-thinning walkthrough for remote `SchrodingerBridge/exp`.
- `MANUAL_REMOTE_TOKENIZERCLEAN_CITATION_GRAPH_20260605.md`: narrative manual citation graph and cleanup audit for remote `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`.
- `manual_tokenizerclean_exp_citation_graph_20260605.csv`: pre-cleanup weighted-only TokenizerClean graph from the earlier pass; superseded for decisions by the all-directory graph below.
- `manual_remote_tokenizerclean_exp_internal_evidence_20260605.csv`: pre-cleanup internal evidence table for all 145 remote TokenizerClean `exp` top-level directories.
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`: all-directory docs/reviews/master/paper citation graph for remote TokenizerClean `exp`.
- `manual_remote_tokenizerclean_cleanup_policy_20260605.csv`: all-directory keep/delete policy used for TokenizerClean checkpoint cleanup.
- `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv`: per-file deletion ledger for 141 uncited TokenizerClean exploratory checkpoints, totaling `5198.991 MB`.
- `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_by_dir_20260605.csv`: per-directory summary of the same TokenizerClean checkpoint cleanup.
- `manual_remote_tokenizerclean_exp_internal_evidence_after_cleanup_20260605.csv`: post-delete verification table for all 145 remote TokenizerClean `exp` directories.
- `manual_remote_tokenizerclean_remaining_weight_classes_after_cleanup_20260605.csv`: remaining TokenizerClean checkpoint classes after deleting the uncited summary-backed exploratory checkpoints.
- `manual_remote_tokenizerclean_timing_evidence_20260605.csv`: TokenizerClean summary-level full-eval wall-time timing evidence extracted from post-cleanup `summary.json` files; training time is left blank unless explicitly recorded.
- `MANUAL_REMOTE_TOKENIZERCLEAN_NO_SUMMARY_REVIEW_20260605.md`: narrative review and cleanup audit for the 28 TokenizerClean no-summary checkpoint directories.
- `manual_remote_tokenizerclean_no_summary_review_20260605.csv`: evidence table for the 28 no-summary checkpoint dirs, including config/log/training CSV tails.
- `manual_remote_tokenizerclean_no_summary_cleanup_policy_20260605.csv`: keep/delete policy for the no-summary checkpoint dirs.
- `cleanup/manual_remote_tokenizerclean_no_summary_probe_checkpoint_cleanup_20260605.csv`: per-file deletion ledger for 18 no-summary probe/calibration checkpoints, totaling `362.391 MB`.
- `manual_remote_tokenizerclean_exp_internal_evidence_after_no_summary_cleanup_20260605.csv`: post-delete verification table after the no-summary probe cleanup.
- `manual_remote_tokenizerclean_remaining_weight_classes_after_no_summary_cleanup_20260605.csv`: latest TokenizerClean remaining checkpoint classes after both cleanup passes.
- `MANUAL_REMOTE_TOKENIZERCLEAN_GENERATED_MEDIA_PRUNE_20260605.md`: narrative audit for remote TokenizerClean generated media cleanup.
- `manual_remote_tokenizerclean_generated_media_inventory_20260605.csv`: pre-cleanup media inventory for all 145 TokenizerClean `exp` directories.
- `manual_remote_tokenizerclean_generated_media_cleanup_policy_20260605.csv`: keep/delete policy for generated media based on citation graph and structured evidence.
- `cleanup/manual_remote_tokenizerclean_uncited_generated_media_cleanup_20260605.csv`: per-file deletion ledger for 43008 uncited generated media files, totaling `11883.246 MB`.
- `manual_remote_tokenizerclean_generated_media_inventory_after_cleanup_20260605.csv`: post-delete verification media inventory.
- `manual_remote_tokenizerclean_remaining_media_classes_after_cleanup_20260605.csv`: remaining generated media classes after cleanup.
- `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md`: narrative manual audit for remote main data/cache/archive surfaces after opening each major root; records residue-only cleanup and retained archive gaps.
- `manual_remote_main_data_cache_archive_policy_20260605.csv`: per-scope keep/cleanup policy for remote main `data`, `style_data`, latent roots, `eval_cache`, `SchrodingerBridge/scale/datasets`, historical gates, `Cycle-NCE`, `experiments`, `StarGAN`, `seedream45_api`, and `Related_Works`.
- `manual_remote_main_data_cache_archive_delete_candidates_20260605.csv`: exact whitelist of 11 remote residue/empty-temp deletion targets.
- `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`: deletion ledger for the 11 remote main data/cache/archive residue targets, totaling `381.807 MB`.
- `manual_remote_main_data_cache_archive_post_delete_verify_20260605.csv`: post-delete verification for the same 11 targets; all `post_exists=False`.
- `MANUAL_REMOTE_ARCHIVE_PROVENANCE_20260605.md`: remote archive provenance audit for `eval_cache.zip`, `experiments.rar`, and `Cycle-NCE` archives; records duplicate/stale archive cleanup and retained RAR gaps.
- `manual_remote_archive_provenance_policy_20260605.csv`: per-archive keep/delete/provenance policy after hash and zip-entry checks.
- `cleanup/manual_remote_duplicate_archive_cleanup_20260605.csv`: deletion ledger for 3 duplicate/stale remote archives, totaling `3290.714 MB`.
- `manual_remote_archive_post_delete_verify_20260605.csv`: post-delete verification for deleted archives and retained evidence roots.
- `manual_local_remaining_surface_probe_20260605.csv`: navigation counts for the local remaining data/cache/dependency/archive surface; used only as a map, not as a deletion decision.
- `manual_local_remaining_surface_policy_20260605.csv`: path-by-path manual policy for 31 local remaining-surface rows, including duplicated archive proof and explicit keep decisions for WDS tar, checkpoint tar, dependency gzip, `.git` shallow lock, and `uv.lock` files.
- `MANUAL_LOCAL_REMAINING_SURFACE_POLICY_20260605.md`: narrative local remaining-surface walkthrough and delete whitelist.
- `delete_local_remaining_surface.ps1`: exact whitelist script for the local remaining-surface cleanup block.
- `cleanup/manual_local_remaining_surface_cleanup_20260605.csv`: deletion ledger for 5 local remaining-surface whitelist targets, totaling `237.860 MB`.
- `manual_local_remaining_surface_post_delete_verify_20260605.csv`: post-delete verification for the 5 deleted targets and retained evidence roots; all 15 checks passed.
- `manual_cleanup_retention_and_next_candidates_20260605.csv`: remaining weight/cache classes, keep reasons, and next deletion candidates.
- `MANUAL_EXPERIMENT_AUDIT_20260605.md`: current hand-checked audit narrative and gap list.
- `manual_directory_audit_20260605.csv`: checked directory-level evidence rows.
- `manual_timing_evidence_20260605.csv`: checked training/inference timing rows with source paths.
- `TIMING_EVIDENCE_QUALITY_PASS_20260605.md`: timing quality pass separating claim-candidate full-eval/train+eval rows from quick-eval, smoke, invalidated, anomalous, historical, and audit-only timing rows.
- `timing_quality_master_20260605.csv`: 1093-row timing quality overlay built from manual timing evidence and TokenizerClean summary timing; keeps original units and missing values.
- `timing_quality_summary_20260605.csv`: row counts by timing quality class and claim-use status.
- `manual_remote_phase_space_sweep_20260605.csv`: per-run remote phase-space sweep audit.
- `manual_remaining_weight_classes_20260605.csv`: post-cleanup remaining weight classes and keep reasons.
- `MANUAL_REMOTE_PHASE_SPACE_SWEEP_20260605.md`: detailed notes for the remote phase-space sweep cleanup.
- `cleanup/manual_deleted_checkpoints_20260605.csv`: local manual deletion ledger.
- `cleanup/manual_empty_directory_cleanup_20260605.csv`: local empty-directory cleanup ledger for non-evidence zero-file probe trees.
- `cleanup/manual_cache_cleanup_20260605.csv`: local cache cleanup ledger for invalid `.incomplete`/empty cache residue found during manual `eval_cache` audit.
- `cleanup/manual_root_misc_cleanup_20260605.csv`: local root misc cleanup ledger for duplicate archive tar and stale launcher residue.
- `cleanup/manual_dataset_cache_cleanup_20260605.csv`: local dataset/cache cleanup ledger for failed `SchrodingerBridge/scale/datasets/wikiart_81k` download residue.
- `cleanup/remote_manual_deleted_checkpoints_20260605.csv`: remote manual deletion ledger.
