# Experiment Archaeology

This directory contains the curated local G and remote I experiment archaeology outputs.

## Main Outputs

- `AUTHORITATIVE_LOCAL_REMOTE_LINEAGE_CONCLUSIONS_20260605.md`: newest clean entrypoint for local conclusions, remote-main conclusions, TokenizerClean conclusions, dataset lineage, timing state, cleanup ledger policy, and the 8-hour continuation plan.
- `authoritative_local_remote_lineage_conclusions_20260605.csv`: row-level conclusion index behind the newest clean entrypoint.
- `GRAND_LOCAL_REMOTE_ARCHAEOLOGY_SYNTHESIS_ZH_20260605.md`: direct Chinese synthesis for local state, remote main, remote TokenizerClean, experiment lineage, timing, cleanup principles, and the remaining 8-hour plan.
- `grand_local_remote_archaeology_synthesis_index_20260605.csv`: machine-readable index behind the grand synthesis.
- `CURRENT_CLEAR_ARCHAEOLOGY_CONCLUSIONS_ZH_20260605.md`: clear Chinese current-state report for local G, remote main I, remote TokenizerClean, dataset lineage, timing, cleanup policy, and the next 8-hour execution plan; added because older readable dataset conclusion text has encoding damage.
- `current_clear_archaeology_conclusion_index_20260605.csv`: machine-readable index behind the clear Chinese current-state report.
- `ARCHAEOLOGY_DIRECT_CONCLUSIONS_ZH_20260605.md`: direct current answer for local state, remote main state, remote TokenizerClean state, lineage, timing, cleanup ledger totals, gaps, and the 8-hour continuation plan.
- `archaeology_direct_conclusions_index_20260605.csv`: compact machine-readable index for the direct conclusions report.
- `ARCHAEOLOGY_CURRENT_STATUS_AND_CONCLUSIONS_ZH_20260605.md`: current readable status report separating local, remote main, remote TokenizerClean, timing, lineage, cleanup, remaining gaps, and the 8-hour continuation plan.
- `archaeology_current_status_requirements_20260605.csv`: machine-readable requirement/status/gap index for the current status report.
- `CROSS_CACHE_DEDUP_AUDIT_20260605.md`: local/remote cache duplicate hash audit; no deletion performed, exact duplicates retained pending loader/path-reference audit.
- `CROSS_CACHE_LOADER_PATH_REFERENCE_AUDIT_20260605.md`: manual loader/path-reference audit for local and remote duplicate cache roots; no deletion whitelisted because multiple consumers still encode root, SchrodingerBridge, and Cycle-NCE cache paths.
- `cross_cache_loader_path_reference_audit_20260605.csv`: row-level source-open evidence for the cache path audit.
- `TIMING_CANDIDATE_MISSING_DOCS_SOURCE_OPEN_20260605.md`: manual source-open pass for the 26 claim-candidate timing rows missing from the docs timing master.
- `timing_candidate_missing_docs_source_open_20260605.csv`: row-level source-open table for those 26 timing candidates, including train/infer units, exact sources, verification notes, and promotion decisions.
- `AUTHORITATIVE_ARCHAEOLOGY_REPORT_ZH_20260605.md`: current readable Chinese authority report for local state, remote state, TokenizerClean, lineage, timing, cleanup totals, gaps, and the 8-hour continuation plan.
- `authoritative_report_evidence_map_20260605.csv`: conclusion-to-evidence map for the authority report.
- `MANUAL_REMOTE_TOKENIZERCLEAN_RETAINED_NO_SUMMARY_OWNER_REVIEW_20260605.md`: second-pass owner review for the 10 retained TokenizerClean no-summary directories.
- `manual_remote_tokenizerclean_retained_no_summary_owner_policy_20260605.csv`: policy table splitting 3 pure orphan probe dirs from 7 trained no-summary payload dirs.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_THIRD_PASS_20260605.md`: third-pass current-state review for the 7 retained trained no-summary payload dirs.
- `manual_remote_tokenizerclean_trained_no_summary_third_pass_20260605.csv`: current config/training/weight/summary absence check for those 7 dirs.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_OWNER_DECISION_20260605.md`: owner-decision table for the 7 retained trained no-summary payload dirs; no deletion whitelisted.
- `manual_remote_tokenizerclean_trained_no_summary_owner_decision_20260605.csv`: row-level keep/recover decision table for those 7 dirs.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_DEEP_OPEN_20260605.md`: one-by-one remote deep-open pass for the same 7 trained no-summary dirs; records the 5 training-log-only payloads, 2 external-evidence payloads, and 1 config lineage anomaly.
- `manual_remote_tokenizerclean_trained_no_summary_deep_open_20260605.csv`: row-level manual-open evidence table with config/data/resume/training/time/weight/eval/decision fields.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINING_LOG_ONLY_WEIGHT_DELETE_PLAN_20260605.md`: exact-path checkpoint-only delete plan for five training-log-only no-summary payload directories.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINING_LOG_ONLY_WEIGHT_DELETE_EXECUTED_20260605.md`: execution summary for deleting 7 exact TokenizerClean training-log-only checkpoint weights while retaining metadata.
- `manual_remote_tokenizerclean_training_log_only_weight_delete_whitelist_20260605.csv`: row-level whitelist for those 7 checkpoint files.
- `manual_remote_tokenizerclean_training_log_only_weight_delete_execution_20260605.csv`: row-level deletion ledger, totaling `248.429 MB`.
- `manual_remote_tokenizerclean_training_log_only_weight_post_delete_verify_20260605.csv`: post-delete verification for absent checkpoints, retained metadata, and retained evidence-bearing weights.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINING_LOG_ONLY_LIVE_RECHECK_20260605.md`: fixed-path live recheck of all 7 trained no-summary payload directories after deletion.
- `manual_remote_tokenizerclean_training_log_only_live_recheck_20260605.csv`: row-level live recheck with current weight counts, config resume fields, training CSV tails, and summary/full_eval absence.
- `manual_remote_tokenizerclean_training_log_only_remaining_weights_20260605.csv`: current post-cleanup remaining weight table for the 7 trained no-summary payload dirs.
- `MANUAL_REMOTE_TOKENIZERCLEAN_MISSING_RESUME_ANOMALY_20260605.md`: fixed-path annotation for `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`, whose config points to absent `epoch_0004.pt`.
- `manual_remote_tokenizerclean_missing_resume_anomaly_20260605.csv`: row-level anomaly record; marks the directory as metadata-only, not clean lineage.
- `cleanup/manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv`: deletion ledger for the 3 pure orphan probe dirs, totaling `170.017 MB`.
- `manual_remote_tokenizerclean_orphan_probe_post_delete_verify_20260605.csv`: post-delete verification for orphan probe cleanup and retained diagnostics/payload evidence.
- `manual_remote_tokenizerclean_remaining_weight_classes_after_owner_review_cleanup_20260605.csv`: current post-owner-review TokenizerClean remaining weight classes.
- `MANUAL_REMOTE_RAR_DEEP_PROVENANCE_20260605.md`: deep RAR provenance pass using temporary UnRAR listing and same-size expanded-file comparison.
- `manual_remote_rar_deep_provenance_policy_20260605.csv`: per-RAR keep/delete policy for `experiments.rar`, `Gate.rar`, `Attn_48.part*.rar`, `chess.part*.rar`, and `45.rar`.
- `cleanup/manual_remote_rar_weight_only_archive_cleanup_20260605.csv`: deletion ledger for weight-only RAR archives, totaling `6553.384 MB`.
- `manual_remote_rar_weight_only_archive_post_delete_verify_20260605.csv`: post-delete verification for RAR cleanup and retained expanded evidence.
- `MANUAL_REMOTE_EXPERIMENTS_RAR_RESOLVED_POLICY_20260605.md`: follow-up manual proof that `experiments.rar` cache mismatches are HF snapshot symlink targets with matching blob payloads.
- `manual_remote_experiments_rar_symlink_targets_20260605.csv`: 9-row fixed-target symlink target audit for `experiments.rar`.
- `cleanup/manual_remote_experiments_rar_resolved_duplicate_cleanup_20260605.csv`: deletion ledger for resolved duplicate `experiments.rar`, totaling `8091.026 MB`.
- `manual_remote_experiments_rar_resolved_duplicate_post_delete_verify_20260605.csv`: post-delete verification for `experiments.rar`, expanded `experiments`, and all 9 symlink targets.
- `../EXPERIMENT_ARCHAEOLOGY_MASTER.csv`: final root-level master CSV.
- `final_master_experiments.csv`: same final master CSV inside this directory.
- `final_by_dataset/*.csv`: one CSV per dataset/setting family.
- `final_timeline.csv`: chronological experiment event index.
- `EXPERIMENT_TIMELINE.md`: narrative timeline and experiment lineage.
- `remote_i_curated/`: remote-side curated outputs generated after filtering and checkpoint cleanup.
- `cleanup/local_deleted_checkpoints.csv`: local per-file deletion audit.
- `remote_i_curated/remote_i_deleted_checkpoints.csv`: remote per-file deletion audit.
- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`: timing-focused subset.
- `timing_candidate_missing_docs_source_open_20260605.csv`: 26/26 source-opened candidates that were missing from the docs timing master; docs master itself was not edited in this pass.

`final_by_dataset/` is the authoritative per-dataset/per-setting split after merging and reclassifying both G: and I: evidence. `remote_i_curated/` is kept as the remote-side audit bundle generated on I: before final local reclassification.

The latest direct status should be read from `AUTHORITATIVE_LOCAL_REMOTE_LINEAGE_CONCLUSIONS_20260605.md` first. Some older count lines below predate the later manual generated-media, RAR, TokenizerClean owner-review, timing reconciliation, source-open, and local/remote lineage synthesis passes; use the per-pass ledgers and `cleanup/CLEANUP_AUDIT_SUMMARY.md` for cleanup totals.

## Counts

- Final experiment rows: 22629
- Timing rows: mixed historical count; current timing counts are docs timing master `419`, archaeology timing overlay `1093`, and missing-docs source-opened candidates `26`.
- Timeline events: 7829
- Source roots: {'G:/GitHub/Latent_Style': 17134, 'I:\\': 5495}
- Local deleted checkpoints: 329, MB=11575.67
- Remote deleted checkpoints: 405, MB=14535.700
- Remote deleted generated media: 43008, MB=11883.246
- Remote main data/cache/archive residue deleted: 11, MB=381.807
- Remote TokenizerClean checkpoint/probe files deleted across citation/no-summary/owner-review/training-log-only passes: 177, MB=5979.828
- Remote TokenizerClean trained no-summary payload weights after training-log-only cleanup: 7 dirs, 3 files, MB=130.883
- Remote RAR weight-only archives deleted: 6 files, MB=6553.384
- Remote experiments.rar resolved duplicate deleted: 1 file, MB=8091.026
- Remote Cycle-NCE 45.rar original archive deleted after curated nonweight extraction: 1 file, MB=507.452

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
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_20260605.md`: first owner-level local generated-media review; separates dataset mirrors, paper bundles, formal evals, diagnostics, inference sweeps, and frame-only delete whitelist.
- `manual_local_generated_media_owner_review_20260605.csv`: row-level evidence/decision table for the first local generated-media owner review.
- `manual_local_generated_media_intermediate_frame_post_delete_verify_20260605.csv`: post-delete verification for the five frame-only local video work directories.
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_PASS2_20260605.md`: second local generated-media owner review covering protocol smoke/eval dirs, curve evals, aggregate baseline image dirs, and Seedream protocol output.
- `manual_local_generated_media_owner_review_pass2_20260605.csv`: row-level evidence/decision table for local generated-media pass 2.
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_PASS3_20260605.md`: third local generated-media owner review covering no-op/IDT docs controls, timing benchmark outputs, Distinct5 compact anchors, and local ckptsync generation-only evidence.
- `manual_local_generated_media_owner_review_pass3_20260605.csv`: row-level evidence/decision table for local generated-media pass 3; all rows are retained and no cleanup was performed in this pass.
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_PASS4_20260605.md`: fourth local generated-media owner review covering v350 full-eval, SaMST external-eval, highres paired/variant outputs, and Seedream diagnostic inputs.
- `manual_local_generated_media_owner_review_pass4_20260605.csv`: row-level evidence/decision table for local generated-media pass 4.
- `cleanup/manual_local_generated_media_pass4_cleanup_20260605.csv`: deletion ledger for two exact duplicate media targets, totaling `101.913 MB`.
- `manual_local_generated_media_pass4_post_delete_verify_20260605.csv`: post-delete verification for pass4 duplicate media cleanup.
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_REVIEW_PASS5_20260605.md`: fifth local generated-media owner review; manually reopens `seedream_gap`, `inference_param_sweep_t01e8_quick`, `inference_param_sweep_t01e8_fine`, and CUT raw web outputs, with no cleanup whitelisted.
- `manual_local_generated_media_owner_review_pass5_20260605.csv`: row-level evidence/decision table for local generated-media pass 5, including git tracked/ignored state for each reviewed path.
- `MANUAL_LOCAL_GENERATED_MEDIA_OWNER_DECISION_MANIFEST_20260605.md`: owner-decision manifest for `seedream_gap` and the quick/fine inference parameter sweeps; includes visual sample confirmation and no-delete boundary.
- `manual_local_generated_media_owner_decision_manifest_20260605.csv`: row-level owner-decision packet for 7 Seedream input sets, 14 quick sweep points, and 8 fine sweep points.
- `MANUAL_LOCAL_ARCHIVE_TMP_PAPER_SCRATCH_PROVENANCE_20260605.md`: exact-path provenance pass for local `archive`, `tmp`, paper snapshot, config archive, and active paper workspace; no deletion whitelisted.
- `manual_local_archive_tmp_paper_scratch_provenance_20260605.csv`: row-level archive/tmp/paper scratch provenance table.
- `MANUAL_LOCAL_CUT_RAW_TRACKED_FILE_POLICY_20260605.md`: fixed-path CUT `raw_results` / `raw_results_val` tracked-file policy; no deletion whitelisted; training-log timing and missing inference-time fields recorded.
- `manual_local_cut_raw_tracked_file_policy_20260605.csv`: row-level CUT raw tracked-boundary policy with method, dataset, resolution, timing, source path, and no-delete decision fields.
- `manual_local_cut_raw_timing_required_fields_20260605.csv`: CUT timing rows with the required timing columns (`method`, `dataset_or_setting`, `resolution`, train/infer time, `params_m`, `source_path`, `note`); missing inference time is blank.
- `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`: remote `I:\Github\Latent_Style\SchrodingerBridge\exp` top-level inventory with file/log/summary/weight counts and sample evidence paths.
- `MANUAL_REMOTE_SCHRODINGERBRIDGE_EXP_20260605.md`: manual remote `SchrodingerBridge/exp` walkthrough with opened README/config/log/summary evidence, timing examples, weight retention, and cleanup boundary.
- `MANUAL_REMOTE_MAIN_SURFACE_RECHECK_20260605.md`: fixed-path live recheck for remote main `I:\Github\Latent_Style`, covering `SchrodingerBridge`, `Related_Works`, `Cycle-NCE`, review RAR, CUT media, and SaMAM baseline results; no deletion whitelisted.
- `manual_remote_main_surface_recheck_20260605.csv`: row-level remote main recheck table with decision, no-delete boundary, and next proof required per path.
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
- `MANUAL_REMOTE_TOKENIZERCLEAN_CITED_CURRENT_MEDIA_POLICY_20260605.md`: source-open keep/archive policy for the 26 retained cited/current TokenizerClean media directories.
- `manual_remote_tokenizerclean_cited_current_media_archive_policy_20260605.csv`: row-level keep/no-delete policy for those cited/current media directories.
- `MANUAL_REMOTE_TOKENIZERCLEAN_CITED_CURRENT_MEDIA_MANIFEST_20260605.md`: fixed-path live manifest for the same 26 cited/current media directories; maps each directory to representative summaries, CSVs, grids, generated media buckets, and checkpoints.
- `manual_remote_tokenizerclean_cited_current_media_manifest_20260605.csv`: row-level cited/current media manifest; no deletion whitelisted.
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_DEEP_OPEN_20260605.md`: manual one-by-one reopen of the 7 retained trained no-summary payloads.
- `manual_remote_tokenizerclean_trained_no_summary_deep_open_20260605.csv`: exact CSV evidence for those 7 payloads, including the missing `epoch_0004.pt` resume anomaly in `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`.
- `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md`: narrative manual audit for remote main data/cache/archive surfaces after opening each major root; records residue-only cleanup and retained archive gaps.
- `manual_remote_main_data_cache_archive_policy_20260605.csv`: per-scope keep/cleanup policy for remote main `data`, `style_data`, latent roots, `eval_cache`, `SchrodingerBridge/scale/datasets`, historical gates, `Cycle-NCE`, `experiments`, `StarGAN`, `seedream45_api`, and `Related_Works`.
- `manual_remote_main_data_cache_archive_delete_candidates_20260605.csv`: exact whitelist of 11 remote residue/empty-temp deletion targets.
- `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`: deletion ledger for the 11 remote main data/cache/archive residue targets, totaling `381.807 MB`.
- `manual_remote_main_data_cache_archive_post_delete_verify_20260605.csv`: post-delete verification for the same 11 targets; all `post_exists=False`.
- `MANUAL_REMOTE_ARCHIVE_PROVENANCE_20260605.md`: remote archive provenance audit for `eval_cache.zip`, `experiments.rar`, and `Cycle-NCE` archives; records duplicate/stale archive cleanup and later RAR follow-up status.
- `manual_remote_archive_provenance_policy_20260605.csv`: per-archive keep/delete/provenance policy after hash and zip-entry checks.
- `cleanup/manual_remote_duplicate_archive_cleanup_20260605.csv`: deletion ledger for 3 duplicate/stale remote archives, totaling `3290.714 MB`.
- `manual_remote_archive_post_delete_verify_20260605.csv`: post-delete verification for deleted archives and retained evidence roots.
- `MANUAL_REMOTE_EXPERIMENTS_RAR_RESOLVED_POLICY_20260605.md`: follow-up proof that `experiments.rar` is a resolved duplicate after opening all 9 HF snapshot symlink mismatches.
- `manual_remote_experiments_rar_cache_mismatch_20260605.csv`: fixed-target audit of the 9 known `experiments.rar` CLIP cache mismatch rows.
- `manual_remote_experiments_rar_symlink_targets_20260605.csv`: symlink target audit proving all 9 blob targets exist and match RAR entry sizes.
- `manual_remote_experiments_rar_resolved_policy_20260605.csv`: delete whitelist policy for `experiments.rar`.
- `delete_remote_experiments_rar_resolved_duplicate.ps1`: exact-path deletion script for the resolved duplicate archive.
- `cleanup/manual_remote_experiments_rar_resolved_duplicate_cleanup_20260605.csv`: deletion ledger for `experiments.rar`, totaling `8091.026 MB`.
- `manual_remote_experiments_rar_resolved_duplicate_post_delete_verify_20260605.csv`: post-delete verification for archive absence, expanded evidence presence, and 9 symlink target checks.
- `MANUAL_REMOTE_CYCLE_NCE_45_RAR_REVIEW_20260605.md`: manual review of retained `Cycle-NCE\45.rar`; records why it is unique nonweight evidence and not a delete target yet.
- `manual_remote_cycle_nce_45_rar_policy_20260605.csv`: keep policy for `45.rar`.
- `MANUAL_REMOTE_CYCLE_NCE_45_RAR_CURATED_EXTRACTION_POLICY_20260605.md`: curated extraction policy for `45.rar`; no deletion performed.
- `manual_remote_cycle_nce_45_rar_curated_extraction_policy_20260605.csv`: entry-class extraction/delete policy for `45.rar`.
- `manual_remote_cycle_nce_45_rar_run_ledger_20260605.csv`: per-run file/weight/image/config/summary/metrics counts inside `45.rar`.
- `manual_remote_cycle_nce_45_rar_summary_overview_20260605.csv`: 8-row overview extracted from the archive's full-eval summaries.
- `MANUAL_REMOTE_CYCLE_NCE_45_RAR_CURATED_EXTRACTION_EXECUTED_20260605.md`: executed remote curated nonweight extraction for `45.rar`; original archive was retained at that point and later deleted by the exact whitelist below.
- `MANUAL_REMOTE_CYCLE_NCE_45_RAR_DELETE_EXECUTED_20260605.md`: exact-path remote deletion of original `Cycle-NCE\45.rar` after curated nonweight extraction; released `507.452 MB`.
- `manual_remote_cycle_nce_45_rar_delete_whitelist_20260605.csv`: delete whitelist for the original `45.rar` archive.
- `manual_remote_cycle_nce_45_rar_delete_execution_20260605.csv`: execution ledger for deleting original `45.rar`.
- `manual_remote_cycle_nce_45_rar_post_delete_verify_20260605.csv`: post-delete verification that `45.rar` is absent and the curated nonweight package remains present with `0` weight-extension files.
- `manual_remote_cycle_nce_45_rar_curated_extraction_execution_20260605.csv`: one-row execution summary for the remote curated package.
- `manual_remote_cycle_nce_45_rar_curated_extraction_manifest_20260605.csv`: pulled remote manifest for 6084 extracted nonweight payload files.
- `manual_remote_cycle_nce_45_rar_curated_extraction_verify_20260605.csv`: path/byte verification against the original archive entry-class table.
- `manual_remote_cycle_nce_45_rar_curated_extraction_class_counts_20260605.csv`: class-level verification counts for the extracted nonweight package.
- `manual_remote_cycle_nce_45_rar_curated_extraction_removed_weights_20260605.csv`: 12 staged `.pt` files removed from the curated nonweight package.
- `README_COUNT_CONSISTENCY_AUDIT_20260605.md`: current README/count consistency audit after the latest source-open and policy blocks.
- `readme_count_consistency_audit_20260605.csv`: row-level count checks for current high-signal archaeology outputs.
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
