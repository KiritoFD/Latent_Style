# ??????

???????????????????????????????????????? checkpoint ???

## ????

| scope | action | cleanup_class | files | size_mb |
| --- | --- | --- | --- | --- |
| local_G | deleted | likely_non_mainline_delete_candidate | 329 | 11575.67 |
| local_G | skipped | likely_mainline_keep | 8 | 260.552 |
| local_G | skipped | review_delete_candidate | 38018 | 15983.362 |
| remote_I | deleted | non_mainline_delete_candidate | 246 | 9074.318 |
| remote_I | skipped | likely_mainline_keep | 151 | 15714.315 |
| remote_I | skipped | review_delete_candidate | 142454 | 138112.399 |

## ??????

| path | action | files | checkpoint_like_files | size_mb | reason |
| --- | --- | --- | --- | --- | --- |
| G:\GitHub\Latent_Style\Related_Works\Related_Works | deleted | 0 | 0 | 0 | safe_local_cleanup_empty_or_temp_or_malformed_nonmainline_path |
| G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\no_op_identity_5x5 | deleted | 0 | 0 | 0 | safe_local_cleanup_empty_or_temp_or_malformed_nonmainline_path |
| G:\GitHub\Latent_Style\_codex_tmp | deleted | 10 | 0 | 0.013 | safe_local_cleanup_empty_or_temp_or_malformed_nonmainline_path |
| G:\GitHub\Latent_Style\ | deleted | 16 | 5 | 17.842 | safe_local_cleanup_empty_or_temp_or_malformed_nonmainline_path |
| G:\GitHub\Latent_Style\ | deleted | 0 | 0 | 0 | safe_local_cleanup_empty_or_temp_or_malformed_nonmainline_path |

## ????

- `skipped/review_delete_candidate`?????????????????
- latent/image ???????? checkpoint ????????
- `SchrodingerBridge/configs/archive/20260605_local_distinct5_ema/`?? JSON ????????????????????
- ?? tex/pdf ??? modified ???????/???????????

## 2026-06-05 manual eval_cache cache cleanup

Additional manual cleanup after file-level inspection of `G:\GitHub\Latent_Style\eval_cache`:

| scope | action | cleanup_class | files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | invalid_hf_incomplete_blob | 1 | 55.994 | `manual_cache_cleanup_20260605.csv` |
| local_G | deleted | empty_modelscope_temp_dirs | 0 | 0 | `manual_cache_cleanup_20260605.csv` |

All valid ArtFID, CLIP, VAE, DINO/offline-pairing, ref feature, VAE compile, and VAE ONNX cache artifacts were retained. This cleanup was not checkpoint thinning; it removed only failed/empty cache residue.

## 2026-06-05 manual root misc cleanup

Additional manual cleanup after root-level archive/tmp/exp inspection:

| scope | action | cleanup_class | files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | duplicate_cycle_nce_archive_tar | 1 | 1503.203 | `manual_root_misc_cleanup_20260605.csv` |
| local_G | deleted | stale_failed_launcher_residue | 7 | 0.000316 | `manual_root_misc_cleanup_20260605.csv` |
| local_G | deleted | empty_probe_directory | 0 | 0 | `manual_root_misc_cleanup_20260605.csv` |

The current `Cycle-NCE` evidence tree was retained; only the duplicate monolithic archive tar under `archive/2026-05-19_cleanup/root` was deleted. Recent paper/PDF/TEX/PNG scratch under `tmp` was retained.

## 2026-06-05 manual dataset/cache cleanup

Additional manual cleanup after dataset/latent/feature cache inspection:

| scope | action | cleanup_class | files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | failed_hf_dataset_download_cache | 6 | 63.948 | `manual_dataset_cache_cleanup_20260605.csv` |

Representative `.pt` files in the remaining dataset/cache roots were loaded read-only and confirmed as VAE latent tensors or CLIP feature tensors, not training checkpoints. Valid dataset/cache roots were retained.

## 2026-06-05 remote main data/cache/archive residue cleanup

Additional remote cleanup after manual path-by-path inspection of `I:\Github\Latent_Style` data/cache/archive surfaces:

| scope | action | cleanup_class | files_or_dirs | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | deleted | failed_hf_incomplete_blobs | 5 | 381.807 | `manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |
| remote_I_main | deleted | stale_cache_locks | 3 | 0 | `manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |
| remote_I_main | deleted | stale_ref_feats_tmp | 1 | 0.000122 | `manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |
| remote_I_main | deleted | recursively_empty_temp_dirs | 2 | 0 | `manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |

All 11 whitelist targets were post-delete verified with `post_exists=False`. Valid data roots, latent roots, complete eval caches, SchrodingerBridge historical gate outputs, baseline repos, and large archives without proven duplicate/provenance status were retained.

## 2026-06-05 remote duplicate/stale archive cleanup

Additional remote cleanup after archive provenance checks of `I:\Github\Latent_Style`:

| scope | action | cleanup_class | files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | deleted | stale_eval_cache_zip | 1 | 704.467 | `manual_remote_duplicate_archive_cleanup_20260605.csv` |
| remote_I_main | deleted | legacy_checkpoint_archive_zip | 1 | 2078.795 | `manual_remote_duplicate_archive_cleanup_20260605.csv` |
| remote_I_main | deleted | exact_duplicate_archive | 1 | 507.452 | `manual_remote_duplicate_archive_cleanup_20260605.csv` |

Post-delete verification confirms the three deleted archives are absent and the retained evidence roots still exist: `eval_cache`, `experiments\1-decoder-patch5-15`, root `Cycle-NCE\45.rar`, and `Cycle-NCE\src`.

## 2026-06-05 local remaining surface cleanup

Additional local cleanup after path-by-path inspection of the remaining data/cache/dependency/archive surface:

| scope | action | cleanup_class | targets | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | empty cache/probe directories | 2 | 0 | `manual_local_remaining_surface_cleanup_20260605.csv` |
| local_G | deleted | fully duplicated zip archives | 2 | 205.674 | `manual_local_remaining_surface_cleanup_20260605.csv` |
| local_G | deleted | fully duplicated output tar archive | 1 | 32.186 | `manual_local_remaining_surface_cleanup_20260605.csv` |

Total released in this block: `237.860 MB`. The policy explicitly retained WebDataset tar shards, `samst_ckpts_epoch50.tar`, CLIP vocabulary gzip, external repo `.git\shallow.lock`, and `uv.lock` dependency files because they were not proven disposable. Post-delete verification passed all 15 absent/present checks in `manual_local_remaining_surface_post_delete_verify_20260605.csv`.

## 2026-06-05 remote TokenizerClean retained no-summary orphan cleanup

Second-pass owner review of the 10 retained TokenizerClean no-summary directories split them into 3 pure orphan probe directories and 7 trained payload directories. Only the 3 pure orphan probe directories were deleted.

| scope | action | cleanup_class | files_or_dirs | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_tokenizerclean | deleted | orphan probe weight files | 11 | 170.017 | `manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv` |
| remote_tokenizerclean | deleted | empty orphan probe dirs | 3 | 0 | `manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv` |

Post-delete verification passed all 11 checks in `manual_remote_tokenizerclean_orphan_probe_post_delete_verify_20260605.csv`: the 3 orphan dirs are absent, diagnostics outputs remain, and representative trained no-summary payload dirs remain. The latest TokenizerClean remaining-weight table is `manual_remote_tokenizerclean_remaining_weight_classes_after_owner_review_cleanup_20260605.csv` with 29 directories, 156 weight-like files, and `5643.952 MB`.

## 2026-06-05 remote RAR weight-only archive cleanup

Deep RAR provenance used a temporary remote copy of local `UnRAR.exe` to list archive contents and compare file entries against expanded directories by same relative path and exact byte size.

| scope | action | cleanup_class | archive_files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | deleted | weight-only RAR archive | 1 | 3384.032 | `manual_remote_rar_weight_only_archive_cleanup_20260605.csv` |
| remote_I_main | deleted | weight-only multipart RAR archive | 3 | 1975.113 | `manual_remote_rar_weight_only_archive_cleanup_20260605.csv` |
| remote_I_main | deleted | weight-only multipart RAR archive | 2 | 1194.239 | `manual_remote_rar_weight_only_archive_cleanup_20260605.csv` |

Total released in this block: `6553.384 MB`. Deleted archives were `Cycle-NCE\Gate.rar`, `Cycle-NCE\Attn_48.part1.rar`, `part2.rar`, `part3.rar`, `Cycle-NCE\chess.part1.rar`, and `part2.rar`. Post-delete verification passed all 11 checks in `manual_remote_rar_weight_only_archive_post_delete_verify_20260605.csv`: the archives are absent, expanded `Gate`, `Attn_48`, and `chess` evidence directories remain, and at that point `experiments.rar` plus `Cycle-NCE\45.rar` were still retained pending follow-up.

## 2026-06-05 remote experiments.rar resolved duplicate cleanup

Follow-up manual audit opened the 9 `experiments.rar` cache mismatches one by one. All 9 were HuggingFace CLIP snapshot `SymbolicLink` entries whose `..\..\blobs\...` targets still exist and match the RAR entry byte sizes.

| scope | action | cleanup_class | archive_files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | deleted | resolved duplicate RAR archive | 1 | 8091.026 | `manual_remote_experiments_rar_resolved_duplicate_cleanup_20260605.csv` |

Post-delete verification passed all 11 checks in `manual_remote_experiments_rar_resolved_duplicate_post_delete_verify_20260605.csv`: `experiments.rar` is absent, expanded `experiments` remains, and all 9 CLIP snapshot symlink target blobs remain same-size as the original RAR entries.

## 2026-06-05 remote Cycle-NCE 45.rar retained review

`Cycle-NCE\45.rar` was opened with temporary remote `UnRAR.exe` after the duplicate archive and RAR cleanup passes. No deletion was performed.

| scope | action | cleanup_class | archive_files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | retained | unique historical RAR archive | 1 | 507.452 | `manual_remote_cycle_nce_45_rar_policy_20260605.csv` |

Reason for retention: the archive has no expanded `Cycle-NCE\45` directory and contains unique nonweight evidence: 4 configs, 8 summaries, 8 metrics CSVs, 6008 generated/eval images, root ma-probe artifacts, and 12 old weight files. Deleting the whole archive would delete more than checkpoints. A future cleanup can extract a curated nonweight evidence package and then delete the archive under a new whitelist policy.

## 2026-06-05 local generated-media intermediate-frame cleanup

Manual owner-level review opened the largest local generated-media candidates. Formal eval directories, paper-facing bundles, dataset mirrors, and baseline protocol outputs were retained. Only five CUT video work directories were deleted because each contained only `_work` intermediate PNG frames, no mp4/json/csv evidence, no text references to the timestamp, and final video evidence plus `summary.json` are retained under `Cycle-NCE\video`.

| scope | action | cleanup_class | dirs | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | unreferenced intermediate video frames | 5 | 3068.463 | `manual_local_generated_media_intermediate_frame_cleanup_20260605.csv` |

Post-delete verification passed all 5 checks in `manual_local_generated_media_intermediate_frame_post_delete_verify_20260605.csv`: the deleted `Related_Works\runs\cut_5x5\video\head_20260404_*` work-frame directories are absent.

## 2026-06-05 local generated-media pass4 duplicate cleanup

Fourth-pass local generated-media owner review reopened the next exact candidate
cluster: Distinct5 v350 full-eval packet, SaMST WikiArt512 external-eval
packets, highres local qualitative outputs, and Seedream diagnostic inputs.
Formal metric/timing packets, qualitative paired evidence, non-identical
variants, and diagnostic inputs were retained. Only two exact duplicate targets
were deleted.

| scope | action | cleanup_class | files_or_dirs | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| local_G | deleted | duplicate image archive | 1 | 41.253 | `manual_local_generated_media_pass4_cleanup_20260605.csv` |
| local_G | deleted | duplicate highres image directory | 1 dir / 750 files | 60.660 | `manual_local_generated_media_pass4_cleanup_20260605.csv` |

Total released in this block: `101.913 MB`. Post-delete verification passed all
11 checks in `manual_local_generated_media_pass4_post_delete_verify_20260605.csv`:
the duplicate e15 zip and standalone highres `samst_same_test` directory are
absent, while retained e15 summary/metrics/grid/images, paired highres
SaMST/LBM images, non-identical highres v2, and Seedream diagnostic input dirs
remain present.

## 2026-06-05 remote Cycle-NCE 45.rar original archive cleanup

After the curated nonweight extraction pass verified all `6084` nonweight
payload entries by relative path and byte size, and recorded/removes the `12`
staged `.pt` files from the curated package, the original compressed archive was
deleted as an exact whitelist target.

| scope | action | cleanup_class | archive_files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_I_main | deleted | curated_nonweight_preserved_original_archive_with_old_weights | 1 | 507.452 | `manual_remote_cycle_nce_45_rar_delete_execution_20260605.csv` |

Post-delete verification passed all 6 checks in
`manual_remote_cycle_nce_45_rar_post_delete_verify_20260605.csv`: original
`I:\Github\Latent_Style\Cycle-NCE\45.rar` is absent, the curated nonweight
package is present with `6086` files / `145.512 MB`, the extracted `45\`
payload dir is present, package manifest and removed-weight ledger are present,
and recursive weight-extension count remains `0`.

## 2026-06-05 remote TokenizerClean training-log-only weight cleanup

Follow-up fixed-path review reopened the 7 trained no-summary payload
directories. Five had config/training CSV evidence only and no external
downstream or diagnostic evidence, so only their exact checkpoint weights were
deleted. Directories, `config.json`, `logs\training_*.csv`, source snapshots,
and numeric debug logs were retained as archaeology metadata.

| scope | action | cleanup_class | checkpoint_files | size_mb | ledger |
| --- | --- | --- | ---: | ---: | --- |
| remote_tokenizerclean | deleted | training-log-only no-summary checkpoint weights | 7 | 248.429 | `manual_remote_tokenizerclean_training_log_only_weight_delete_execution_20260605.csv` |

Post-delete verification passed all 20 checks in
`manual_remote_tokenizerclean_training_log_only_weight_post_delete_verify_20260605.csv`:
the 7 deleted checkpoints are absent, all 5 parent configs are present, all 5
parent training CSV sets are present, and the 3 external-evidence checkpoints
remain. A fixed-path live recheck is recorded in
`manual_remote_tokenizerclean_training_log_only_live_recheck_20260605.csv`.
