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
