# Remote Main Data/Cache/Archive Manual Policy - 2026-06-05

Remote root:

`I:\Github\Latent_Style`

This pass is the manual follow-up to the shallow open table. It does not treat a script scan as a conclusion. Each major remote-main surface was opened by path, then only proven failed residue or empty temp directories were removed.

## What Was Checked

Checked and retained as data/latent backends:

- `data`: opened immediate dirs `cezanne`, `latents`, `vangogh`; sample files are small `.pt` latents; no `.incomplete`/temp markers.
- `style_data`: opened `latents`, `overfit_eval`, `overfit50`, `test`, `train`, plus `data.py`; no bad markers.
- `latents`, `latents_overfit50`: opened style/content subdirs and sample `.pt` files; no bad markers.
- `latent-256`, `latent-256-flux1`, `latent-256-flux2`, `latent-256-kl-f4`, `latent-256-kl-f4-mode`, `latent-256-sd15-ema`, `latent-256-sdxl`, `latent-256-sdxl-fp32`: opened subdirs and manifests; no bad markers. These are reusable latent caches, not failed experiments.

Checked and partially cleaned:

- `eval_cache`: opened immediate cache dirs and largest files. Full manual CLIP weights exist under `manual_clip/openai-clip-vit-base-patch32`; the HF `blobs` path contained one failed `.incomplete`. A shallow scan missed `ref_feats_1558c2de70_m80.pt.tmp.575`; manual listing showed the complete `ref_feats_1558c2de70_m80.pt` exists beside it, so the 128-byte temp residue was removed.
- `SchrodingerBridge\scale\datasets`: opened all dataset split dirs. Dataset bodies are retained; only `wikiart_81k` failed HF download residue and stale lock were removed.
- `Cycle-NCE`: opened root archives, duplicate-looking archive names, failed ArtFID/CLIP download parents, and complete dependency locations. Failed `.incomplete` files and stale `.lock` files were removed. Large archives were not deleted because provenance/duplicate safety is not proven.
- `experiments`: opened immediate experiment families and largest cache files. `uv.lock` and a torch compile hash containing `TMP` were false positives and retained. The ModelScope `._____temp` directory was recursively opened, found to contain zero files, and removed.
- `Related_Works\repos`: opened baseline repos and dependency weights. `S2WAT-main\pre_trained_models\tmp_timing` had `child_count=0` and was removed.

Checked and retained:

- `SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0`: contains `epoch_0001.pt` to `epoch_0008.pt`, `full_eval`, `full_eval_timing_epoch7`, logs and sweeps. This is formal gate evidence.
- `SchrodingerBridge\review_additional_experiments`: contains `lambda_grid`, `step_count_sweep`, `manifest.json`; retained as review sweep evidence.
- `Related_Works\baseline_pipeline\results`: contains SaMAM/Flux/ZImage/SaMST result dirs. No bad markers in this pass; remaining SaMAM step checkpoints are curated after alias cleanup.
- `StarGAN`: opened repo/run structure and summary grids; no bad markers.
- `seedream45_api`: opened `protocol_a_800` generated jpg outputs; no bad markers.

## Deleted Items

Deletion policy CSV:

`manual_remote_main_data_cache_archive_delete_candidates_20260605.csv`

Per-item deletion ledger:

`cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`

Post-delete verification:

`manual_remote_main_data_cache_archive_post_delete_verify_20260605.csv`

Result:

- Deleted 11 exact targets.
- Freed `381.807389 MB`.
- Post-delete verification: `post_exists=False` for all 11 targets.
- Parent directories still exist for all 11 targets, confirming this was residue-only cleanup.

Deleted classes:

- Failed HuggingFace `.incomplete` blobs: CLIP, WikiArt81k, ArtFID.
- Stale zero-byte `.lock` files in failed download parents.
- One stale 128-byte `ref_feats` temp file with a complete sibling `.pt`.
- Two recursively empty temp directories.

## Not Deleted

Large but retained pending owner/archive policy:

- `Cycle-NCE\Gate.rar` - `3384.032 MB`
- `Cycle-NCE\1-decoder-patch5-15_eAzEC.zip` - `2078.795 MB`
- `Cycle-NCE\Attn_48.part1.rar`, `part2.rar`, `part3.rar`
- `Cycle-NCE\chess.part1.rar`, `part2.rar`
- `Cycle-NCE\45.rar` and `Cycle-NCE\src\45.rar`
- `experiments.rar` at remote root
- complete CLIP/VAE/ArtFID/manual eval caches
- all dataset and latent roots

Reason: these are large recovery opportunities, but this pass did not prove they are safe duplicates or non-mainline disposable archives. They are indexed for follow-up, not removed.

## Files Produced By This Block

- `inspect_remote_main_data_cache_archive.ps1`: earlier full recursive inspection attempt, retained as evidence of the timeout boundary.
- `inspect_remote_main_data_cache_archive_shallow.ps1`: shallow open helper.
- `manual_remote_main_data_cache_archive_shallow_open_20260605.csv`: shallow open table.
- `manual_remote_main_data_cache_archive_policy_20260605.csv`: per-scope manual policy table.
- `manual_remote_main_data_cache_archive_delete_candidates_20260605.csv`: exact delete candidate whitelist.
- `delete_remote_main_data_cache_archive_residue.ps1`: whitelist deletion script with root path checks.
- `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`: deletion ledger.
- `manual_remote_main_data_cache_archive_post_delete_verify_20260605.csv`: remote post-delete verification.

## Remaining Gaps

- Cycle-NCE archive ownership is still unresolved. It needs a separate duplicate/provenance audit before deleting root `.rar/.zip/.7z` payloads.
- Remote root `experiments.rar` was later classified by deep RAR listing plus 9-row symlink-target audit and deleted as a resolved duplicate archive; see `MANUAL_REMOTE_EXPERIMENTS_RAR_RESOLVED_POLICY_20260605.md`.
- Complete model/eval caches may be deduplicable across local/remote surfaces, but this requires a cross-cache hash audit, not a cleanup-by-name pass.
