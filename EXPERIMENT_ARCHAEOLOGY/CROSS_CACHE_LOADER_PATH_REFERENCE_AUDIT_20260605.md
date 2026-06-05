# Cross-Cache Loader/Path Reference Audit - 2026-06-05

This pass follows `CROSS_CACHE_DEDUP_AUDIT_20260605.md`. It is a manual path-consumer audit, not a hash scan. No cache files were deleted.

## Scope Opened

Local files and directories opened:

- `G:\GitHub\Latent_Style\eval_cache`
- `G:\GitHub\Latent_Style\SchrodingerBridge\eval_cache`
- `G:\GitHub\Latent_Style\Cycle-NCE\eval_cache`
- `Related_Works\baseline_pipeline\evaluation\run_sb_eval_v2.py`
- `Related_Works\baseline_pipeline\evaluation\run_sb_eval_all.py`
- `Related_Works\baseline_pipeline\evaluation\eval_batch.py`
- `Related_Works\baseline_pipeline\evaluation\eval_all_baselines.py`
- `Related_Works\baseline_pipeline\evaluation\eval_wikiart512_grid_outputs.py`
- `Related_Works\baseline_pipeline\scripts\eval_samam_checkpoint_curve.py`
- `SchrodingerBridge\src\utils\run_evaluation.py`
- `SchrodingerBridge\src\utils\artfid_metric.py`
- `SchrodingerBridge\tools\eval_wikiart512_latent.py`
- Representative `Cycle-NCE\*\config.json` and `SchrodingerBridge\configs/exp\*.json` path references.

Remote files and directories opened through:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

- `I:\Github\Latent_Style\eval_cache`
- `I:\Github\Latent_Style\SchrodingerBridge\eval_cache`
- `I:\Github\Latent_Style\Cycle-NCE\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\Cycle-NCE\eval_cache` (missing)
- `I:\Github\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\src\utils\run_evaluation.py`
- `I:\Github\Latent_Style\SchrodingerBridge\tools\eval_wikiart512_latent.py`
- `I:\Github\Latent_Style\SchrodingerBridge\configs\aaai2027\longer_train_k_seed42_b44_e8.json`
- `I:\Github\Latent_Style\SchrodingerBridge\configs\aaai2027\longer_train_f_seed42_b44_e8.json`
- `I:\Github\Latent_Style\Cycle-NCE\src\config.json`
- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\scripts\eval_samam_checkpoint_curve.py`
- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\scripts\run_samam_256_repro.py`

## Manual Findings

The hash audit found exact duplicate cache payloads, but this path audit shows the duplicate roots are still consumed by different launch paths.

Local findings:

- `SchrodingerBridge/src/utils/run_evaluation.py` defaults to `../eval_cache`, stores reference feature files as `cache_dir/ref_feats_*.pt`, and passes `cache_dir` into the ArtFID checkpoint loader.
- The same evaluator intentionally keeps compatibility with old `Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32` layouts.
- `artfid_metric.py` reads or creates `cache_dir/artfid/art_inception.pth`; the root and SchrodingerBridge `art_inception.pth` files are SHA256-identical but are tied to different `cache_dir` launches.
- `Cycle-NCE` configs still encode `./eval_cache` or `../eval_cache` for full-eval cache and CLIP HF cache.
- Local Related_Works evaluation wrappers hard-code `Cycle-NCE/eval_cache/manual_clip/...`; local `Cycle-NCE/eval_cache` currently contains ref feature caches but no manual CLIP directory, so these wrappers are also evidence of path drift that needs migration, not a safe deletion signal.

Remote findings:

- Remote main has all three cache roots present: root, `SchrodingerBridge`, and `Cycle-NCE`.
- Remote SchrodingerBridge current AAAI2027 configs point to `/mnt/i/Github/Latent_Style/eval_cache`.
- Remote `SchrodingerBridge/tools/eval_wikiart512_latent.py` defaults to root `I:/Github/Latent_Style/eval_cache/hf` and root `I:/Github/Latent_Style/eval_cache/manual_clip/...`.
- Remote `Cycle-NCE/src/config.json` points to `./eval_cache`, keeping a separate Cycle-NCE cache-local contract.
- Remote Related_Works has no `baseline_pipeline/evaluation` directory, but its `scripts` directory contains two cache consumers: `eval_samam_checkpoint_curve.py` defaults to Cycle-NCE manual CLIP, while `run_samam_256_repro.py` passes root manual CLIP to SchrodingerBridge eval.
- Remote TokenizerClean has root and SchrodingerBridge eval_cache roots, no Cycle-NCE eval_cache root, and its evaluator still carries root/Cycle-NCE/manual-clip compatibility logic.

## Decision

No duplicate cache deletion is whitelisted.

Reasons:

- Exact SHA256 equality only proves file payload equality; it does not prove every loader uses the same path.
- Multiple local and remote consumers still encode different cache roots.
- Some same-name/same-size remote cache groups were already proven hash-mismatched in `cross_cache_remote_duplicate_groups_20260605.csv`; name/size-based cleanup is unsafe.
- Deleting ref feature caches would usually be recoverable by recomputation, but it would lose immediate offline reproducibility and can change timing/cost behavior.
- Deleting ArtFID or manual CLIP copies can break offline runs for the cache root supplied by that launch/config.

## Required Before Any Future Cache Dedup Cleanup

Future cache dedup cleanup needs a separate migration proof:

1. Pick one canonical cache root for local and one for remote.
2. Update or document every consumer path that currently points to `Cycle-NCE/eval_cache`, `SchrodingerBridge/eval_cache`, or root `eval_cache`.
3. Use documented symlinks/junctions if old paths must remain valid.
4. Run an offline eval smoke for each migrated consumer class.
5. Only then create a per-file deletion whitelist, ledger, and post-delete verification.

Current action remains `retain_all`.

Detailed row-level evidence is in `cross_cache_loader_path_reference_audit_20260605.csv`.
