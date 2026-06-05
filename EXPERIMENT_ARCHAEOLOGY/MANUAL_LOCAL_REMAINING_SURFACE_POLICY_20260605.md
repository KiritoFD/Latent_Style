# Manual Local Remaining Surface Policy - 2026-06-05

This pass corrects the earlier broad-scan risk: the probe CSV is treated only as navigation. Each cleanup decision below is based on an exact path check, archive listing, same-size duplicate comparison, or lock/dependency inspection.

## Delete whitelist

Only five local targets are approved for deletion in this pass:

- `eval_cache\vae_onnx\ema_b2_64\trt_cache`: exact path exists and recursive child count is `0`.
- `SchrodingerBridge\exp\frontier\decision_tree_clip_style\s21_temp_var0p0_temp0p03`: exact path exists and recursive child count is `0`.
- `SchrodingerBridge\datasets\horse2zebra\raw\horse2zebra.zip`: zip opened; `2661` file entries; every entry has an extracted same-size copy; archive size `111.454 MB`.
- `Related_Works\runs\cut_5x5\cut.zip`: zip opened; `2520` file entries; every entry has an extracted same-size copy; archive size `94.220 MB`.
- `exp\highres_eval_local\samst_outputs_epoch50.tar`: tar listed; `750` file entries; every file has an extracted same-size copy under `exp\highres_eval_local\samst`; archive size `32.186 MB`.

## Explicit keep decisions

The following archive-like or lock-like files are intentionally not deleted:

- `Related_Works\runs\lbm_train_wds_smoke_photo_to_monet\train-000000.tar`: WebDataset shard; `80` entries; no same-path extracted equivalent was found.
- `Related_Works\runs\lbm_train_wds_smoke_photo_to_monet\val-000000.tar`: WebDataset shard; `20` entries; no same-path extracted equivalent was found.
- `exp\highres_eval_local\samst_ckpts_epoch50.tar`: checkpoint archive; `5` model entries; no extracted same-path checkpoint copy was found.
- `Related_Works\repos\ArtBank\clip\bpe_simple_vocab_16e6.txt.gz`: CLIP vocabulary dependency, not experiment output clutter.
- `Related_Works\repos\AdaIN-style-official\.git\shallow.lock`: inside an external repo `.git`; active git/GitHubDesktop processes were observed; not part of experiment payload cleanup.
- `Cycle-NCE\uv.lock` and `Cycle-NCE\src\uv.lock`: dependency lock files; first lines identify uv resolver metadata, not stale temp locks.

## Retained roots

The large roots `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `eval_cache`, `SchrodingerBridge\scale`, `SchrodingerBridge\datasets\horse2zebra`, `SchrodingerBridge\exp`, `Related_Works`, `Cycle-NCE`, `archive`, `tmp`, `seedream45_api`, root `exp`, `efficiency`, `fast_infer_ablate43`, `latent_cyclegan`, `o20_d3`, and `wikiart_fewshot` remain retained. They are data, latent/feature caches, metric/model dependencies, experiment evidence roots, paper/review scratch surfaces, or legacy code/docs. None is deleted by name or size.

## Files produced by this block

- `manual_local_remaining_surface_probe_20260605.csv`: navigation surface counts.
- `manual_local_remaining_surface_policy_20260605.csv`: manual path-by-path decision table.
- `delete_local_remaining_surface.ps1`: exact whitelist deletion script.
- `cleanup/manual_local_remaining_surface_cleanup_20260605.csv`: per-target deletion ledger written by the script.
- `manual_local_remaining_surface_post_delete_verify_20260605.csv`: post-delete and retained-evidence verification written by the script.

## Remaining gaps

This pass does not claim the whole repository has every nested generated image directory owner-reviewed. Remaining gaps are still: TokenizerClean no-summary owner review, cited/current media archive migration, RAR provenance, cross-cache dedup, and final docs timing master reconciliation.
