# Manual Check Session - 2026-06-05

This file records the current manual check pass after the user correction that a script scan is not enough. Scans and ledgers are navigation aids only. A conclusion in this session is only accepted when tied to a path opened directly or to a previously committed policy/cleanup ledger that was opened again in this session.

## Current Position

- No file was deleted in this session.
- No paper, tex, pdf, png, source, or Related_Works dirty file was modified.
- The current write scope is only `EXPERIMENT_ARCHAEOLOGY`.
- The local and remote trees are not globally complete. The remaining gaps are owner-level generated-media review, cross-cache dedup hash audit, docs timing master reconciliation, TokenizerClean cited/current media migration policy, and 7 TokenizerClean no-summary payload owner or summary-recovery decisions.

## Direct Local Checks

1. `G:\GitHub\Latent_Style` was opened at top level. It is a multi-surface repository root containing `SchrodingerBridge`, `Cycle-NCE`, `Related_Works`, `Dataset`, `eval_cache`, latent/feature caches, root `exp`, archive/tmp, and archaeology outputs. It is not a single clean experiment directory.
2. `G:\GitHub\Latent_Style\SchrodingerBridge` was opened. It contains dirty paper/submission areas, docs, exp, eval cache, configs, datasets, source, tools, and the historical `S-add__K-1_C-0_W-20_Col-0` baseline. These areas are evidence surfaces, not deletion targets by extension.
3. `SchrodingerBridge\exp\local_wsl_wikiart512_hist_b32_e8` was opened. It contains `config.json`, `logs/training_20260601_203435.csv`, `train_stdout.log`, and `epoch_0001.pt` through `epoch_0008.pt`. `train_stdout.log` reports `Model params: 3,914,997` and `TRAIN_WALL_SECONDS=53.19`.
4. `SchrodingerBridge\exp\local_wsl_wikiart512_hist_b32_e8\full_eval_epoch_0008_b2_opt_nocls` was opened. `summary.json` has `count=750`, `clip_style=0.7922978092034658`, `content_lpips=0.3550378331343333`, and `timings_sec.wall_total=206.792325715`; this is a formal timing/eval anchor.
5. `SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0\full_eval\epoch_0008` was opened. It contains `summary.json` and `metrics.csv`, and is historical baseline evidence.
6. `Cycle-NCE` was opened. It contains many named historical experiment families, archaeology reports, aggregate CSVs, docs, eval cache, experiment directories, source snapshots, summaries, and media. It must stay under explicit per-family policy.
7. `Related_Works\baseline_pipeline\results` was opened. It contains baseline and output families including agnes, seedream, cut, s2wat, samam, samst, sdedit, and Distinct5 timing/calibration runs. These are not deletion targets without an owner/policy row.
8. `manual_local_remaining_surface_policy_20260605.csv` and `manual_local_remaining_surface_post_delete_verify_20260605.csv` were opened. They confirm previous local cleanup was exact whitelist only; retained surfaces include datasets, WDS shards, dependency gzip/locks, checkpoint tar, and eval dependencies.

## Direct Remote Checks

Remote entrypoint: `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`.

1. `I:\` was opened via PowerShell `-EncodedCommand`. It is not empty. It contains `Github`, dataset/latent/cache roots, curated archaeology directories, wikiart latent/image roots, and unrelated system/helper files. Broad root cleanup is not valid.
2. `I:\Github` was opened. It contains `Latent_Style` and `Latent_Style_TokenizerClean`.
3. The 7 retained TokenizerClean no-summary payload directories were opened one by one:
   - `tokenizer_t01_carrier_base_b160`
   - `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`
   - `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56`
   - `wikiart512_ema_pair_budget_tokonly_e1_b80`
   - `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
   - `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48`
   - `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48`
4. Each of those 7 directories exists and has `config.json`, a `logs/training_*.csv`, `src`, `numeric_debug.jsonl`, and at least one checkpoint. Each has `summary_like_count=0`. The correct decision is retain pending summary recovery or owner decision, not delete as junk.
5. Remote RAR status was directly checked: `I:\Github\Latent_Style\experiments.rar` is absent; `I:\Github\Latent_Style\experiments` is present; `I:\Github\Latent_Style\Cycle-NCE\45.rar` is present at 507.452MB; removed weight-only RARs such as `Gate.rar`, `Attn_48.part1.rar`, and `chess.part1.rar` remain absent.

## Non-Negotiable Policy From This Check

- Do not delete by `.pt`, `.ckpt`, `.tar`, `.zip`, `.rar`, image extension, or size.
- Do not treat a generated directory as removable until it has an owner/paper/citation/current-run classification.
- Do not treat no-summary as failed. Some no-summary payloads have real configs, training CSVs, source snapshots, and checkpoints.
- Deletion requires a whitelist policy CSV/MD, per-file cleanup ledger, and post-delete verification.

## Next Manual Blocks

1. Local nested generated-image owner review.
2. TokenizerClean cited/current media archive or migration policy.
3. Summary recovery or owner decision for the 7 no-summary TokenizerClean payloads.
4. Cross-cache dedup hash audit across local and remote cache roots.
5. Reconcile `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` with `EXPERIMENT_ARCHAEOLOGY/timing_quality_master_20260605.csv`.
