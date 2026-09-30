# Local Experiment Inventory

Generated: 2026-06-03 17:41:36

Centralized outputs in this directory:

- `local_experiment_inventory_20260603.csv`
- `local_experiment_inventory_20260603.json`
- `local_prune_manifest_20260603.json`

## Scan roots

- `sb_exp`: `G:\GitHub\Latent_Style\SchrodingerBridge\exp`
- `sb_legacy_anchor`: `G:\GitHub\Latent_Style\SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0`
- `baseline_results`: `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results`

## Counts by root family

- `baseline_results`: 57
- `sb_exp`: 106
- `sb_legacy_anchor`: 1

## Counts by classification

- `active_paper_evidence`: 11
- `baseline_misc`: 11
- `baseline_paper_or_historical`: 12
- `external_baseline_or_probe`: 12
- `historical_anchor`: 1
- `local_exploratory_frozen`: 52
- `local_smoke`: 35
- `runtime_support`: 12
- `unclassified_review`: 18

## Paper-facing or review-worthy results

- `samam_wsl_mamba_b2_15ep_15000` | `baseline_results` | `legacy256_overfit50` | class=`baseline_paper_or_historical` | ckpt=16 | summary=15 | artfid=0 | best_style=0.6959188950061798 | best_lpips=0.43768593544
- `samam_wsl_mamba_512_scratch_clean_silent_b1_20k` | `baseline_results` | `wikiart512_5style` | class=`baseline_paper_or_historical` | ckpt=11 | summary=8 | artfid=2 | best_style=0.7912443998654684 | best_lpips=0.16433595481733335
- `S-add__K-1_C-0_W-20_Col-0` | `sb_legacy_anchor` | `legacy256_overfit50` | class=`historical_anchor` | ckpt=0 | summary=15 | artfid=0 | best_style=0.7218544493118922 | best_lpips=0.42220418536000004
- `timing_20260602` | `baseline_results` | `wikiart512_5style` | class=`baseline_paper_or_historical` | ckpt=0 | summary=6 | artfid=3 | best_style=0.7767400147120159 | best_lpips=0.6088616661733334
- `samam_wsl_mamba_256_formal_750_eval` | `baseline_results` | `legacy256_overfit50` | class=`baseline_paper_or_historical` | ckpt=0 | summary=6 | artfid=1 | best_style=0.6968672373692193 | best_lpips=0.41912689210666665
- `samst_distinct5_512_real_b2_e15_20260602` | `baseline_results` | `distinct5_512` | class=`baseline_paper_or_historical` | ckpt=5 | summary=1 | artfid=1 | best_style=0.7247245136102042 | best_lpips=0.6255497488
- `samam_distinct5_512_mamba_b1_20k_ckpt250_20260601_175723` | `baseline_results` | `distinct5_512` | class=`baseline_paper_or_historical` | ckpt=4 | summary=0 | artfid=0 | best_style=None | best_lpips=None
- `samst` | `baseline_results` | `legacy256_overfit50` | class=`baseline_paper_or_historical` | ckpt=0 | summary=2 | artfid=1 | best_style=0.7556747878922356 | best_lpips=0.41044156119999997
- `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote` | `sb_exp` | `distinct5_512` | class=`active_paper_evidence` | ckpt=0 | summary=2 | artfid=0 | best_style=0.6993825696706771 | best_lpips=0.32133288292
- `distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote` | `sb_exp` | `distinct5_512` | class=`active_paper_evidence` | ckpt=0 | summary=1 | artfid=0 | best_style=0.696914507508278 | best_lpips=0.31864464036
- `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote` | `sb_exp` | `distinct5_512` | class=`active_paper_evidence` | ckpt=0 | summary=1 | artfid=0 | best_style=0.7009949656327565 | best_lpips=0.36229389774666665
- `timing_20260601` | `baseline_results` | `unknown` | class=`baseline_paper_or_historical` | ckpt=0 | summary=1 | artfid=0 | best_style=None | best_lpips=None
- `weight_sweep_40` | `sb_exp` | `legacy256_overfit50` | class=`active_paper_evidence` | ckpt=0 | summary=320 | artfid=0 | best_style=0.7161262236833571 | best_lpips=0.37773701265333337
- `pareto_probe_4` | `sb_exp` | `legacy256_overfit50` | class=`unclassified_review` | ckpt=0 | summary=80 | artfid=0 | best_style=0.7048981204430262 | best_lpips=0.37545324995999996
- `review_additional_experiments` | `sb_exp` | `legacy256_overfit50` | class=`active_paper_evidence` | ckpt=0 | summary=78 | artfid=0 | best_style=0.7166270259221394 | best_lpips=0.35275963541333333
- `orth12` | `sb_exp` | `legacy256_overfit50` | class=`active_paper_evidence` | ckpt=0 | summary=30 | artfid=0 | best_style=0.7037303335666656 | best_lpips=0.40251263518666663
- `timing_20260602` | `sb_exp` | `unknown` | class=`active_paper_evidence` | ckpt=8 | summary=20 | artfid=0 | best_style=0.7739746244748433 | best_lpips=0.3940862842533334
- `legacy` | `sb_exp` | `legacy256_overfit50` | class=`active_paper_evidence` | ckpt=0 | summary=19 | artfid=0 | best_style=0.7176444892485937 | best_lpips=0.2977638070986666
- `timing_20260601` | `sb_exp` | `unknown` | class=`active_paper_evidence` | ckpt=0 | summary=15 | artfid=0 | best_style=None | best_lpips=None
- `ablation_destructive_7epoch` | `sb_exp` | `legacy256_overfit50` | class=`active_paper_evidence` | ckpt=0 | summary=12 | artfid=0 | best_style=0.7225049327214559 | best_lpips=0.29762808341999997

## Local prune boundary

- prune-eligible directories: 87 (only `local_smoke` / `local_exploratory_frozen`)
- deleted non-data artifacts: 0 files, 0.0 MB

## Notes

- This inventory keeps logs, configs, summaries, metrics, notes, and checkpoints.
- Pruning deletes only obvious non-data artifacts such as grids, images, and videos.
- Remote results are intentionally not included here; they belong in the separate remote inventory.
