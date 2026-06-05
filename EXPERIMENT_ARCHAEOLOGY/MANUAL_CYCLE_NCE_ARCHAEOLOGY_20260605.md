# Manual Cycle-NCE Archaeology - 2026-06-05

Scope: local `G:\GitHub\Latent_Style\Cycle-NCE`.

This pass was done as a manual directory walkthrough, not as a one-shot scan. I opened the top-level files, the main family directories, representative configs, full-eval summaries, training logs, aggregate CSVs, cache folders, video outputs, and prior archaeology notes. The detailed row ledger is:

- `EXPERIMENT_ARCHAEOLOGY/manual_cycle_nce_directory_ledger_20260605.csv`

## What This Directory Is

`Cycle-NCE` is a historical Latent AdaCUT / Cycle-NCE experiment archive. Its own reports describe the project as a Jan 13 to Apr 2 exploration covering DiT, CNN/AdaGN, SWD, color, identity, NCE removal, architecture ablations, micro-batch training, and few-shot ukiyo-e follow-up.

The current local directory is evidence-heavy:

- `500` `summary.json` files.
- `496` `metrics.csv` files.
- `260` `training_*.csv` logs.
- `3` weight-like `.pt` files, all under `eval_cache`, and all are reference feature caches rather than training checkpoints.

The important correction is that `Cycle-NCE` is not a checkpoint cleanup target in its current local state. It is mostly summaries, logs, source snapshots, configs, visualization outputs, and CSV indexes.

## Manually Opened Evidence

Primary orientation docs:

- `Cycle-NCE/ARCHAEOLOGY_FINAL_CN.md`
- `Cycle-NCE/History_Report.md`
- `Cycle-NCE/ARCHEOLOGY_PLAN.md`
- `Cycle-NCE/vram.md`
- `Cycle-NCE/config.json`

Aggregate metric CSVs opened:

- `Cycle-NCE/exp.csv`
- `Cycle-NCE/freq.csv`
- `Cycle-NCE/Aline120.csv`
- `Cycle-NCE/hf.csv`
- `Cycle-NCE/cgw.csv`
- `Cycle-NCE/micro.csv`
- `Cycle-NCE/clocor1_full_eval_summary.csv`
- `Cycle-NCE/grid_search_3epoch_scatter.csv`

Representative training logs opened:

- `Cycle-NCE/46/46_00_holy_grail/logs/training_20260406_013242.csv`
- `Cycle-NCE/46_09_real_holy_grail/logs/training_20260406_102104.csv`
- `Cycle-NCE/46_splash/logs/training_20260406_185348.csv`
- `Cycle-NCE/Ablate43/Ablate43_S01_Baseline_Gold/logs/training_20260402_221747.csv`
- `Cycle-NCE/Aline120/Aline120_aline_01_oracle/logs/training_20260404_011547.csv`
- `Cycle-NCE/arch/arch_ablate_A1_swin_h2_g1_d2/logs/training_20260328_213955.csv`
- `Cycle-NCE/freq/freq_01_conservative_baseline/logs/training_20260405_075401.csv`
- `Cycle-NCE/tmp_batch96_smoke/logs/training_20260321_182949.csv`
- `Cycle-NCE/tmp_grad_probe_bs96_e2/logs/training_20260325_110449.csv`
- `Cycle-NCE/weight_exp4_latent_adain_swd60_tv00_id40_r16_e60/logs/training_20260322_044742.csv`

Other opened evidence:

- `Cycle-NCE/fewshot_ukiyoe_runs/fewshot_ukiyo_e_sid5/meta.json`
- `Cycle-NCE/fewshot_ukiyoe_runs/fewshot_ukiyo_e_sid5/full_eval_lpips_clip_style/summary.json`
- `Cycle-NCE/video/summary.json`
- `Cycle-NCE/artifacts/eval_classifier/*.report.json`
- `Cycle-NCE/summary/summary_aggregate.csv`

## Timing Findings

The strongest timing evidence in `Cycle-NCE` is the per-epoch training log format. The logs include:

- `data_time_sec`
- `transfer_time_sec`
- `fwd_loss_time_sec`
- `backward_time_sec`
- `optimizer_time_sec`
- `step_overhead_time_sec`
- `compute_time_sec`
- `epoch_time_sec`
- `samples_seen`
- `samples_per_sec`
- `compute_samples_per_sec`

Examples confirmed by direct opening:

- `46_09_real_holy_grail`: after warmup, epochs around `30s` for `10240` samples, with `samples_per_sec` around `341-342`.
- `46_splash`: after warmup, epochs around `24.5s` for `10240` samples, with `samples_per_sec` around `418-420`.
- `tmp_batch96_smoke`: one opened epoch has `epoch_time_sec=158.76438736915588`.
- `tmp_grad_probe_bs96_e2`: two opened epochs have `epoch_time_sec=443.1042754650116` and `437.7721347808838`.

Family-level summed `epoch_time_sec` values in the ledger are not claimed as wall-clock run duration. They are sums of rows currently present in local `training_*.csv` files and are useful only as local evidence inventory.

## Metric Anchors

Opened aggregate CSVs gave these navigation anchors:

- `exp.csv`: best checked row `style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10`, epoch `100`, `transfer_clip_style=0.729723026394844`.
- `Aline120.csv`: best checked row `Aline120_aline_03_ghost_wireframe`, epoch `20`, `transfer_clip_style=0.7146436547239621`.
- `hf.csv`: best checked row `p_base_hf_3p0_distill_epochs200_tokenized`, epoch `60`, `transfer_clip_style=0.6734027210871381`.
- `freq.csv`: best checked row `freq_04_no_idt_abyss`, epoch `80`, `transfer_clip_style=0.6453720057010651`.
- `summary/summary_aggregate.csv`: example strong row `ablate_M1-Aggressive-Fine`, latest epoch `120`, `latest_transfer_clip_style=0.7169979644318423`.

These are index anchors. They do not replace per-run summary inspection when a paper or table needs a final cited value.

## Cleanup Decision

Safe cleanup performed in this pass:

- Deleted `Cycle-NCE/eval_cache/hf`.
- Reason: recursive file count `0`, recursive dir count `0`, git-tracked file count `0`.
- Bytes reclaimed: `0`.
- Logged in `EXPERIMENT_ARCHAEOLOGY/cleanup/manual_empty_directory_cleanup_20260605.csv`.

Retained intentionally:

- All `full_eval` summaries and metrics.
- All training logs.
- All aggregate CSVs.
- All source snapshots and history configs.
- Video outputs and visual summaries.
- The three `eval_cache/ref_feats_*.pt` files, because they are feature caches and only about `1.106 MB` total.

No local Cycle-NCE training checkpoint deletion was possible because no local training checkpoints were found in the current directory state.

## Gaps

- Many old summaries have metrics but no explicit inference wall time.
- Some aggregate CSV `source_file` paths point to remote `I:\Github\Latent_Style\Cycle-NCE`, so local evidence is a curated copy, not always the original run location.
- Few-shot ukiyo-e meta references base/patched `.pt` checkpoint paths, but those payloads are not present locally.
- Duplicate-looking `freq_01..freq_08` top-level directories were retained because the duplicate relationship to `Cycle-NCE/freq/*` was not proven enough to delete safely.
- RAR/archive policy is still unresolved for broader historical Cycle-NCE archives if they exist outside this local directory block.
