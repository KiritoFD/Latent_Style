# SaMST Distinct5 E5 Rerun

Date: 2026-06-03

## Purpose

The retained local Distinct5 SaMST evidence only preserved the `e15` endpoint:

- result root:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602`
- retained checkpoints:
  - `Early_Renaissance\epoch_15.model`
  - `Impressionism\epoch_15.model`
  - `Minimalism\epoch_15.model`
  - `Rococo\epoch_15.model`
  - `Ukiyo_e\epoch_15.model`

No Distinct5-local `e5` or `e10` checkpoints were retained, so the convergence
claim for SaMST cannot currently be shown from existing artifacts alone.

This rerun reconstructs the first midpoint needed for the paper-facing
comparison:

- same dataset split
- same five styles
- same local Windows training path
- same image/style sizes
- lower batch only when required for memory stability

## Existing retained reference

Reference `e15` training run:

- output:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602`
- launch settings from `run.log`:
  - `data_root=F:\wikiart_distinct5_samam_512_classview_real`
  - `styles=[Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e]`
  - `epochs=15`
  - `batch_size=2`
  - `max_train_per_class=0`
- retained eval:
  - `eval_epoch15\epoch_0015\summary.json`
  - `eval_epoch15\epoch_0015\aggregate_targetwise_artfid.json`

## New rerun

Primary rerun attempt:

- output:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e5_20260603`
- status:
  - aborted as unstable due user-reported memory pressure

Stabilized rerun:

- output:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603`
- launcher:
  - `py -3 Related_Works\baseline_pipeline\scripts\run_samst_distinct5_local.py`
- arguments:
  - `--data-root F:\wikiart_distinct5_samam_512_classview_real`
  - `--styles Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e`
  - `--epochs 5`
  - `--batch-size 1`
  - `--image-size 256`
  - `--style-size 512`
  - `--max-train-per-class 0`
- run log:
  - `run.log`
- per-style logs:
  - `logs\train_<style>.log`

## Runtime status at launch audit

Observed after the stabilized `b1` launch:

- `run.log` confirms:
  - `started=2026-06-03T17:57:31.111486`
  - `batch_size=1`
- GPU state after launch:
  - approximately `7.35 GB / 8.19 GB`
  - sustained nonzero utilization and power draw
- note:
  - SaMST training logs appear line-buffered; lack of rapid console growth should
    not be mistaken for immediate failure while GPU activity remains live

## Completed result

The stabilized `b1` e5 run completed successfully:

- `started=2026-06-03T17:57:31`
- `finished=2026-06-03T19:53:29`
- wall time:
  - about `1.93h`
- all five target trainings returned `rc=0`

The e5 eval bundle is available at:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\summary.json`

The retained e15 reference is:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\summary.json`
- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\aggregate_targetwise_artfid.json`

Transfer-only comparison:

| metric | e5 | e15 | e15 - e5 |
| --- | ---: | ---: | ---: |
| CLIP-S | 0.698919 | 0.695741 | -0.003178 |
| LPIPS | 0.633500 | 0.631950 | -0.001550 |
| targetwise ArtFID | 465.686 | 444.487 | -21.199 |

All-pairs comparison:

| metric | e5 | e15 | e15 - e5 |
| --- | ---: | ---: | ---: |
| CLIP-S | 0.727581 | 0.724725 | -0.002857 |
| LPIPS | 0.627069 | 0.625550 | -0.001520 |

Interpretation:

- SaMST's reported CLIP-S / LPIPS quality is effectively at plateau by e5 on Distinct5-512.
- e15 is still the safer endpoint for manuscript tables because targetwise ArtFID is modestly lower.
- The supported claim is therefore: e15 is a conservative saturated endpoint, while e5 already shows that the CLIP-S / LPIPS curve has essentially converged.
- The e5 eval summary contains ArtFID fields, but the standalone `aggregate_targetwise_artfid.json` was not emitted under the e5 folder. Do not use the e5 ArtFID row in paper tables until that standalone artifact is regenerated or the table explicitly cites the embedded summary values.

## Next gate

1. regenerate or materialize the e5 standalone
   `aggregate_targetwise_artfid.json` if the paper needs an auditable ArtFID
   midpoint row;
2. add the e5 point to the convergence plot as a SaMST midpoint;
3. keep e15 as the manuscript-safe SaMST endpoint unless a later checkpoint
   improves targetwise ArtFID materially.

For the post-training bundle, use:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\run_samst_distinct5_eval_bundle.py`
- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\compare_samst_distinct5_epochs.py`
- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\watch_samst_distinct5_run_and_eval.ps1`

This wrapper standardizes:

- SaMST Distinct5 inference output layout
- SB reuse evaluation
- ArtFID generation
- per-step logging under the run root
- temporary `train.yml` / `test.yml` overwrite with automatic restoration, so
  local SaMST runs do not keep dirtying the tracked repo config files after
  the process exits
- a direct epoch-to-epoch comparison export (`json/csv/md`) once both `e5` and
  retained `e15` metrics are available
- low-frequency local polling of `run.log`, followed by automatic `e5` eval and
  `e5` vs retained `e15` comparison after the training wrapper writes
  `finished=...`

## Evidence paths

- local wrapper:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\run_samst_distinct5_local.py`
- Distinct5 local dataset:
  - `F:\wikiart_distinct5_samam_512_classview_real`
- retained `e15` eval:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\summary.json`
- active `e5` rerun root:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603`
