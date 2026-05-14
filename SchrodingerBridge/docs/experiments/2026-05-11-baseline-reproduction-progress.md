# 2026-05-11 Baseline Reproduction Progress

## Purpose

This is the live status board for the baseline comparison plan.

It tracks four things separately:

1. baseline assets
2. output completeness
3. evaluation coverage
4. paper-readiness

## Summary

Current repo state:

- baseline reproduction is `partially underway`
- lightweight comparison tables already exist
- stronger metric code exists in `SchrodingerBridge`
- baseline pipeline and main evaluation pipeline are still not fully unified

Update on `2026-05-11` after pipeline repair:

- `Related_Works/baseline_pipeline/unified_repro_eval.py` now writes protocol-scoped outputs by default under `results/<baseline>/protocol_a_800/`
- baseline wrappers now accept explicit `--output_root` so unified orchestration no longer mixes protocol outputs with legacy exploratory folders
- the shared aggregator now filters for generated transfer filenames only, avoiding accidental inclusion of assets like `summary_grid.png`
- a real `CUT` protocol-A copy run succeeded into the new folder contract and produced a clean `1250`-image aggregate candidate set

The biggest current gap is not training alone. The biggest gap is protocol alignment.

Update after first protocol table:

- `Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.csv` is the current deliverable table.
- It contains `Ours`, `CUT`, `SaMST`, `S2WAT`, `StyleID`, `SD-Turbo`, and four `SDEdit` strengths.
- All rows currently use the same 5-style, 750-image manifest from `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`.
- This engineering manifest is `5 source styles x 5 target styles x 30 source images = 750 outputs`, with styles `photo / monet / vangogh / cezanne / Hayao`; `ukiyoe` is intentionally not part of this run.
- Current table metrics include `CLIP-style`, `CLIP-content`, and `content LPIPS`; ArtFID/FID and missing paper metrics still need refresh.
- `StyleID` full protocol-A inference finished. Per-target wall time is recorded in `Related_Works/baseline_pipeline/results/runtime_summary_protocol_a_800.csv` at about `614-622s` per target, about `51.4min` total for five targets.

## Status Matrix

| Item | Current Status | Notes |
| --- | --- | --- |
| Ours internal CLIP/LPIPS evaluation | `done` | Already used heavily in `SchrodingerBridge` full-eval flows |
| Ours `ArtFID/FID` code path | `implemented, not yet baseline-unified` | Supported in `SchrodingerBridge/src/utils/run_evaluation.py` |
| Ours modern post-hoc metrics | `done` | `cmmd`, `dino_structure`, `gram_micro`, `gram_macro` |
| Baseline old evaluation table | `done` | `Related_Works/baseline_pipeline/results/metrics_batch.csv` |
| Baseline `CUT` outputs | `done for current 750-manifest protocol` | migrated manually from `Related_Works/runs/cut_5x5/infer_5x5/images` |
| Baseline `StyleID` outputs | `done for current 750-manifest protocol` | regenerated via unified entrypoint and aggregated cleanly |
| Baseline `SaMST` outputs | `done for current 750-manifest protocol` | migrated from reusable external SaMST full-eval outputs |
| Baseline `S2WAT` outputs | `done for current 750-manifest protocol` | regenerated via unified entrypoint and aggregated cleanly |
| Baseline `StyleAligned` outputs | `not verified` | script exists, no trustworthy result table found |
| Baseline protocol-scoped output roots | `landed for unified entrypoint` | new default is `results/<baseline>/protocol_a_800/` |
| Baseline `AdaIN` | `not started` | needed for paper table |
| Baseline `StyTr2` | `not started` | needed for paper table |
| Baseline `CAST` | `not started` | high-priority paper baseline |
| Baseline `AesPA-Net` | `not started` | high-priority paper baseline |
| Baseline `AesFA` | `not started` | recommended for AAAI-style story |
| Baseline `ArtBank` | `not started` | required if story stays artist/domain-centric |
| Current `Protocol A` engineering manifest | `frozen` | 750-image manifest from Ours reference folder; no `ukiyoe` |
| Paper `Protocol A` 800-output dataset | `not frozen` | still needed for direct CAST/StyleID paper-protocol alignment |
| `Protocol A` baseline eval with ArtFID/FID | `not landed` | code exists, pipeline glue missing |
| `CFSD / CSFD` | `not implemented` | no working implementation found |
| `CF / GE+LP` | `not implemented` | no working implementation found |
| `Time / Params / FLOPs` unified table | `not implemented` | no single collection path found |
| user preference study tooling | `not started` | future stage only |

## Verified Assets

### Baseline scripts present

- `Related_Works/baseline_pipeline/scripts/copy_cut_results.py`
- `Related_Works/baseline_pipeline/scripts/run_s2wat.py`
- `Related_Works/baseline_pipeline/scripts/run_samst.py`
- `Related_Works/baseline_pipeline/scripts/run_styleid.py`
- `Related_Works/baseline_pipeline/scripts/run_style_aligned.py`

### Verified checkpoints

`SaMST`

- `monet`
- `vangogh`
- `cezanne`
- `ukiyoe`
- `Hayao`
- `photo` is still missing

`S2WAT`

- `photo`
- `monet`
- `vangogh`
- `cezanne`
- `Hayao`
- `ukiyoe` is not complete

### Verified result coverage snapshots

These are rough inventory counts, useful for progress tracking rather than for paper claims.

| Baseline | Path | Approx Count | Meaning |
| --- | --- | ---: | --- |
| StyleID | `results/styleid/photo` | 22 | incomplete `photo` target coverage |
| StyleID | `results/styleid/monet` | 252 | target-style outputs exist |
| StyleID | `results/styleid/images` | 1001 | large mixed aggregate folder exists |
| SaMST | `results/samst/monet` | 401 | target-style outputs exist |
| SaMST | `results/samst/images` | 751 | mixed aggregate folder exists |
| S2WAT | `results/s2wat/photo` | 250 | target-style outputs exist |
| S2WAT | `results/s2wat/monet` | 310 | target-style outputs exist |
| S2WAT | `results/s2wat/images` | 1000 | mixed aggregate folder exists |
| CUT | `results/cut/monet` | 250 | target-style outputs exist |
| CUT | `results/cut/images` | 740 | mixed aggregate folder exists |

## Existing Quantitative Tables

## Checkpoint / Training Status

Authoritative local status table:

- `Related_Works/baseline_pipeline/BASELINE_CKPT_STATUS.md`

Current decision:

- do not retrain `S2WAT`, `SaMST`, `CUT`, or `StyleID` for the current 750-image protocol
- `S2WAT` already has five local 2000-epoch checkpoints
- `SaMST` has current artist checkpoints except `photo`; current table uses migrated complete outputs
- `CUT/FastCUT` has local checkpoints under `Related_Works/runs/cut_5x5/checkpoints/cut_to_*` plus complete current-protocol outputs, so it should not be retrained now
- `StyTR-2`, `AesFA`, `AesPA-Net`, `ArtBank`, `AdaIN` currently do not have their required official model weights locally
- `CycleGAN` is the first true local-training candidate if we expand artist-domain baselines

CycleGAN local training setup:

- launcher: `Related_Works/baseline_pipeline/scripts/train_cyclegan_targets.py`
- smoke run: `Related_Works/runs/cyclegan_5x5_smoke`
- smoke status: `ok`, checkpoint written for `cyclegan_to_monet`
- planned background run: serial `monet / vangogh / cezanne / Hayao` under `Related_Works/runs/cyclegan_5x5`

### Current protocol-A bridge table

File:

- `Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.csv`

Current rows:

- `ours_pareto_probe_4_epoch_0001`
- `cut`
- `samst`
- `s2wat`
- `styleid`
- `sdturbo`
- `sdedit_str_0p10`
- `sdedit_str_0p20`
- `sdedit_str_0p35`
- `sdedit_str_0p40`

Current columns:

- `clip_style`
- `clip_content`
- `content_lpips`
- `eval_sec`
- placeholder columns for `fid`, `art_fid`, and modern metrics when refreshed

Interpretation:

- good enough as a live engineering table
- not yet the final AAAI claim table
- next target is to add `CAST`, `AdaIN`, `StyTr2`, `AesPA-Net`, `AesFA`, `CycleGAN`, and `ArtBank`

Current values:

| Baseline | Images | CLIP-style up | CLIP-content up | LPIPS-content down | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ours_pareto_probe_4_epoch_0001` | 750 | 0.6908 | 0.8394 | 0.4184 | 21.7s |
| `cut` | 750 | 0.7588 | 0.7794 | 0.4906 | 23.0s |
| `samst` | 750 | 0.7253 | 0.7752 | 0.5390 | 21.9s |
| `s2wat` | 750 | 0.7138 | 0.7464 | 0.5263 | 21.6s |
| `styleid` | 750 | 0.7777 | 0.6402 | 0.5928 | 27.7s |
| `sdturbo` | 750 | 0.7769 | 0.6505 | 0.6265 | 21.6s |
| `sdedit_str_0p10` | 750 | 0.7023 | 0.8759 | 0.3236 | 21.7s |
| `sdedit_str_0p20` | 750 | 0.7063 | 0.7772 | 0.4087 | 21.7s |
| `sdedit_str_0p35` | 750 | 0.6966 | 0.6899 | 0.4904 | 23.1s |
| `sdedit_str_0p40` | 750 | 0.6968 | 0.6727 | 0.5155 | 21.8s |

### Old baseline batch table

File:

- `Related_Works/baseline_pipeline/results/metrics_batch.csv`

What it gives:

- `lpips`
- `clip_style`
- `clip_content`

Covered baselines:

- `s2wat`
- `samst`
- `styleid`
- `cut`

Interpretation:

- useful for internal triage
- not enough for fair comparison to 2024-2025 papers

### Old SB aggregate table

File:

- `Related_Works/baseline_pipeline/results/metrics_sb.csv`

Current coverage is sparse and should not be treated as the final comparison table.

## Blocking Gaps

### High-priority blockers

1. Paper-exact `20 x 40 = 800` protocol-A subset is still not frozen
2. No baseline-wide `ArtFID/FID` table
3. No `CFSD / CSFD`
4. No runnable local checkpoints found yet for `AdaIN / StyTr2 / CAST / AesPA / AesFA`

### Medium-priority blockers

1. No unified timing/params/FLOPs collection
2. No explicit `ArtBank` artist/domain evaluation track
3. No clean separation between exploratory outputs and paper outputs

## Next Actions

### Must do next

1. Freeze `Protocol A` dataset and directory contract.
2. Upgrade baseline evaluation to reuse `SchrodingerBridge` strong metrics.
3. Rebuild a clean first comparison table for:
   - `Ours`
   - `StyleID`
   - `SaMST`
   - `CUT`
4. Add `AdaIN` and `StyTr2`.

Current state of the first three actions:

- current 750-image engineering manifest is frozen and all current rows match it exactly
- `SchrodingerBridge` reuse evaluation now produces a clean 10-method table for CLIP/LPIPS
- `ArtFID/FID` refresh and the paper-exact 800-output set remain separate next steps

### After that

1. Implement `CFSD / CSFD`.
2. Add `CAST` and `AesPA-Net`.
3. Add timing, params, and FLOPs.
4. Decide whether the paper story stays on arbitrary style transfer, multi-style transfer, or artist/domain transfer.
