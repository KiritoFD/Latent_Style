# 2026-05-11 Baseline Reproduction Lab Notes

## Scope

This file records factual observations discovered while auditing the current baseline reproduction stack.

It is intentionally closer to a lab notebook than a polished plan.

## Findings

### 1. The repo is ahead of the current baseline docs

`Related_Works/baseline_pipeline/evaluation/EVAL_REQUIREMENTS.md` was still describing an older 5x5-style execution checklist.

But the codebase now already contains:

- `ArtFID/FID` computation support
- modern post-hoc metrics
- batch aggregation helpers

So the main issue is no longer lack of metric code alone. The issue is that baseline reproduction and main-model evaluation evolved separately.

### 2. Baseline evaluation is split across two generations

Old generation:

- `eval_all_baselines.py`
- `eval_with_sb.py`
- `run_sb_eval_all.py`
- `run_sb_eval_v2.py`

What they mainly track:

- `clip_style`
- `clip_content`
- `content_lpips`

New generation:

- `SchrodingerBridge/src/utils/run_evaluation.py`
- `SchrodingerBridge/src/utils/modern_metrics.py`
- `SchrodingerBridge/append_modern_metrics.py`

What the newer stack can support:

- `ArtFID`
- `FID`
- `cmmd`
- `dino_structure`
- `gram_micro`
- `gram_macro`

### 3. Baseline assets are partial, not absent

This matters for planning.

The repo already has:

- reproducible launch scripts for `StyleID / SaMST / S2WAT / StyleAligned`
- copied `CUT` outputs
- real checkpoints for `SaMST` and `S2WAT`
- recorded baseline metrics in `metrics_batch.csv`

So the right framing is:

- not "start baseline reproduction from zero"
- but "normalize, finish, and upgrade baseline reproduction"

### 4. Photo coverage is still the most fragile area

Observed during the audit:

- `StyleID` photo outputs are clearly incomplete
- `SaMST` photo checkpoint/output is still missing
- `S2WAT` photo checkpoint/output exists

This means `photo_to_art` comparisons will be uneven until the photo branch is repaired for the missing baselines.

### 5. Several paper-critical metrics are still absent

No working repo-level implementation was found for:

- `CFSD / CSFD`
- `CF`
- `GE+LP`
- `AesPA pattern difference`
- `AesPA style loss`
- unified `Time / Params / FLOPs / OIP / style capacity`

So even after baseline output cleanup, the comparison table is still missing headline paper columns.

## Interpretation

The current repo can support a credible phase-1 comparison story:

- strong internal content preservation
- partially reproduced external baselines
- evaluation stack already moving toward modern metrics

The current repo cannot yet support a credible phase-2 headline claim:

- "better than SaMST"
- "better than CAST"
- "better than StyleID"
- "better than AesFA"

because the protocol-aligned tables still do not exist.

## Recommended Working Order

1. Make `Protocol A` the first-class benchmark.
2. Unify baseline output structure around that protocol.
3. Reuse `SchrodingerBridge` evaluation for both `Ours` and baselines.
4. Add missing paper baselines in order of review importance:
   - `AdaIN`
   - `StyTr2`
   - `CAST`
   - `AesPA-Net`
5. Only after that, implement missing specialized metrics.

## Update Rule

When a new baseline is added or a missing metric lands, append a short dated note here:

- what changed
- what file/script was added or updated
- whether it improves `Protocol A`, `Protocol B`, or `Protocol C`

This keeps the planning docs stable while preserving an audit trail.

## Dated Notes

### 2026-05-11: unified protocol output repair

- Updated `Related_Works/baseline_pipeline/unified_repro_eval.py` to default to protocol-scoped result roots such as `results/<baseline>/protocol_a_800/`
- Updated baseline wrappers in `Related_Works/baseline_pipeline/scripts/` to accept `--output_root` and to resolve the workspace root correctly
- Fixed `StyleID` wrapper to use an img2img pipeline class compatible with the current diffusers call signature
- Tightened baseline image aggregation so evaluation reuse only collects actual generated transfer files containing `_to_`
- Verified this path concretely with a `CUT` run through the unified entrypoint; protocol-A folder population succeeded and produced a clean `1250`-file aggregate candidate set

Impact:

- materially improves `Protocol A`
- reduces protocol drift between baseline generation and `SchrodingerBridge` evaluation reuse

### 2026-05-11: manual migration of existing run outputs

- User clarified that existing `Related_Works/runs` results should be migrated manually, not skipped dynamically by the large runner.
- Added `Related_Works/baseline_pipeline/migrate_existing_results.py` as a one-shot migration helper.
- Migrated only files matching the 750-image reference manifest from `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`.
- Successfully migrated:
  - `cut` from `Related_Works/runs/cut_5x5/infer_5x5/images`
  - `samst` from `Related_Works/external/SaMST/full_eval/repro_5style_train2/epoch_0100_overfit50/images`
  - `sdturbo` from `Related_Works/runs/sdturbo_5x5/images`
  - `sdedit_str_0p10` from `Related_Works/runs/sdedit_multi/str_0.10/images`
  - `sdedit_str_0p20` from `Related_Works/runs/sdedit_multi/str_0.20/images`
  - `sdedit_str_0p35` from `Related_Works/runs/sdedit_multi/str_0.35/images`
  - `sdedit_str_0p40` from `Related_Works/runs/sdedit_multi/str_0.40/images`
- Migration report:
  - `Related_Works/baseline_pipeline/results/manual_migration_protocol_a_800.csv`
  - `Related_Works/baseline_pipeline/results/manual_migration_protocol_a_800.json`

Impact:

- `CUT` no longer needs training or inference for this protocol; it only needs evaluation from the migrated `protocol_a_800` folder.
- `SaMST` also has a complete migrated `protocol_a_800` result folder, so it can be evaluated directly before any retraining experiments.
- The large runner should not contain historical-run skip logic. Existing outputs are now first-class protocol folders.

### 2026-05-11: first 10-method protocol table

- Completed full current-protocol `StyleID` inference for `photo / monet / vangogh / cezanne / Hayao`.
- Aggregated `StyleID` to `Related_Works/baseline_pipeline/results/styleid/protocol_a_800/images` with `750` manifest-matched images.
- Re-ran reuse evaluation for:
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
- Main table:
  - `Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.csv`
  - `Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.md`
- Runtime table:
  - `Related_Works/baseline_pipeline/results/runtime_summary_protocol_a_800.csv`

Impact:

- establishes a real 10-method engineering comparison table over one frozen 750-image manifest
- confirms `StyleID` has the strongest current CLIP-style score in this table but substantially weaker content preservation than Ours/SDEdit low-strength
- still does not justify final paper claims against `CAST / AesFA / AesPA-Net` because those baselines and ArtFID/FID/CFSD are not yet complete

### 2026-05-11: timing persistence fix

- Updated `Related_Works/baseline_pipeline/unified_repro_eval.py` so `unified_repro_eval_summary_*.csv` includes `elapsed_sec`.
- Updated runtime summary writing so future timed training/inference rows append to existing protocol runtime logs instead of overwriting them.
- Verified syntax with `python -m py_compile` for:
  - `Related_Works/baseline_pipeline/unified_repro_eval.py`
  - `Related_Works/baseline_pipeline/evaluate_protocol_results.py`
  - `Related_Works/baseline_pipeline/migrate_existing_results.py`

Impact:

- future baseline training and inference runs will preserve wall-clock timing for comparison
- already completed manual migrations still have no true training/inference timing unless separately recovered from their original logs
