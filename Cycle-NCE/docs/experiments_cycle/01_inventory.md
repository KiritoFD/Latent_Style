# Step 1: Inventory And Data Quality

## Scope
- Analysis root: `experiments-cycle/`
- Included runs: any directory that has at least one of:
  - `full_eval/`
  - `logs/training_*.csv`
  - `epoch_*.pt`
- Nested runs under `experiments-cycle/experiments/*` are included.

## Extraction Method
- Script: `scripts/analyze_experiments_cycle.py`
- Command:
  - `python scripts/analyze_experiments_cycle.py --root experiments-cycle --out-dir docs/experiments_cycle/data`
- Outputs:
  - `docs/experiments_cycle/data/runs_detailed.json`
  - `docs/experiments_cycle/data/runs_metrics.csv`
  - `docs/experiments_cycle/data/history_rounds.csv`
  - `docs/experiments_cycle/data/snapshot_timeline.csv`
  - `docs/experiments_cycle/data/family_summary.csv`

## Inventory Summary
- Total detected runs: `52`
- Runs with `full_eval`: `51`
- Runs with parseable transfer metric (`analysis.style_transfer_ability.clip_style`): `35`
- Runs with `summary_history.json`: `8`
- Runs with `src_snapshot_*`: `18`
- Runs with training csv logs: `40`
- Runs with checkpoints (`epoch_*.pt`): `39`

Family split (from `runs_metrics.csv`):
- `overfit50`: `28`
- `small-exp`: `8`
- `adacut`: `5`
- `full_300`: `5`
- `full_250`: `2`
- `experiments_misc`: `1`
- `full-300`: `1`
- `full_other`: `1`
- `other`: `1`

## Data Quality Tiers

### Tier A (directly comparable)
Criteria:
- Has parseable `best_transfer_clip_style`
- `matrix_complete_square=True` (complete style-pair matrix)
- Has explicit metric fields in `summary.json`

Count: `33/52`

### Tier B (partially comparable)
Typical issues:
- `content_lpips` hardcoded `0.0` (older evaluator output style)
- Different eval sample count (`30` vs `50` per pair)
- Missing `summary_history.json` (only single-point results)

### Tier C (not directly comparable)
Runs with `full_eval` artifacts but no parseable summary metric:
- `adacut`
- `adacut_overfit0`
- `experiments/main-style-distill-struct-v1`
- `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_145326`
- `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150950`
- `experiments/small-exp-overfit50_e13_hires6_spatialproto_v1-overfit50_e13_hires6_spatialproto_v1-bd128-dsp1-hp0p2-whf3p0-wprob0p8-wproto0p2-wcyc8p0-20260209_151448`
- `experiments/small-exp-overfit50_e14_hires6_weakcls_v1-overfit50_e14_hires6_weakcls_v1-bd128-dsp1-hp0p18-whf3p4-wprob0p35-wproto0p2-wcyc8p0-20260209_151947`
- `experiments/small-exp-overfit50_e15_style_only_from_smoke`
- `experiments/small-exp-overfit50_e16_style_only_flowboost`
- `experiments/small-exp-overfit50_e17_style_forcepath`
- `experiments/small-exp-smoke_e12_skipfusion_v2-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150800`
- `full-300-3060-313`
- `full_300_strong-style-v2`
- `overfit50-clipstyle-probe-v1`
- `overfit50-clipstyle-probe-v2`
- `overfit50-clipstyle-probe-v3`

Only run without `full_eval`: `adacut_overfit`

## Snapshot Coverage
- Runs with snapshots: `18`
- Total snapshot dirs: `45`
- Runs with dense snapshot history:
  - `full_300_distill_low_only_v1` (8 snapshots)
  - `full_300_gridfix_v2` (7 snapshots)
  - `overfit50-style-distill-struct-v4` (6 snapshots)

Snapshot code drift is concentrated in a few lines:
- `full_300_gridfix_v2`: model/loss/trainer hashes all changed multiple times.
- `overfit50-distill_low_only`: substantial config and code shift between two snapshots.
- `overfit50-v5-mse-sharp-style_back`: model/trainer stable, loss changed once.

## Inventory-Level Risks
- Metric comparability is not uniform across all 52 runs.
- Some top-scoring historical runs use evaluator outputs with `content_lpips=0.0`, which should be treated as low-confidence for content retention conclusions.
- Several runs preserve generated images but not machine-readable summary metrics, reducing auditability.
