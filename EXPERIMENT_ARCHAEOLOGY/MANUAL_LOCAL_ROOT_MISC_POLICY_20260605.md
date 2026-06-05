# Local Root Misc / Archive / Tmp Manual Policy - 2026-06-05

Scope:

- Local root: `G:\GitHub\Latent_Style`
- Directories checked in this pass: `archive`, `tmp`, root `exp`, `final_works`, `seedream45_api`, `fast_infer_ablate43`, `latent_cyclegan`, `o20_d3`, `wikiart_fewshot`, `lambda_grid`, `step_count_sweep`, `review_additional_experiments_aggregates`, plus root single-file candidates.

This pass is not an extension scan. It opens the candidate roots one by one, checks representative files/logs/configs, and separates cleanup from evidence retention.

## Summary

Actions:

- Delete one duplicate monolithic archive tar: `archive/2026-05-19_cleanup/root/Cycle-NCE.tar`.
- Delete stale root `exp` launcher residue: two dead PID sets plus one empty probe PID.
- Retain current `Cycle-NCE`, `final_works`, `tmp`, valid generated image evidence, tracked root files, dataset/code/docs roots, and tiny placeholders.

Expected disk release:

- `Cycle-NCE.tar`: 1503.203MB.
- stale launcher residue: less than 1KB.

## Why `Cycle-NCE.tar` Is Deletable

Opened evidence:

- `archive/2026-05-19_cleanup/root/Cycle-NCE.tar`
- tar listing via `tar -tf`
- current `Cycle-NCE` top-level and summary/metrics/training CSV counts
- `EXPERIMENT_ARCHAEOLOGY/MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md`

The tar begins with:

- `Cycle-NCE/.ruff_cache`
- `Cycle-NCE/.venv`
- `Cycle-NCE/.venv/Lib/site-packages/...`
- LPIPS dependency weights inside the archived venv
- historical `full_eval`, `metrics.csv`, `summary.json`, and `training_*.csv` payloads

The current repo still has the active `Cycle-NCE` tree:

- 5069 files;
- 1541 dirs;
- 3079.423MB;
- 500 `summary.json`;
- 496 `metrics.csv`;
- 260 `training_*.csv`.

The current `Cycle-NCE` tree has already been manually indexed in:

- `manual_cycle_nce_directory_ledger_20260605.csv`
- `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md`

Decision: delete the tar. It is a duplicate cleanup-time monolithic backup, not the current evidence directory and not a checkpoint that should be thinned. Its `.venv` content also makes it poor archival evidence compared with the current checked ledger.

## Why `tmp` Is Not Deleted

Opened evidence:

- top-level `tmp` listing ordered by timestamp;
- extension grouping;
- largest files by size.

`tmp` is mostly a 2026-06-04 paper/PDF review scratch surface:

- 295 PNG files, 160.291MB;
- 10 PDF files, 12.119MB;
- 10 TEX files, 0.355MB;
- git fsck recovery text logs;
- rendered PDF page checks and visual review subdirectories.

Decision: keep for now. The active user constraint is not to touch paper tex/pdf or existing paper artifacts. Deleting `tmp` would risk crossing that boundary and losing recent recovery context.

## Why Root `exp` Is Partially Cleaned

Opened evidence:

- `exp/split_axis_tokenizer_geostat_g005_120b.pid`
- `exp/split_axis_tokenizer_geostat_g005_120b_train.err.log`
- `exp/wikiart512_ema_lowcell_weighted_from_0790_e1_b48.pid`
- `exp/wikiart512_ema_lowcell_weighted_from_0790_e1_b48_train.err.log`
- `exp/reference_memory_generation_probe/run.pid`
- representative files under `exp/highres_eval_local`

The two PID values, `37960` and `30604`, are not running. The two error logs say Python tried to open `G:\GitHub\Latent_Style\src\run.py`, which does not exist. `reference_memory_generation_probe/run.pid` is empty.

Decision: delete stale launcher files only. Keep generated highres/reference output images because they are qualitative experiment evidence.

## Retained Roots

`final_works`:

- opened metrics, summaries, training logs, full-eval logs, and configs;
- no residual real checkpoint files found;
- retained as baseline/final comparison evidence.

`seedream45_api`:

- only weight-like file is `protocol_a_800/fake_eval_checkpoint.pt`, 0.001460MB;
- retained because it saves no space and may be referenced by result metadata.

`fast_infer_ablate43`:

- inference/export tooling;
- retained.

`latent_cyclegan`, `o20_d3`, `wikiart_fewshot`:

- historical code/docs/data/snapshot surfaces;
- retained pending archive or dataset policy.

`lambda_grid` and `step_count_sweep`:

- negative/dry-run evidence;
- retained, but their `0.000/0.001s` elapsed values must not be cited as real timing.

Root tracked misc files:

- `git ls-files` shows root `__tmp_prepare_clip.py`, `__tmp_redownload_clip.py`, page PNGs, NoMachine deb, LGT figure, and `EXPERIMENT_ARCHAEOLOGY_MASTER.csv` are tracked.
- retained because deleting tracked files would be a repo reorganization change, not sidecar cleanup.

## Follow-Up

Remaining high-value cleanup is not local root misc. The next meaningful disk-recovery decisions are:

- remote `SchrodingerBridge/exp` epoch thinning;
- remote SaMAM central checkpoint thinning;
- local/remote dataset and backend latent cache policy.
