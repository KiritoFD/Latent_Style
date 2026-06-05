# Authoritative Local / Remote Experiment Archaeology Conclusions - 2026-06-05

This is the current human-readable entry point for the archaeology work. It is
not a completion claim. It summarizes what has been opened, indexed, cleaned,
retained, and what still needs an 8-hour-level continuation.

Write scope in this pass: `EXPERIMENT_ARCHAEOLOGY` only. No paper TeX/PDF,
source code, or existing user dirty files were edited, staged, or reverted.

## Executive Conclusion

The repository is not one experiment directory. It is a merged evidence
workspace with three active surfaces:

- Local `G:\GitHub\Latent_Style`: datasets, latent/cache roots,
  SchrodingerBridge experiments, Related_Works baselines, Cycle-NCE history,
  root archive/tmp surfaces, and paper workspace.
- Remote main `I:\Github\Latent_Style`: SchrodingerBridge formal and
  exploratory runs, SaMAM baselines, data/cache/archive surfaces, Cycle-NCE RAR
  history, and expanded experiment evidence.
- Remote TokenizerClean `I:\Github\Latent_Style_TokenizerClean`: a separate
  closing/evidence surface for AAAI2027/tokenizer experiments, including
  cited/current packets and no-summary trained payloads.

The broad experiment data is already materialized as:

- `final_master_experiments.csv`: 22629 rows.
- `final_timeline.csv`: 7829 events.
- `conclusions_by_dataset.csv`: 25 dataset/setting conclusion rows.
- `final_by_dataset/`: per-dataset/per-setting CSV split.
- `timing_quality_master_20260605.csv`: 1093 timing rows with quality labels.

The gap was not that no data existed. The gap was that the conclusion layer was
too fragmented and some readable reports were not reliable as a clean entry
point. This file and
`authoritative_local_remote_lineage_conclusions_20260605.csv` are the current
clean entry point.

## Current Live State Checked This Pass

Remote `I:\Github` was checked over SSH on 2026-06-05. It is not empty.

Top-level remote `I:\Github` currently contains:

- `26AI-H`
- `26AI-H.zip`
- `Latent_Style`
- `Latent_Style_TokenizerClean`
- `find_clip_remote.bat`

Remote main live facts:

- `I:\Github\Latent_Style` exists with 23 dirs and 53 files at root.
- `I:\Github\Latent_Style\SchrodingerBridge\exp` exists with 123 dirs and 1 file.
- `I:\Github\Latent_Style\Cycle-NCE` exists with 25 dirs and 79 files.
- `I:\Github\Latent_Style\Cycle-NCE\45.rar` still exists, size 507.452 MB,
  last write `2026-04-06 00:11:44`.
- `I:\Github\Latent_Style\experiments.rar` and
  `I:\Github\Latent_Style\Cycle-NCE\experiments.rar` are absent after resolved
  duplicate cleanup.

Remote TokenizerClean live facts:

- `I:\Github\Latent_Style_TokenizerClean` exists with 17 dirs and 37 files at root.
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp` currently has
  142 dirs and 23 files after cleanup.
- Earlier TokenizerClean graph work covered 145 exp top-level dirs before owner
  cleanup removed pure orphan probe dirs.

Local current facts:

- `G:\GitHub\Latent_Style` still has unrelated dirty files outside
  `EXPERIMENT_ARCHAEOLOGY`. They were intentionally not touched.
- `EXPERIMENT_ARCHAEOLOGY` is the only staged/committed work area for this
  archaeology continuation.

## Local Conclusions

Local coverage has three levels:

- Broad row-level indexes: `final_master_experiments.csv`,
  `final_timeline.csv`, `final_by_dataset/`.
- Manual surface coverage: `manual_top_level_directory_index_20260605.csv`
  has 67 rows, including 32 local rows; `manual_coverage_matrix_20260605.csv`
  has 41 coverage rows; `manual_family_walkthrough_20260605.csv` has 31
  family rows.
- Policy and cleanup evidence: local `manual_local_*_policy_20260605.csv`
  files plus `cleanup/*.csv`.

Local directory meaning:

- `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`,
  `SchrodingerBridge/scale`, and `horse2zebra` are data, latent, feature, or
  dataset-cache surfaces. They are not training checkpoints.
- `eval_cache` is model/eval dependency surface: ArtFID, CLIP, VAE, DINO,
  offline-pairing, reference-feature, compile, and ONNX cache files.
- `SchrodingerBridge/exp` is the current LANCET/LBM evidence surface. It holds
  formal eval summaries, timing packets, compact-anchor outputs, and historical
  probes.
- `Related_Works` is baseline evidence: SaMAM, SaMST, CUT, CycleGAN, S2WAT,
  StyleID, Seedream, and protocol/eval outputs.
- `Cycle-NCE` is historical evidence, not just stale output. Its summaries,
  metrics, videos, source files, and archive relationships matter.
- Root `archive`, `tmp`, and `exp` are mixed scratch/history surfaces. They
  require provenance checks and cannot be swept by extension or size.

Local cleanup already performed:

- Local checkpoint-like cleanup ledger: 875 deleted files / 46032.053 MB in
  `cleanup/manual_deleted_checkpoints_20260605.csv`.
- Invalid local eval cache residue: 55.994 MB.
- Duplicate root archive tar and stale launcher residue: 1503.203 MB plus tiny
  residue.
- Failed dataset download cache residue: 63.948 MB.
- Local remaining-surface whitelist cleanup: 237.860 MB.
- Five unreferenced CUT video work-frame dirs: 3068.463 MB.

Local generated-media conclusion:

Generated media is not a single delete class. Passes 1, 2, and 3 opened exact
dirs and kept formal evals, paper bundles, no-op/IDT controls, timing
benchmarks, protocol baselines, compact-anchor evals, diagnostics, and
generation-only calibration evidence. Only the five frame-only video work dirs
were deleted because they had no mp4/json/csv and final video evidence is
retained elsewhere.

Local remaining gap:

- Continue local generated-media owner review below the current candidate
  cluster.
- Build a separate archive/temp/paper-scratch provenance pass without touching
  paper TeX/PDF.
- Keep dataset mirrors separate from generated-output cleanup.

## Remote Main Conclusions

Remote main `I:\Github\Latent_Style` has already had multiple manual cleanup
passes with policy CSVs, per-file ledgers, and post-delete verification.

Remote main cleanup already performed:

- SchrodingerBridge epoch thinning: 84 remote checkpoint files deleted,
  4961.604 MB released.
- SaMAM checkpoint thinning: 7 redundant `last*.ckpt` aliases deleted,
  1931.291 MB released.
- Data/cache/archive residue cleanup: 11 exact whitelist targets deleted,
  381.807 MB released.
- Duplicate/stale archive cleanup: 3 archives deleted, 3290.714 MB released.
- Weight-only RAR cleanup: 6 RAR files deleted, 6553.384 MB released.
- Resolved duplicate `experiments.rar`: deleted, 8091.026 MB released.

Remote main retained evidence:

- Current/formal SchrodingerBridge exp anchors.
- SaMAM curve checkpoints and non-duplicate step checkpoints.
- Valid data roots, latent roots, and eval caches.
- Expanded `experiments` evidence after `experiments.rar` deletion.
- Cycle-NCE evidence directories and `45.rar`.

Remote main remaining gap:

- `Cycle-NCE\45.rar` cannot be deleted yet. The curated extraction policy shows
  it contains 6096 entries: 4 configs, 8 summaries, 8 metrics CSVs, 5 training
  CSVs, 42 source/structured files, 9 other files, 6008 generated/eval images,
  and 12 weights. It is not a weight-only archive.
- Cache duplicates cannot be deleted from hash equality alone. The current
  policy requires canonical cache-root migration, symlink/junction policy, and
  offline eval verification.

## Remote TokenizerClean Conclusions

TokenizerClean is separate from remote main and must not be collapsed into it.
It is the closing/evidence surface for AAAI2027/tokenizer work.

TokenizerClean cleanup already performed:

- Uncited exploratory checkpoints: 141 files deleted, 5198.991 MB released.
- No-summary probe/calibration checkpoints: 18 files deleted, 362.391 MB released.
- Pure orphan probe targets: 3 dirs and associated weights deleted, 170.017 MB
  released.
- Uncited generated media: 43008 files deleted, 11883.246 MB released.

TokenizerClean retained media:

- 26 cited/current media dirs were source-opened.
- Retained media surface: 46483 media files / 7501.518 MB.
- Policy split: 8 current AAAI2027 media dirs and 18 cited media dirs.
- All 26 rows are `keep_no_delete` pending archive migration.

TokenizerClean retained no-summary payloads:

- 7 trained no-summary payload dirs remain retained.
- Retained no-summary payload weight surface: 10 weight files / 379.322 MB.
- 5 payloads have no external evidence and remain training-log-only.
- 1 payload is retained as a downstream resume source.
- 1 payload is retained as a diagnostic evaluated payload.

TokenizerClean remaining gap:

- Recover or generate summaries for the 5 training-log-only payloads, or get an
  explicit owner delete decision.
- Build a citation-to-artifact manifest for the 26 retained media dirs before
  any archive migration.
- Do not delete current/cited media by count or size.

## Dataset And Lineage Conclusions

The 25 dataset/setting rows in `conclusions_by_dataset.csv` are the current
dataset-level split. The most important current surfaces are:

- `distinct5_512`: 417 rows, 278 metric rows, 55 train-timing rows, 113
  infer-timing rows. This is a core current claim surface with LANCET/LBM,
  SaMAM, SaMST, SD-Turbo, no-op, and IDT evidence.
- `wikiart512_5style`: 200 rows, 144 metric rows, 14 train-timing rows, 10
  infer-timing rows. This is the main WikiArt512 formal/timing surface.
- `strict_protocol_750`: timing and full-eval packet surface for strict 750
  evaluations.
- `schrodingerbridge_exp_general`, `schrodingerbridge_grid_search`,
  `schrodingerbridge_weight_sweep`, `schrodingerbridge_frontier`,
  `schrodingerbridge_vae_backend`, and
  `schrodingerbridge_representation_probe`: exploration and ablation surfaces.
- `cycle_nce` and `legacy_style_transfer_experiments`: historical/background
  evidence surfaces, not current formal headline surfaces by default.

The current experiment lineage is:

- Phase A: 2026-02 to 2026-03, legacy/no-edge/style-transfer sanity and failed
  or historical experiments.
- Phase B: 2026-03 to 2026-04, legacy256, StyleID, IDT, tokenized/no-tokenized,
  and baseline sanity.
- Phase C: 2026-04 to 2026-05, Cycle-NCE and Latent AdaCUT history.
- Phase D: 2026-05, SchrodingerBridge/LANCET sweeps, grid/search/frontier,
  VAE backend, representation, and tokenizer probes.
- Phase E: 2026-05-30 to 2026-06-02, WikiArt512 and Distinct5 formal evidence.
- Phase F: 2026-06-03 onward, AAAI2027/TokenizerClean claim closing:
  flow-loss, SA-SWD, tokenizer execution, localization, and time-to-parity.

Lineage gap:

Some older generated reports and dataset conclusion text remain rough or
mojibake in places. The authoritative row data is still usable, but the final
human-readable dataset-by-dataset prose needs another cleanup pass.

## Timing Conclusions

Timing state:

- Docs timing master:
  `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`, 419 rows.
- Archaeology timing overlay:
  `timing_quality_master_20260605.csv`, 1093 rows.
- Missing-docs timing source-open:
  `timing_candidate_missing_docs_source_open_20260605.csv`, 26 rows.

Timing policy:

- Preserve original units.
- Leave missing train/infer values blank.
- Do not force train time into seconds.
- Keep generation-only, full-eval, train+eval, smoke, anomalous, and audit-only
  rows separated by quality labels.

Timing gap:

- Docs timing master was not edited in this pass.
- 370 docs timing rows still lack overlay/source-open coverage.
- Owner selection is needed before promoting rows into a future paper-facing
  timing table.

## Cleanup Ledger Conclusion

The current ledger synthesis reports about 92162.847 MB released. That number
is not a claim that the whole repo is clean. It means every counted deletion
block has a policy/ledger/post-delete verification path.

Deletion rule:

- No broad deletion by extension.
- No broad deletion by size.
- No deletion from scan-only output.
- Delete only exact whitelist targets after source-open evidence, policy CSV,
  deletion ledger, and post-delete verification.

## 8-Hour Continuation Plan

Block 1, 1.0h:
Continue local generated-media owner review below pass3. Produce pass4 CSV/MD
and delete only if exact whitelist proof exists.

Block 2, 1.0h:
Create a curated nonweight extraction package plan for `Cycle-NCE\45.rar`.
Verify entry counts before any archive deletion decision.

Block 3, 1.0h:
Remote TokenizerClean no-summary recovery. For the 5 training-log-only payloads,
attempt summary recovery or prepare an owner-decision table.

Block 4, 1.0h:
TokenizerClean cited/current media manifest. Map docs/paper references to
summary, metrics, grids, generated images, and retained checkpoints.

Block 5, 1.0h:
Timing promotion plan. Select candidate rows for docs timing master update,
keeping original units and caveats.

Block 6, 1.0h:
Triage the 370 docs timing rows that lack overlay/source-open coverage.

Block 7, 1.0h:
Final consistency audit over `final_by_dataset`, `final_timeline`, README
counts, cleanup totals, and direct conclusion index.

Block 8, 1.0h:
Requirement-by-requirement completion audit. Either prove the objective is
complete with current evidence or leave a precise remaining gap list.

## Current Completion Status

Not complete.

The task is substantially advanced, but completion is unproven because the
remaining gaps are concrete: local nested generated media, `45.rar` curated
extraction, TokenizerClean summary recovery/owner decisions, retained-media
manifests, docs timing promotion, and final consistency audit.
