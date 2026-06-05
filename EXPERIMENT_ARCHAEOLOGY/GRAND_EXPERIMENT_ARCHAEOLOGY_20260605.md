# Grand Experiment Archaeology - 2026-06-05

Scope: local `G:\GitHub\Latent_Style`, remote `I:\Github\Latent_Style`, and remote `I:\Github\Latent_Style_TokenizerClean`.

Write scope for this pass: `EXPERIMENT_ARCHAEOLOGY/**` only. No paper `tex/pdf`, source files, configs, PDF QA images, or unrelated dirty files were edited.

## Why this pass exists

The earlier broad indexes are useful for navigation, but they are not enough for a cleanup conclusion. This pass re-opened the major directories one by one and separates five classes that must not be conflated:

- data and latent tensors;
- metric/model caches and pretrained dependencies;
- current paper-facing experiment evidence;
- historical or negative-closure evidence;
- actual disposable checkpoint residue.

The main correction is that raw `.pt/.pth/.ckpt/.safetensors` counts are misleading. In this repository, those extensions include per-image latent tensors, CLIP features, VAE/ArtFID/LPIPS dependencies, fake eval placeholders, and real train checkpoints.

## Manual evidence products from this pass

New manual products:

- `manual_top_level_directory_index_20260605.csv`
- `manual_family_walkthrough_20260605.csv`
- `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv`
- `manual_related_works_directory_ledger_20260605.csv`
- `manual_cycle_nce_directory_ledger_20260605.csv`
- `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`
- `manual_cleanup_retention_and_next_candidates_20260605.csv`
- `GRAND_EXPERIMENT_ARCHAEOLOGY_20260605.md`
- `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md`
- `MANUAL_REMOTE_SCHRODINGERBRIDGE_EXP_20260605.md`

Existing manual products extended/used as context:

- `manual_timing_evidence_20260605.csv`
- `manual_directory_classification_20260605.csv`
- `manual_evidence_log_20260605.csv`
- `manual_remaining_weight_classes_20260605.csv`
- cleanup ledgers under `cleanup/`

Broad auto indexes remain useful but are not treated as sufficient proof:

- `final_master_experiments.csv`
- `final_timeline.csv`
- `final_by_dataset/*.csv`
- `ARCHAEOLOGY_REPORT.md`
- `EXPERIMENT_TIMELINE.md`

## Chronological lineage

### February to March: legacy latent and early style-transfer line

The earliest surfaces are dataset, latent, and Cycle-NCE style-transfer experiments:

- local `style_data`, `latent-256`, `clip-feats-vitb32`, `wikiart_fewshot`;
- remote `data`, `style_data`, `latents`, `latents_overfit50`, `latent-256`;
- remote `experiments`;
- remote/local `Cycle-NCE`.

Interpretation:

- Most `.pt` files in these areas are per-image latents or features, not model checkpoints.
- Remote `experiments` is a large legacy archive with old ablation names from February to April. It should be archived with a separate historical policy, not cleaned by a checkpoint-only rule.
- `Cycle-NCE` contains historical reports, summary/metrics/log evidence, source snapshots, eval caches, and visualization outputs. The local manual pass found `500` `summary.json`, `496` `metrics.csv`, and `260` `training_*.csv`; local residual `.pt` files are only `eval_cache/ref_feats_*.pt`, not train checkpoints.

### March to April: baseline and external method reproduction

The baseline family lives mostly under:

- local `Related_Works`;
- remote `I:\Github\Latent_Style\Related_Works`;
- remote `I:\Github\Latent_Style\StarGAN`;
- local `final_works`.

Methods covered:

- AdaIN / AesFA / AesPA-Net dependency surfaces;
- CUT and CycleGAN 5x5 runs;
- SaMST and SaMAM;
- StarGAN;
- SDEdit / SDTurbo / Seedream placeholders.

Cleanup status:

- Local actual non-mainline checkpoints in checked CUT/CycleGAN/final_works targets were already deleted in earlier cleanup ledgers.
- Remaining local Related_Works weights are primarily VGG/Inception/LPIPS dependencies and tiny `fake_eval_checkpoint.pt` placeholders.
- Remote Related_Works retains 19 SaMAM Distinct5 step checkpoints under the central `samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag` run. These are current baseline curve evidence, not unreviewed junk.

### May: SchrodingerBridge historical and phase-space sweeps

Important historical roots:

- `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0`;
- remote `SchrodingerBridge/review_additional_experiments`;
- `lambda_grid`;
- `step_count_sweep`;
- `review_additional_experiments_aggregates`;
- `efficiency`.

Manual correction:

- Root `manifest.json` has `dry_run: true`.
- `run_summary.json` and `run_summary.csv` contain `0.000` / `0.001` elapsed values for `lambda_grid` and `step_count_sweep`.
- Those rows must be treated as negative evidence: they prove the sweep shell existed, not that real training or inference happened.

Historical anchor:

- `S-add__K-1_C-0_W-20_Col-0` remains a benchmark gate and should be retained until a packaged reproduction archive exists.

### Late May to early June: WikiArt512 timing and Distinct5 current evidence

Current local evidence surfaces:

- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`
- `SchrodingerBridge/exp/timing_20260602`

Current remote evidence surfaces:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_*_remote`
- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_*`
- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_path_kinetic_*`
- `I:\Github\Latent_Style\SchrodingerBridge\exp\sadd_*`

Key conclusion:

- `SchrodingerBridge/exp` on the remote main checkout contains 101 retained train weights and 5945.063 MB. These are not a generic trash pile. They are current Distinct5 evidence, path-kinetic evidence, or reproduction lineage.

### June 3: TokenizerClean and AAAI2027 claim-closing packets

Remote `I:\Github\Latent_Style_TokenizerClean` is not an empty or irrelevant directory. It is a clean remote checkout used for current tokenizer/AAAI work.

Opened TokenizerClean evidence:

- `SchrodingerBridge/docs/experiments/2026-06-03-exp-surface-classification.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-timing-artifact-prune.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-repo-cleanup-and-archive-pass.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- `SchrodingerBridge/docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-saswd-axis-ablation/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-execution-alignment-l-family/README.md`

TokenizerClean weight split:

- total `SchrodingerBridge`: 334 files, 11822.873 MB;
- normal `SchrodingerBridge/exp`: 326 files, 11375.355 MB;
- two special SA-SWD evidence paths displayed as `exp?saswd...`: 6 files, 403.558 MB;
- `artifacts`: 1 file, 43.635 MB;
- `eval_cache`: 1 file, 0.324 MB.

Important path note:

- The special SA-SWD paths are displayed as `exp?saswd_axis_h_base_seed42_b44_saswd_random` and `exp?saswd_axis_h_base_seed42_b44_saswd_semantic`; the separator after `exp` is not a normal slash. Earlier notes observed character code 61532. Treat these as real evidence paths and do not delete them by assuming they are malformed duplicates.

TokenizerClean cleanup boundary:

- Its own docs explicitly say not to mass-move or mass-delete `exp/` while docs and master logs cite direct paths.
- Frozen local probes should stop growing, but cited paths stay until the citation graph is migrated.
- New formal packets should use `aaai2027_*` naming plus paired docs and master-log rows.

## Local tree conclusions

### Clean or not clean

The local tree is clean with respect to the previously targeted non-mainline checkpoint cleanup:

- `Related_Works/runs/cut_5x5/checkpoints` reopened clean;
- `Related_Works/runs/cyclegan_5x5*/checkpoints` reopened clean;
- `Related_Works/final_works/trial_0016`, `trial_0019`, and `trial_0044` reopened clean;
- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync` reopened clean.

The local tree is not "zero weights", and it should not be. Remaining major classes:

- formal WikiArt512 timing anchor weights;
- metric dependencies;
- data/feature/latent caches;
- baseline dependency weights;
- fake eval placeholders.

### Local next cleanup candidates

The earlier local model-weight candidate was refreshed in the continuation pass:

- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_vramprobe`
  - previous scan: 3 files, 203.052 MB
  - current scan: 0 files, 0 MB
  - current class: stale probe checkpoint candidate

No delete was performed in this continuation because the directory/weights were no longer present when reopened. Current local `SchrodingerBridge/exp` exact extension scan has only 9 weight-like files: 8 formal WikiArt512 epoch weights and 1 ArtFID metric dependency.

Tiny `fake_eval_checkpoint.pt` placeholders exist in baseline result folders. They save almost no disk and may be referenced by result metadata. They belong in a separate placeholder-only cleanup policy, not a disk-recovery pass.

## Remote `I:\Github\Latent_Style` conclusions

### Remote root boundary

`I:\Github` contains:

- `Latent_Style`;
- `Latent_Style_TokenizerClean`;
- `26AI-H`;
- `26AI-H.zip`.

`26AI-H` is outside this task's Latent_Style scope and was not cleaned or indexed as a Latent_Style experiment.

### Data and latent roots

Remote data/latent sizes from manual root counts:

| root | files matched by weight-like extension | size MB | conclusion |
|---|---:|---:|---|
| `data` | 3422 | 218.828 | data |
| `style_data` | 8284 | 530.470 | data |
| `latents` | 10361 | 663.374 | latent data |
| `latents_overfit50` | 100 | 1.713 | latent data |
| `latent-256` | 10361 | 177.702 | latent data |
| `latent-256-flux1` | 10361 | 3900.374 | backend latent cache |
| `latent-256-flux2` | 10361 | 5195.436 | backend latent cache |
| `latent-256-kl-f4` | 10361 | 3899.171 | backend latent cache |
| `latent-256-kl-f4-mode` | 10361 | 3899.171 | backend latent cache |
| `latent-256-sd15-ema` | 10361 | 1310.264 | backend latent cache |
| `latent-256-sdxl` | 10361 | 1310.264 | backend latent cache |
| `latent-256-sdxl-fp32` | 10361 | 1310.264 | backend latent cache |

These are large, but they are not checkpoint cleanup targets. Deleting them is a dataset/backend-cache decision.

### Remote SchrodingerBridge split

Raw remote `SchrodingerBridge` weight-like total:

- 11469 files;
- 9441.059 MB.

First-level split:

| first-level root | files | size MB | interpretation |
|---|---:|---:|---|
| `exp` | 101 | 5945.063 | current and lineage train weights |
| `scale` | 11349 | 2859.902 | scale/dataset tensors |
| `S-add__K-1_C-0_W-20_Col-0` | 8 | 345.642 | historical strict750 gate |
| `review_additional_experiments` | 9 | 289.800 | historical review evidence |
| `eval_cache` | 2 | 0.652 | eval cache |

The first row is the true remote experiment-checkpoint surface. The second row is data, not train checkpoint clutter.

The 2026-06-05 remote top-level walkthrough refined `SchrodingerBridge/exp`:

- `124` top-level entries in `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`.
- `17` top-level directories contain weights.
- `101` weight files total, `5945.064 MB`.
- Weighted directories are current Distinct5/AAAI2027 packets or SADD lineage: `aaai2027_longer_train_*`, `aaai2027_path_kinetic_*`, `distinct5_512_ema_*`, `sadd_exact_*`, and `sadd_repro_*`.
- Opened timing examples show Distinct5/AAAI2027 training `epoch_time_sec` around `62-67s` for 4972 samples and full-eval `wall_total` around `94.8-136.4s`, depending on variant.
- `vae_backend`, `inference`, `frontier`, `tokenizer`, and `representation` are zero-weight evidence surfaces with summaries/logs/source/image outputs; they are not checkpoint deletion targets.

No remote `SchrodingerBridge/exp` deletion was performed in this pass. Future cleanup here should be an epoch-thinning policy, not broad extension deletion.

### Remote Related_Works split

Remote `Related_Works` retains 27 weight-like files, 5394.883 MB.

The actual disk-heavy part is:

- `baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/step_checkpoints`
  - 19 files;
  - about 5242 MB;
  - central SaMAM Distinct5 baseline curve.

Other files:

- VGG dependencies under SaMAM and S2WAT;
- LPIPS dependency under StarGAN;
- tiny fake eval placeholders.

Conclusion: remote SaMAM cleanup is clean under the current retention rule. The remaining central step checkpoints are evidence, not random old ckpt files.

### Remote Cycle-NCE and experiments

Remote `Cycle-NCE`:

- 37 weight-like files;
- 937.553 MB;
- opened detail shows CLIP/VAE/ArtFID/ref feature caches, classifier latent cache, tokenizer/pretrained tiny weights, and virtualenv package weights.

Remote `experiments`:

- 3 weight-like files;
- 319.141 MB;
- opened detail shows VAE/cache style files.

Conclusion: both roots need an archive policy, not a checkpoint deletion pass.

### Remote StarGAN and Seedream

Remote `StarGAN`:

- 4 tiny fake eval checkpoint placeholders;
- total about 0.006 MB.

Remote `seedream45_api`:

- 1 tiny fake eval checkpoint placeholder;
- 0.00146 MB.

These can be removed only if a placeholder-only cleanup rule is adopted. They do not matter for disk recovery.

## Cleanup state

### Already cleaned

Earlier cleanup ledgers record deletion of non-mainline checkpoint payloads and post-delete verification for local cleanup targets and remote SaMAM excess checkpoint classes.

Current checked clean surfaces:

- local CUT/CycleGAN checkpoint folders;
- local final_works trial checkpoint folders;
- local Distinct5 ckptsync calibration weights;
- remote SaMAM non-central extras, leaving only the central 19 step checkpoints.

### Still retained, intentionally

Retained because current evidence depends on them:

- local WikiArt512 formal anchor weights;
- remote SchrodingerBridge/exp current Distinct5 and SADD lineage weights;
- remote SaMAM central step checkpoints;
- remote TokenizerClean exp tokenizer/AAAI chains;
- TokenizerClean special SA-SWD paths.

Retained because they are not checkpoint cleanup targets:

- data roots;
- latent backend caches;
- CLIP/VAE/ArtFID/LPIPS dependencies;
- feature caches;
- fake placeholders.

### Not safe to delete without a new policy

The following need explicit approval and a written retention rule before deletion:

- TokenizerClean `SchrodingerBridge/exp` thinning;
- remote main `SchrodingerBridge/exp` epoch thinning;
- remote SaMAM central curve thinning;
- remote latent backend cache pruning;
- remote `experiments` and `Cycle-NCE` archive/rar removal;
- placeholder-only deletion of `fake_eval_checkpoint.pt`.

## Timing evidence rules

Rules used in the timing CSV:

- preserve original units;
- do not convert training time to seconds unless the source already uses seconds;
- blank unknown values;
- do not promote dry-run `0.000` / `0.001` placeholder values;
- separate generation-only timing from full-eval timing;
- mark runtime-anomalous rows as quality-only when the source says they are not admissible speed evidence.

Strong timing sources:

- local `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`;
- local `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`;
- local `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`;
- local `Related_Works/results/metrics_summary/timing_summary.csv`;
- remote TokenizerClean `aaai2027_master_experiment_log.csv`;
- remote TokenizerClean SA-SWD and endpoint metric README packets.

Known timing traps:

- `lambda_grid` and `step_count_sweep` are dry-run placeholders.
- SA-SWD random arm completed, but its runtime is anomalous and should be quality-only.
- No repo-local DisDict 512 timing evidence was found.
- Some legacy baseline timing rows remain smoke/failed/unfair and require evidence-grade filtering.

## Missing methods or gaps

Gaps still unresolved:

- DisDict 512 timing evidence not found locally or in the opened remote surfaces.
- Several legacy baseline methods have metrics but incomplete or mixed-quality train/infer timing.
- SaMST has current Distinct5 point timing, but not a fully matched time-to-parity curve equivalent to the LBM/SaMAM framing.
- TokenizerClean H-family execution-alignment packet is blocked by missing H e1 payload; L e1 is documented as a successor family, not a same-family fallback.
- Remote archive deletion policy is not established for old `experiments`, `Cycle-NCE` rar/zips, or latent backend caches.

## Final working conclusion

The repository is cleaner than before, but "clean" cannot mean "no weight-like files". The correct state is:

- non-mainline checkpoint cleanup already landed for the explicit local and remote targets;
- current evidence, baseline evidence, cache/data, and dependencies remain;
- the remaining disk-heavy areas are mostly current evidence or data/cache, not obvious trash;
- further deletion should be a second retention-policy pass, not a broad `*.pt` sweep.
