# Manual Experiment Audit - 2026-06-05

Scope: manual archaeology pass for `G:\GitHub\Latent_Style` and remote `I:\Github\Latent_Style`. This file is intentionally separate from the earlier broad auto-index outputs because those outputs contain auto-classification artifacts and mojibake in some markdown summaries.

No paper `tex` or `pdf` files were edited. No unrelated dirty code files were staged or reverted.

## Outputs from this manual pass

- `EXPERIMENT_ARCHAEOLOGY/manual_directory_audit_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/manual_timing_evidence_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/manual_distinct5_remote_longer_train_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/manual_remote_phase_space_sweep_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/manual_remaining_weight_classes_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/MANUAL_REMOTE_PHASE_SPACE_SWEEP_20260605.md`
- `EXPERIMENT_ARCHAEOLOGY/cleanup/manual_deleted_checkpoints_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/cleanup/remote_manual_deleted_checkpoints_20260605.csv`

The existing broad files remain useful as indexes:

- `EXPERIMENT_ARCHAEOLOGY/final_master_experiments.csv`
- `EXPERIMENT_ARCHAEOLOGY/final_by_dataset/*.csv`
- `EXPERIMENT_ARCHAEOLOGY/final_timeline.csv`
- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`

Treat the broad files as search indexes. Treat the new manual files as the current checked evidence layer.

## Manual pass summary

### Distinct5-512

Checked local docs, local synced eval directories, and remote owner runs.

Key retained compact LBM points:

- `LBM F e1`: train `1.2161 min`, full `CLIP-S 0.6969145075`, full `LPIPS 0.3186446404`, transfer `CLIP-S 0.6643604031`, transfer `LPIPS 0.3245282069`.
- `LBM H e1`: train `1.2207 min`, full `CLIP-S 0.6973625582`, full `LPIPS 0.3213328829`; targetwise ArtFID packet is missing.
- `LBM H e2`: train `2.2656 min`, full `CLIP-S 0.6993825697`, full `LPIPS 0.3484066154`; targetwise ArtFID packet is missing.
- `LBM K e1`: train `1.2077 min`, full `CLIP-S 0.7009949656`, full `LPIPS 0.3622938977`, full targetwise ArtFID `157.1687`, transfer `CLIP-S 0.6711669415`.

Remote longer-train update:

- `K-longer` now has remote `full_eval_artfid/epoch_0005..0008/summary.json` and metrics.
- `K-longer e8` transfer `CLIP-S 0.6704904129`, transfer `LPIPS 0.4072183290`.
- `K-longer e8` remains worse than compact `K e1` on transfer CLIP-S and LPIPS.
- `K-longer e5..e8` do not have `aggregate_targetwise_artfid.json`; the matrix ArtFID values in `summary.json` are not promoted as official targetwise ArtFID.
- `F-longer e8` transfer `CLIP-S 0.6662981239`, transfer `LPIPS 0.3846649028`; this is also negative or neutral evidence versus compact F.

Baseline rows:

- `SaMST e5`: train `115.9750 min`, packet-bound generation `323.071 s / 750`, transfer `CLIP-S 0.6989188100`, transfer `LPIPS 0.6334999498`, transfer targetwise ArtFID `465.6860`.
- `SaMST e15`: train `347.2567 min`, metrics closed, same-run-root pure inference missing.
- `SaMAM step 2250`: train `458.5503 min`, manuscript-valid boundary, transfer `CLIP-S 0.5522515383`, transfer `LPIPS 0.3604523678`, transfer ArtFID `148.2059`.
- `SaMAM step 3000`: train `612.5845 min`, audit-only closed packet, not active manuscript path.

### WikiArt512

Checked `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` and related docs.

- Full eval anchor: external wall `210.67 s`, internal wall `206.7923 s`.
- Metrics anchor: `CLIP-S 0.7922978`, `content LPIPS 0.3550378`.
- This is an eval wall timing anchor, not pure model-generation timing.

### Related_Works legacy timing

Checked `Related_Works/results/metrics_summary/timing_summary.csv`.

Formal or usable rows:

- `AdaIN v32k`: train `9220.393 s`, strict-750 inference `9.281 s`.
- `AdaIN vgg19`: train `262.78 s`, strict-750 inference `9.098 s`.
- `SaMST strict 750`: inference `39.826 s`, training not preserved there.

Rows that must not be promoted:

- `CAST smoke*`: smoke or failed inference.
- `StyTr2 smoke*`: failed or tiny smoke; `smoke6` is only 5-image inference.
- `StyleID strict 750`: unfair because most targets were reused/copied.
- `AdaIN 4g` rows: timing/probe only or visually invalid output, depending on row.

### Legacy root experiments

Checked:

- `Cycle-NCE`
- `final_works`
- `lambda_grid`
- `step_count_sweep`
- root `exp`
- `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0`

Current classification:

- `Cycle-NCE` is a large historical archive with March lineage docs, many CSV/JSON summaries, and a small number of local `.pt` files. It was retained. It needs a separate, careful pass before any remaining weights are deleted.
- `final_works` contains performance evidence for CUT, SaMST, StarGAN, SDEdit, and LANCET trials. It has partial training timing in trial logs but no canonical inference timing table.
- `lambda_grid` and `step_count_sweep` status elapsed values are launcher placeholders around `0` or `0.001 s`; they are indexes, not valid timing.
- `S-add__K-1_C-0_W-20_Col-0` remains a mainline legacy gate. Its local training CSV appears header-misaligned, so training timing from that CSV is not used without revalidation.

### Remote root phase-space sweeps

Checked manually after the broad index pass:

- `I:/Github/Latent_Style/SchrodingerBridge/orthogonal_phase_space_sweep_60`
- `I:/Github/Latent_Style/SchrodingerBridge/high_tension_phase_space_sweep`

Evidence opened:

- Root `manifest.json` and `train_status.csv`.
- Each materialized `g*` run directory.
- Per-run `config.json`, `logs/training_*.csv`, `src/`, and actual checkpoint file.
- Targeted external lookup under `I:/Github/Latent_Style_TokenizerClean/SchrodingerBridge`, which found copied configs/scripts but no manuscript-facing metric packet.

Classification:

- Both families are legacy `../latent-256` five-style sweeps (`photo`, `Hayao`, `monet`, `vangogh`, `cezanne`).
- They preserve training-trajectory evidence but do not preserve `full_eval` outputs or quality metric summaries in the run roots.
- Their weights are not current mainline anchors. The config/log/src evidence is enough to reconstruct the historical run context.

Cleanup and timing:

- Deleted 13 `orthogonal_phase_space_sweep_60/*/epoch_0060.pt` weights, freeing `550.929223 MB`.
- Deleted 3 `high_tension_phase_space_sweep/*/epoch_00*.pt` weights, freeing `127.137391 MB`.
- Added per-run timing to `manual_timing_evidence_20260605.csv` and `manual_remote_phase_space_sweep_20260605.csv`.
- The legacy CSV headers are shifted. Timing is recorded from the third-from-end value in each row, which is the visible per-epoch wall time followed by `samples_seen` and `samples_per_sec`.
- `high_tension/g1_high_tension_base` has two logs; the 80-row `training_20260508_233525.csv` is used, not the adjacent 1-row fragment.
- `high_tension/g3_kinetic_vise` is interrupted: status `FAIL`, `train_rc=-1073741510`, and only partial `epoch_0040.pt` existed.

### Local Distinct5 ckptsync calibration

Checked manually after the local remaining-weight scan:

- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync`

Evidence opened:

- `config.json`: Distinct5-512 EMA latent training root, batch 16, `num_epochs=2`, `save_step_milestones=[350]`, `stop_after_global_steps=350`.
- `logs/training_20260605_061543.csv`: two training rows; total logged `epoch_time_sec=100.500922 s`.
- `generation_only_step_000350_timed/summary.json`: generation-only mode, 750 generated images, `wall_total=78.317428 s`, metrics intentionally skipped.
- `generation_only_step_000350_timed/images`: 750 files retained.

Classification:

- This is a local WSL 2-epoch/350-step timing calibration, not a mainline checkpoint anchor.
- Deleted `epoch_0001.pt`, `epoch_0002.pt`, and `step_000350.pt`.
- Freed `203.107647 MB`.
- Retained config, training CSV, generation summary, summary grid, and generated images.

## Cleanup performed

Local cleanup early-stage note:

- Deleted 6 local SaMAM Distinct5 stepalign calibration `.ckpt` files.
- Freed `1655.350814 MB`.
- Retained `train.log`, `gpu_samples.csv`, step/eval outputs, and directory structure.
- Audit file: `EXPERIMENT_ARCHAEOLOGY/cleanup/manual_deleted_checkpoints_20260605.csv`.

Remote cleanup early-stage note:

- Deleted 13 remote SaMAM duplicate segment, probe, or corrupt `.ckpt` files under the authoritative Distinct5 SaMAM run.
- Freed `3310.769927 MB`.
- Retained the main remote `step_checkpoints` directory with 19 checkpoints, plus `segmented.log` and `eval_curve`.
- Audit file: `EXPERIMENT_ARCHAEOLOGY/cleanup/remote_manual_deleted_checkpoints_20260605.csv`.

Continuation totals from the current manual deletion CSVs:

- `cleanup/manual_deleted_checkpoints_20260605.csv`: 875 local deleted checkpoint/weight rows, `46032.052664 MB` recorded.
- `cleanup/remote_manual_deleted_checkpoints_20260605.csv`: 2191 remote deleted checkpoint/weight rows after the phase-space append, `81459.438515 MB` recorded.
- The phase-space append specifically added 16 remote rows and `678.066614 MB`.
- The local ckptsync append specifically added 3 local rows and `203.107647 MB`.

Nothing ambiguous was deleted in this continuation pass.

## Remaining weight classes after cleanup

Post-cleanup remaining weights are indexed in `manual_remaining_weight_classes_20260605.csv`.

Local:

- Total remaining local `.pt/.ckpt/.pth/.safetensors`: 37711 files, `9813.856 MB`.
- Most are dataset latents or feature/eval caches: `latent-256`, `clip-feats-vitb32`, `eval_cache`, `SchrodingerBridge/scale/datasets`, and `SchrodingerBridge/datasets/horse2zebra`.
- Retained experiment checkpoints: `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` with 8 WikiArt512 timing-anchor weights.
- Retained metric dependency: `SchrodingerBridge/exp/video/.../art_inception.pth`.
- `Related_Works` remaining files are VGG/Inception/LPIPS dependencies and tiny `fake_eval_checkpoint` sentinels.
- `Cycle-NCE` local residual weights are kept pending a separate Cycle-NCE-focused pass.

Remote:

- `SchrodingerBridge/exp`: 101 retained current/mainline or current-ablation weights, `5945.063 MB`.
- `Related_Works/baseline_pipeline/results`: 19 retained authoritative SaMAM Distinct5 step checkpoints, `5242.074 MB`.
- `Related_Works/repos`: 2 retained VGG dependency files, `152.797 MB`.
- `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0`: 8 retained legacy mainline gate weights, `345.642 MB`.
- `Cycle-NCE`: 43 retained historical mixed/cache/dependency weights, `1097.427 MB`, pending a separate focused pass.
- Remote `orthogonal_phase_space_sweep_60` and `high_tension_phase_space_sweep` now have zero `.pt/.ckpt/.pth/.safetensors` files after this cleanup.

## Current gaps

- `LBM H e1/e2`: no retained targetwise ArtFID packet.
- `K-longer e5..e8`: metrics and summaries exist, but no `aggregate_targetwise_artfid.json`; matrix ArtFID means should not be treated as official targetwise ArtFID.
- `SaMST e15`: no same-run-root pure inference timing.
- `SaMAM step 2250`: no same-scope pure inference timing.
- `Cycle-NCE`: too large and historically mixed for checkpoint deletion without another focused pass.
- Remote phase-space sweeps: training time is recoverable, but inference timing and quality metrics are missing; treat them as lineage/timing history, not as benchmark rows.
- Local ckptsync calibration: generation timing exists, but metrics are intentionally skipped; treat as timing-only evidence.
- `remote_i_curated/remote_i_deleted_checkpoints.csv`: contains records outside `I:\Github\Latent_Style`, such as `I:\Github\26AI-H`; do not use it as a Latent_Style-only cleanup conclusion.
- Existing broad markdown summaries with mojibake should be replaced or ignored in final reporting.
