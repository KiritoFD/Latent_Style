# Manual Remote Phase-Space Sweep Audit - 2026-06-05

Scope: remote `I:\Github\Latent_Style\SchrodingerBridge\orthogonal_phase_space_sweep_60` and `I:\Github\Latent_Style\SchrodingerBridge\high_tension_phase_space_sweep`.

This note records a manual pass. Script output was used only to navigate and format repeated fields; each run directory was opened for root manifest/status, run config, training CSV, and checkpoint presence before deletion.

## Evidence Opened

- `orthogonal_phase_space_sweep_60/manifest.json`: lists 13 runs `g0` through `g12`, each with an `epoch_0060.pt` checkpoint and planned `full_eval/...` output path.
- `orthogonal_phase_space_sweep_60/train_status.csv`: all 13 rows are `OK`, `train_rc=0`, `checkpoint_exists=YES`.
- Each `orthogonal_phase_space_sweep_60/g*/`: opened `config.json`, `logs/training_*.csv`, `src/`, and the single `epoch_0060.pt`.
- `high_tension_phase_space_sweep/manifest.json`: lists g1-g8 planned runs, but only g1-g3 directories exist in this remote root.
- `high_tension_phase_space_sweep/train_status.csv`: g1/g2 are `OK`; g3 is `FAIL`, `train_rc=-1073741510`, `checkpoint_exists=NO`.
- Each `high_tension_phase_space_sweep/g*/`: opened `config.json`, `logs/training_*.csv`, `src/`, and the actual checkpoint file if present.
- External check: `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge` contains copied configs/scripts for these sweep families. No current `Latent_Style\SchrodingerBridge\docs` or `configs` manuscript-facing metric summary was found for these names.

## Classification

These are legacy May 8-9 phase-space sweeps on `../latent-256` with five styles: `photo`, `Hayao`, `monet`, `vangogh`, `cezanne`.

They are useful as historical training-trajectory evidence, but not as mainline reusable checkpoints:

- The run roots preserve config, source snapshot, and training CSVs.
- No `full_eval` directory or metric `summary.json` was present under either run root.
- Current manuscript/AAAI Distinct5 and WikiArt anchors live in later `SchrodingerBridge/exp/...` and `Related_Works/baseline_pipeline/...` directories, not these root sweeps.
- The checkpoints were therefore deleted after manual evidence capture, while configs/logs/manifests/source snapshots were retained.

## Timing Notes

The training CSV header is shifted for these legacy logs: the nominal `epoch_time_sec` column imports blank via `Import-Csv`, but the row still carries wall timing near the end of each line. The audit uses the third-from-end value per row, which aligns with visible per-epoch wall seconds and is followed by `samples_seen` and `samples_per_sec`.

For `high_tension/g1_high_tension_base`, two logs exist:

- `training_20260508_233247.csv`: 1 row, warm/interrupted fragment.
- `training_20260508_233525.csv`: 80 rows, selected as the run timing source.

## Per-Run Summary

| family | run | period | epochs | train_time_s | checkpoint deleted | retained evidence |
| --- | --- | --- | ---: | ---: | --- | --- |
| orthogonal | g0_universe_center | 2026-05-09 03:08:41 to 04:08:05 | 60 | 3562.601 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g1_absolute_release | 2026-05-09 04:08:19 to 05:07:40 | 60 | 3560.009 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g2_absolute_freeze | 2026-05-09 05:07:54 to 06:07:29 | 60 | 3572.826 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g3_gravity_black_hole | 2026-05-09 06:07:44 to 07:05:27 | 60 | 3462.183 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g4_gravity_vacuum | 2026-05-09 07:05:40 to 08:03:20 | 60 | 3457.994 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g5_midfreq_strangulation | 2026-05-09 08:03:33 to 09:01:12 | 60 | 3457.612 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g6_structure_amnesty | 2026-05-09 09:01:25 to 09:59:04 | 60 | 3457.507 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g7_flesh_stripping | 2026-05-09 09:59:16 to 10:28:41 | 60 | 1764.245 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g8_absolute_nailgun | 2026-05-09 10:28:53 to 11:26:33 | 60 | 3458.157 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g9_cryogenic_hard_match | 2026-05-09 11:26:46 to 12:24:27 | 60 | 3459.880 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g10_thermal_soft_collapse | 2026-05-09 12:24:41 to 13:22:26 | 60 | 3464.085 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g11_blind_men_slicing | 2026-05-09 13:22:39 to 14:20:10 | 60 | 3449.075 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| orthogonal | g12_limit_approximation | 2026-05-09 14:20:23 to 15:18:18 | 60 | 3473.644 | epoch_0060.pt | manifest, status, config, training CSV, src snapshot |
| high_tension | g1_high_tension_base | 2026-05-08 23:32:47 to 2026-05-09 00:54:31 | 80 | 4744.207 | epoch_0080.pt | manifest, status, config, two training CSVs, src snapshot |
| high_tension | g2_swd_nuke | 2026-05-09 00:54:45 to 02:15:32 | 80 | 4845.270 | epoch_0080.pt | manifest, status, config, training CSV, src snapshot |
| high_tension | g3_kinetic_vise | 2026-05-09 02:15:46 to 02:57:50 | 43 | 2713.465 | epoch_0040.pt | manifest, status, config, training CSV, src snapshot |

## Output Files Updated

- `manual_remote_phase_space_sweep_20260605.csv`: per-run manual index for this focused pass.
- `manual_timing_evidence_20260605.csv`: 16 timing rows added.
- `cleanup/remote_manual_deleted_checkpoints_20260605.csv`: 16 deletion rows added with pre-delete size and mtime.

## Residual Gaps

- No inference timing was found for these sweep roots.
- No quality metrics were found in the sweep roots; do not promote these runs as metric-bearing comparisons.
- g7 is much faster than sibling 60-epoch runs. It is recorded as logged and not normalized.
- g3 high-tension was interrupted; it is an incomplete training fragment with only an epoch 40 checkpoint.
