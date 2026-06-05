# Manual Remote SchrodingerBridge Epoch Thinning - 2026-06-05

Scope: remote `I:\Github\Latent_Style\SchrodingerBridge\exp` on:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

This pass follows the earlier top-level inventory pass in
`MANUAL_REMOTE_SCHRODINGERBRIDGE_EXP_20260605.md`. The earlier pass counted and
sample-opened the weighted families; this pass opened the specific weighted
families one by one, wrote an epoch-level policy, deleted only policy-approved
checkpoint files, and wrote a deletion ledger.

## Output Files

- `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv`
  - 101 rows, one row per pre-cleanup remote checkpoint.
  - Each row records config path, ablation stage, dataset root, training log,
    train timing fields, summary path, transfer/full metrics, eval timing, and
    checkpoint size.
- `manual_remote_schrodingerbridge_epoch_thinning_policy_20260605.csv`
  - 101 rows, one keep/delete decision per checkpoint.
  - Each delete row records why the checkpoint is not a retained/cited/probe
    operating point.
- `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv`
  - 84 rows, one actual deletion result per removed checkpoint.
  - All rows are `deleted`; no missing/error rows.
- `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv`
  - 17 rows, post-delete remaining remote checkpoint list.

## Manual Sources Opened

Local documents opened before deciding the retention policy:

- `SchrodingerBridge/docs/reviews/aaai2027_writing_gate_R20260603O.md`
- `SchrodingerBridge/docs/experiments/2026-06-04-distinct5-aaai-evidence-pack.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-remote-sidecar-recovery-status.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-path-stability-launch-status.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-path-stability-protocol.md`
- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/lancet_runs.md`
- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/resolved_headline_config.md`
- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv`
- tokenizer-localization notes that cite the L-family epoch-1 anchor.

Remote files opened or parsed for the 17 weighted directories:

- `config.json` for each weighted directory.
- `logs/training_*.csv` or root `remote_train.log` where present.
- `full_eval/epoch_*/summary.json` and, for K-longer recovery, `full_eval_artfid/epoch_*/summary.json`.
- Remote process/GPU state before deletion.

This pass did not use file extension alone as a deletion rule.

## Pre-Cleanup State

- Weighted directories checked: 17.
- Checkpoint files before cleanup: 101.
- Checkpoint size before cleanup: 5945.061 MB from the epoch evidence CSV.
- Earlier top-level inventory had recorded the same class as about 5945.064 MB.

The weighted families were:

- `aaai2027_longer_train_f_seed42_b44_e8`
- `aaai2027_longer_train_k_seed42_b44_e8`
- `aaai2027_path_kinetic_h_base_seed42_b44`
- `aaai2027_path_kinetic_h_base_seed42_b44_k000`
- `aaai2027_path_kinetic_h_base_seed42_b44_k025`
- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote`
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote`
- `distinct5_512_ema_variant_b_global_vq_b44_remote`
- `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote`
- `distinct5_512_ema_variant_d_vq_content_guided_b44_remote`
- `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote`
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote`
- `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`
- `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`
- `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote`
- `sadd_exact_e3_saddsrc_8ep_20260528_231954`
- `sadd_repro_38f_8ep_20260528_225252`

The remote F/G/H/I short Distinct5 directories were also opened. They now contain
only `config.json` and no weights/summaries, so no deletion was possible there.

## K-Longer Recovery Update

The earlier sidecar note recorded K-longer eval recovery as active/pending.
This pass rechecked the remote state directly:

- `full_eval_artfid/epoch_0005..epoch_0008` now each contains `metrics.csv`,
  `summary.json`, and `summary_grid.png`.
- `tmux` status returned `DEAD`.
- No WSL `run_evaluation`, `aaai2027_longer_train_k`, or `distinct5_512`
  writer process was alive.
- GPU was idle at about `1038-1039 MiB / 12288 MiB`, `0%` utilization.

K-longer still failed the paper-entry gate:

- base K e1 transfer CLIP-S from the local table: `0.6711669415235519`.
- K-longer e5-e8 transfer CLIP-S: `0.6670102730890115`,
  `0.6693240416049957`, `0.6705300333599249`,
  `0.6704904128611089`.
- This does not clear the `+0.006` improvement rule.
- LPIPS worsened into the later epochs.

Therefore K-longer checkpoints were treated as negative-evidence clutter after
the summaries, metrics, grids, config, and training log were retained.

## Keep Policy

The final retained remote `SchrodingerBridge/exp` checkpoints are exactly:

| directory | kept epochs | reason |
|---|---:|---|
| `aaai2027_path_kinetic_h_base_seed42_b44` | `epoch_0001` | path-stability probe selected H_base e1 |
| `aaai2027_path_kinetic_h_base_seed42_b44_k000` | `epoch_0001` | path-stability probe selected H_k000 e1 |
| `aaai2027_path_kinetic_h_base_seed42_b44_k025` | `epoch_0001` | path-stability probe selected H_k025 e1 |
| `distinct5_512_ema_baseline_direct_atom_residual_b44_remote` | `epoch_0001`, `epoch_0008` | baseline reference operating points |
| `distinct5_512_ema_variant_b_global_vq_b44_remote` | `epoch_0008` | weak-retain B operating point |
| `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote` | `epoch_0002` | retained C operating point |
| `distinct5_512_ema_variant_d_vq_content_guided_b44_remote` | `epoch_0001` | retained D operating point |
| `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote` | `epoch_0001`, `epoch_0003` | strong-retain E operating points |
| `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote` | `epoch_0001` | headline style-anchor K point |
| `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote` | `epoch_0001` | tokenizer-localization/preflight anchor despite rejected status |
| `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote` | `epoch_0001` | one recorded M anchor |
| `sadd_exact_e3_saddsrc_8ep_20260528_231954` | `epoch_0007`, `epoch_0008` | only retained SADD exact full-eval summary epochs |
| `sadd_repro_38f_8ep_20260528_225252` | `epoch_0007`, `epoch_0008` | retained SADD repro full-eval summary epochs |

Post-delete verification found exactly 17 remaining checkpoint files totaling
`983.457 MB`.

## Deleted Policy Classes

The cleanup deleted 84 checkpoints totaling `4961.604 MB`.

| directory | deleted checkpoints | deleted MB | policy |
|---|---:|---:|---|
| `aaai2027_longer_train_f_seed42_b44_e8` | 8 | 538.080 | failed F-longer retention gate; summaries retained |
| `aaai2027_longer_train_k_seed42_b44_e8` | 8 | 541.616 | closed negative K-longer evidence; summaries/grids retained |
| `aaai2027_path_kinetic_h_base_seed42_b44` | 2 | 134.520 | probe uses e1; e2/e3 summaries retained |
| `aaai2027_path_kinetic_h_base_seed42_b44_k000` | 2 | 134.518 | probe uses e1; e2/e3 summaries retained |
| `aaai2027_path_kinetic_h_base_seed42_b44_k025` | 2 | 134.518 | probe uses e1; e2/e3 summaries retained |
| `distinct5_512_ema_baseline_direct_atom_residual_b44_remote` | 6 | 261.546 | e1/e8 retained; intermediates removed |
| `distinct5_512_ema_variant_a_class_prototypes_b44_remote` | 8 | 398.456 | rejected ablation; summaries retained |
| `distinct5_512_ema_variant_b_global_vq_b44_remote` | 7 | 464.751 | e8 retained |
| `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote` | 7 | 352.765 | e2 retained |
| `distinct5_512_ema_variant_d_vq_content_guided_b44_remote` | 7 | 470.813 | e1 retained |
| `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote` | 6 | 403.556 | e1/e3 retained |
| `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote` | 3 | 201.777 | rejected ablation; summaries retained |
| `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote` | 2 | 135.404 | K e1 retained |
| `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote` | 2 | 135.404 | L e1 retained |
| `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote` | 2 | 135.408 | M e1 retained |
| `sadd_exact_e3_saddsrc_8ep_20260528_231954` | 6 | 259.230 | e7/e8 retained as summary anchors |
| `sadd_repro_38f_8ep_20260528_225252` | 6 | 259.242 | e7/e8 retained as summary anchors |

## Safety Checks

Before deletion:

- Verified no live WSL `src/run.py`, `run_evaluation`, `aaai2027`, `distinct5_512`, or `sadd_` writer process.
- Verified GPU idle.
- Verified every delete candidate path matched
  `I:\Github\Latent_Style\SchrodingerBridge\exp\*.pt`.

During deletion:

- Each path was resolved with `Get-Item`.
- The resolved `FullName` had to remain under
  `I:\Github\Latent_Style\SchrodingerBridge\exp\`.
- The extension had to be `.pt`.
- The deletion ledger recorded `before_bytes`, `before_mb`, `last_write_time`,
  `after_exists`, reason, summary path, and training log path.

After deletion:

- `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv` reported
  84 `deleted` rows and no errors.
- `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv`
  reported exactly 17 remaining checkpoints.

No paper `.tex` or `.pdf` files were edited.

## Remaining Gaps

- Remote SaMAM central `step_checkpoints` were not deleted in this block. The
  hash audit showed `last*.ckpt` aliases are not duplicate hashes, and
  2500/2750 ArtFID repair evidence remains incomplete.
- Remote `Latent_Style_TokenizerClean` is still not thinned. It needs a
  citation graph before any destructive action.
- Non-weight high-file-count families such as `vae_backend`, `inference`,
  `frontier`, `tokenizer`, and `representation` were not deleted here; this
  pass targeted checkpoints only.
- K-longer now has ArtFID values in `summary.json`, but no separate
  `aggregate_targetwise_artfid.json` files were observed in the e5-e8
  `full_eval_artfid` roots. If a future paper packet requires those exact JSON
  files, regenerate or reuse from retained images/metrics rather than relying
  on deleted checkpoints.
