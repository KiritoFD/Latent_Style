# Manual Remote SchrodingerBridge Exp Walkthrough - 2026-06-05

Scope: remote `I:\Github\Latent_Style\SchrodingerBridge\exp` over SSH:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

Generated top-level inventory:

- `EXPERIMENT_ARCHAEOLOGY/manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`

Post-pass update:

- This document records the initial top-level/manual-sample pass.
- A later epoch-level pass supersedes the cleanup state here:
  `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md`.
- That later pass deleted 84 remote `SchrodingerBridge/exp` checkpoint files
  totaling `4961.604 MB` and verified 17 remaining retained checkpoints
  totaling `983.457 MB`.

This pass was not treated as proof by counting alone. I opened the remote top-level README, key configs, representative training logs, full-eval summaries, and weight epoch lists for the current weighted families.

## Top-Level Result

The remote `SchrodingerBridge/exp` tree currently has:

- `124` top-level entries in the inventory CSV.
- `17` top-level directories containing weight-like files.
- `101` weight-like files.
- `5945.064 MB` total weight-like size.

The 101 weight files are concentrated in current or lineage evidence directories:

- `aaai2027_longer_train_f_seed42_b44_e8`: 8 epoch weights, `538.077 MB`.
- `aaai2027_longer_train_k_seed42_b44_e8`: 8 epoch weights, `541.619 MB`.
- `aaai2027_path_kinetic_h_base_seed42_b44`: 3 epoch weights, `201.779 MB`.
- `aaai2027_path_kinetic_h_base_seed42_b44_k000`: 3 epoch weights, `201.778 MB`.
- `aaai2027_path_kinetic_h_base_seed42_b44_k025`: 3 epoch weights, `201.778 MB`.
- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote`: 8 epoch weights, `348.725 MB`.
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote`: 8 epoch weights, `398.459 MB`.
- `distinct5_512_ema_variant_b_global_vq_b44_remote`: 8 epoch weights, `531.141 MB`.
- `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote`: 8 epoch weights, `403.162 MB`.
- `distinct5_512_ema_variant_d_vq_content_guided_b44_remote`: 8 epoch weights, `538.070 MB`.
- `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote`: 8 epoch weights, `538.073 MB`.
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote`: 3 epoch weights, `201.778 MB`.
- `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`: 3 epoch weights, `203.107 MB`.
- `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`: 3 epoch weights, `203.107 MB`.
- `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote`: 3 epoch weights, `203.111 MB`.
- `sadd_exact_e3_saddsrc_8ep_20260528_231954`: 8 epoch weights, `345.643 MB`.
- `sadd_repro_38f_8ep_20260528_225252`: 8 epoch weights, `345.657 MB`.

No deletion was performed in this remote pass. These weighted directories are current Distinct5/AAAI2027 packets or historical SADD lineage evidence. They require a separate retention policy before thinning.

## Opened Remote Sources

Remote organization README opened:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\README.md`

The README says the tree was reorganized on `20260526_051909`, with:

- `vae_backend/`: active VAE backend runs and status outputs.
- `inference/`: inference parameter sweeps.
- `frontier/`: frontier/patch/stagewise sweeps.
- `diagnostics/`: diagnostic probes.
- `diffeomorphic_tangent_sweep/`: retained because scripts use it as the `t01` base config.
- archived legacy clutter moved to `I:\Github\Latent_Style\SchrodingerBridge\archives\exp_archive_20260526_051909`.

Representative configs opened:

- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote/config.json`
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote/config.json`
- `aaai2027_longer_train_f_seed42_b44_e8/config.json`
- `aaai2027_longer_train_k_seed42_b44_e8/config.json`
- `sadd_exact_e3_saddsrc_8ep_20260528_231954/config.json`
- `sadd_repro_38f_8ep_20260528_225252/config.json`

Representative logs and summaries opened:

- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote/logs/training_20260602_043649.csv`
- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote/full_eval/epoch_0008/summary.json`
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote/logs/training_20260602_050656.csv`
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote/full_eval/epoch_0008/summary.json`
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote/logs/training_20260602_102114.csv`
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote/full_eval/epoch_0003/summary.json`
- `aaai2027_longer_train_f_seed42_b44_e8/logs/training_20260603_223721.csv`
- `aaai2027_longer_train_f_seed42_b44_e8/full_eval/epoch_0008/summary.json`
- `aaai2027_longer_train_k_seed42_b44_e8/logs/training_20260603_233745.csv`
- `aaai2027_longer_train_k_seed42_b44_e8/full_eval_artfid/epoch_0008/summary.json`
- `sadd_exact_e3_saddsrc_8ep_20260528_231954/logs/training_20260528_232042.csv`
- `sadd_exact_e3_saddsrc_8ep_20260528_231954/full_eval/epoch_0008/summary.json`
- `sadd_repro_38f_8ep_20260528_225252/logs/training_20260528_225502.csv`
- `sadd_repro_38f_8ep_20260528_225252/full_eval/epoch_0008/summary.json`

Large non-weight evidence families opened by file/sample listing:

- `vae_backend`: 251463 files, 1559 dirs, 350 summary JSONs, 421 logs, 0 weights.
- `inference`: 53466 files, 145 dirs, 71 summary JSONs, 0 weights.
- `frontier`: 1902 files, 416 dirs, 8 summary JSONs, 84 logs, 0 weights.
- `tokenizer`: 9455 files, 90 dirs, 12 summary JSONs, 17 logs, 0 weights.
- `representation`: 9609 files, 122 dirs, 36 summary JSONs, 19 logs, 0 weights.

## Timing Evidence Opened

The training logs include timing columns such as:

- `data_time_sec`
- `forward_time_sec`
- `backward_time_sec`
- `optimizer_time_sec`
- `compute_time_sec`
- `epoch_time_sec`
- `samples_seen`
- `samples_per_sec`
- `cuda_peak_allocated_gb`
- `cuda_peak_reserved_gb`

Opened examples:

- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote`: epoch 8 `epoch_time_sec=62.23699736595154`, `samples_seen=4972`, `samples_per_sec=79.8881727980031`, `cuda_peak_allocated_gb=8.774822235107422`, `cuda_peak_reserved_gb=9.09765625`.
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote`: epoch 8 `epoch_time_sec=62.34275555610657`, `samples_seen=4972`, `samples_per_sec=79.75265057902922`, `cuda_peak_allocated_gb=8.802279949188232`, `cuda_peak_reserved_gb=9.134765625`.
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote`: epoch 3 `epoch_time_sec=63.49859118461609`, `samples_seen=4972`, `samples_per_sec=78.30094978870295`, `cuda_peak_allocated_gb=8.988595008850098`, `cuda_peak_reserved_gb=9.193359375`.
- `aaai2027_longer_train_f_seed42_b44_e8`: epoch 8 `epoch_time_sec=67.0106086730957`, `samples_seen=4972`, `samples_per_sec=74.19720695651917`, `cuda_peak_allocated_gb=8.867725372314453`, `cuda_peak_reserved_gb=9.15234375`.
- `aaai2027_longer_train_k_seed42_b44_e8`: epoch 8 `epoch_time_sec=64.67221856117249`, `samples_seen=4972`, `samples_per_sec=76.87999747986161`, `cuda_peak_allocated_gb=8.86839771270752`, `cuda_peak_reserved_gb=9.15234375`.
- `sadd_exact_e3_saddsrc_8ep_20260528_231954`: epoch 8 `epoch_time_sec=42.156864404678345`, `samples_seen=10260`, `samples_per_sec=243.37673460508603`.
- `sadd_repro_38f_8ep_20260528_225252`: epoch 8 `epoch_time_sec=41.12867784500122`, `samples_seen=10260`, `samples_per_sec=249.4609731600453`.

Full-eval timing examples opened:

- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote/full_eval/epoch_0008/summary.json`: `wall_total=94.8003261089998`, `eval_total=23.815491705999193`, `lancet_generation=5.006239374002689`.
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote/full_eval/epoch_0008/summary.json`: `wall_total=95.04837335200136`, `eval_total=24.12782883`, `lancet_generation=4.947571018994495`.
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote/full_eval/epoch_0003/summary.json`: `wall_total=95.3742954019981`, `eval_total=24.29322435600261`, `lancet_generation=5.035244303981017`.
- `aaai2027_longer_train_f_seed42_b44_e8/full_eval/epoch_0008/summary.json`: `wall_total=136.38432930499948`, `eval_total=40.42383317399981`, `lancet_generation=12.736426485002085`.
- `aaai2027_longer_train_k_seed42_b44_e8/full_eval_artfid/epoch_0008/summary.json`: `wall_total=105.18277635000004`, `eval_total=30.503645519999736`, `lancet_generation=6.7843992059947595`.

The older SADD summaries opened did not expose `timings_sec.wall_total` in the same structure, so inference timing was not filled for them.

## Cleanup Decision

No remote files were deleted in this pass.

Current deletion boundary:

- Keep all 101 remote `SchrodingerBridge/exp` weights for now because they belong to active Distinct5/AAAI2027 evidence or historical SADD lineage.
- Do not delete zero-weight high-file-count families like `vae_backend`, `inference`, `frontier`, `tokenizer`, or `representation`; they are summary/log/image/source evidence, not checkpoint clutter.
- Smoke directories have no weights and do not recover meaningful space.
- Any future deletion should be an epoch-thinning policy: define cited epoch(s), best epoch(s), and whether earlier epochs can be archived first.

## Gaps

- This pass did not inspect every nested subdirectory under `vae_backend`, `frontier`, `inference`, `tokenizer`, and `representation`; it opened top-level and representative samples only because each family is large and has no weights.
- Metric extraction from opened summaries was not normalized here; this document focuses on timing, weight retention, and cleanup safety.
- No remote cleanup was performed, so disk space was not reclaimed in this pass.
