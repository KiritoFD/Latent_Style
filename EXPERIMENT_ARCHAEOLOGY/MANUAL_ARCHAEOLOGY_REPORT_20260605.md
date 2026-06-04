# Manual Experiment Archaeology Report - 2026-06-05

Scope: manual follow-up pass for local `G:\GitHub\Latent_Style`, remote `I:\Github\Latent_Style`, and remote `I:\Github\Latent_Style_TokenizerClean`.

Write scope: only `EXPERIMENT_ARCHAEOLOGY/**`. No paper `tex/pdf` files, source files, or unrelated dirty files were edited.

## What changed in this pass

- Added remote per-directory timing rows for `SchrodingerBridge/exp` B44/A-E, J/K/L/M, path kinetic H base/k025/k000, and SADD repro.
- Added `manual_evidence_log_20260605.csv` as a direct opened-file/opened-directory log.
- Extended `manual_directory_classification_20260605.csv` with the specific remote directories opened after the script-only pass was rejected.
- Confirmed the exact retained remote `SchrodingerBridge/exp` weight total remains `101` files and `5945.063 MB`; the groups are current or lineage evidence, not deletion targets in this pass.

## Manual method

This pass used scripts only as navigation aids and validation guards. Conclusions were based on opening the actual directories and source files:

- directory listings for root, run roots, `logs`, `full_eval`, and checkpoint files;
- training CSV tails or full CSVs when a run had split logs;
- `summary.json` files for `timings_sec.wall_total`;
- docs and master logs that define claim safety and keep/delete policy;
- post-delete checks filtered by exact extension, not by broad path names.

The key correction from the failed broad listing is that Windows `-Include` over recursive trees can include unexpected paths if used loosely. The final weight checks use exact extension matching against `.pt`, `.pth`, `.ckpt`, and `.safetensors`.

## Local tree

### Timing sources opened

- `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `Related_Works/results/metrics_summary/timing_summary.csv`

These confirm the current safe timing split:

- Distinct5-512 has the strongest current timing surface.
- WikiArt512 has both generation-only and full-eval timing, and those must not be mixed.
- Legacy strict-750 rows need evidence grades because some rows are smoke, unfair, or estimated.
- No repo-local `DisDict 512` timing evidence was found.

### Post-delete local checks

The following local cleanup targets were opened again after deletion:

- `Related_Works/runs/cut_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5_smoke/checkpoints`
- `Related_Works/final_works/trial_0016`
- `Related_Works/final_works/trial_0019`
- `Related_Works/final_works/trial_0044`
- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync`

Result: no remaining `.pt/.pth/.ckpt/.safetensors` files in those checked deletion targets. The retained evidence is logs, summaries, generated images, metrics, and configs.

### Cycle-NCE local check

Opened:

- `Cycle-NCE/eval_cache`
- `Cycle-NCE/summary/summary_aggregate.csv`
- `Cycle-NCE/weight_exp4_latent_adain_swd60_tv00_id40_r16_e60/logs/training_20260322_044742.csv`
- `Cycle-NCE/weight_exp4_latent_adain_swd60_tv00_id40_r16_e60/full_eval`

Finding:

- Local residual `.pt` files are only `eval_cache/ref_feats_*.pt`.
- `weight_exp4...` retains config, code snapshot, training log, and full_eval logs/summaries, but no local training checkpoint weight.
- The family-level timing row remains representative historical evidence, not a paper-facing single result.

## Remote `I:\Github\Latent_Style`

### Root

Opened root listings:

- `I:\Github`
- `I:\Github\Latent_Style`
- `I:\Github\Latent_Style\SchrodingerBridge`
- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results`
- `I:\Github\Latent_Style\Cycle-NCE`

Important boundary:

- `I:\Github` also contains `26AI-H` and `26AI-H.zip`. These are outside the Latent_Style repo scope and were not cleaned or included as Latent_Style conclusions.

### Remote SchrodingerBridge exp

Exact retained weight total:

- `101` files
- `5945.063 MB`

Opened and classified groups:

- `aaai2027_longer_train_f_seed42_b44_e8`: current longer-train F audit surface; 8 weights retained.
- `aaai2027_longer_train_k_seed42_b44_e8`: current longer-train K audit surface; 8 weights retained.
- `distinct5_512_ema_baseline_direct_atom_residual_b44_remote`: closed B44 ablation; 8 weights retained; timing added.
- `distinct5_512_ema_baseline_direct_atom_residual_b40_remote` and `b48_remote`: header-only/incomplete starts; no timing promoted.
- `distinct5_512_ema_variant_a_class_prototypes_b44_remote`: closed A ablation; timing added.
- `distinct5_512_ema_variant_b_global_vq_b44_remote`: closed B ablation; timing added.
- `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote`: closed C ablation; timing added.
- `distinct5_512_ema_variant_d_vq_content_guided_b44_remote`: closed D ablation; timing added.
- `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote`: closed E ablation; timing added from two training logs.
- `distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote`: config/log/src only on remote; evidence exists locally via synced full_eval and docs.
- `distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3_b44_remote`: config/log/src only; no timing invented.
- `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`: config/log/src only on remote; evidence exists locally via synced full_eval and docs.
- `distinct5_512_ema_variant_i_dual_target_mix_queue_e3_b44_remote`: config/log/src only; no timing invented.
- `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote`: closed 3-epoch compact ablation; timing added.
- `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`: current compact K anchor at e1; e3 timing added as trajectory/audit row.
- `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`: closed 3-epoch compact ablation; timing added.
- `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote`: closed 3-epoch compact ablation; timing added.
- `aaai2027_path_kinetic_h_base_seed42_b44`: path kinetic packet; timing added.
- `aaai2027_path_kinetic_h_base_seed42_b44_k025`: path kinetic packet; timing added.
- `aaai2027_path_kinetic_h_base_seed42_b44_k000`: path kinetic packet; timing added.
- `aaai2027_path_stability_probe_h_base_seed42_b44_e1`: path geometry probe; not train/infer timing.
- `sadd_repro_38f_8ep_20260528_225252`: historical SADD reproduction; train timing added.
- `sadd_exact_e3_saddsrc_8ep_20260528_231954`: historical SADD exact-source run; no usable train/eval timing added.

No new remote `SchrodingerBridge/exp` deletion was performed in this pass. The retained weights are either current Distinct5 evidence, current ablation evidence, path-kinetic evidence, or historical reproduction gates. Deleting them now would require a separate checkpoint-retention policy rather than a broad non-mainline sweep.

### Remote Related_Works baseline

Opened:

- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`
- its `step_checkpoints`
- its `segmented.log`
- nested weight locations outside `step_checkpoints`

Post-delete result:

- Only `19` central step checkpoints remain under this baseline tree.
- Exact retained size is `5242.074 MB`.
- No non-central nested `.pt/.pth/.ckpt/.safetensors` weights remain under the authoritative diag root.

The broad image-heavy output from one failed check is ignored; it was caused by imprecise filtering and was replaced by exact extension filtering.

### Remote Cycle-NCE

Opened:

- `CONFIG_STATUS.md`
- `Experiment_Summary.json`
- `full_eval_summary_complete.csv`
- `experiment_status.json`
- `src` weight payload locations
- post-delete extension-filtered scan

Result:

- Six historical non-mainline `src` `.pt` payloads were already deleted and recorded.
- Remaining 37 weight-like files are eval/pretrained/cache/venv dependency files.
- Archives and compressed historical payloads remain. They were not deleted because they are not simple non-mainline checkpoints and may contain recoverable lineage evidence.

## Remote `I:\Github\Latent_Style_TokenizerClean`

Opened:

- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- `SchrodingerBridge/docs/experiments/2026-06-03-timing-artifact-prune.md`
- representative current/negative evidence dirs under `SchrodingerBridge/exp`
- special-character SA-SWD dirs under `SchrodingerBridge`

Key findings:

- TokenizerClean is not trash. It is a current AAAI2027 clean/snapshot worktree with current evidence, negative closure packets, and tokenizer/WikiArt lineage.
- Flow-loss trio is invalidated by config audit and must remain audit-only timing.
- Endpoint-metric trio is reviewed negative closure evidence, not a mainline replacement.
- Tokenizer localization stylebranch/executoronly packets are current evidence and retain checkpoints.
- SA-SWD semantic/random successful runs are in `exp<special-char>saswd...`; the char after `exp` has code `61532`. Normal `exp/saswd_*` dirs only contain launch/log files from failed or empty resume surfaces.
- SA-SWD random runtime is formally inadmissible as normal speed evidence; keep only as quality/runtime anomaly evidence.

TokenizerClean retained weight groups are large and numerous, especially tokenizer and WikiArt chains. This pass did not thin them because the correct unit of deletion is a tokenizer-lineage policy, not a generic checkpoint sweep.

## Cleanup state

Cleanup already recorded in CSV:

- Local deletion ledger: `cleanup/manual_deleted_checkpoints_20260605.csv`
- Remote deletion ledger: `cleanup/remote_manual_deleted_checkpoints_20260605.csv`

Current verified cleanup status:

- Local checked deletion targets are clean.
- Remote SaMAM diag has only central step checkpoints.
- Remote Cycle-NCE has no remaining deleted-target `src` payloads.
- Remote phase-space sweep directories have zero remaining weights after earlier deletion.

No ambiguous current-evidence checkpoint was deleted in this follow-up pass.

## Remaining gaps

- `LBM H e1/e2`: still no retained targetwise ArtFID packet.
- `K-longer e5..e8`: summaries exist, but no `aggregate_targetwise_artfid.json`.
- `SaMST e15`: no same-run-root pure inference timing.
- `SaMAM step 2250`: no same-scope pure inference timing.
- `SADD exact`: opened and retained, but training CSV does not expose a clean `epoch_time_sec` field and full_eval summaries do not expose usable wall timing.
- `TokenizerClean` tokenizer/WikiArt chains: large retained checkpoint surface needs a separate lineage-aware thinning pass.
- `26AI-H`: visible on remote I drive but outside this Latent_Style archaeology scope; not cleaned or indexed as a Latent_Style result.
- Archives such as `.rar`, `.zip`, and `.7z` were not unpacked or deleted in this pass.
