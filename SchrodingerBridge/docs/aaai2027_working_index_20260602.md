# AAAI 2027 Working Index

Updated: 2026-06-03

This file is the working entrypoint for the current AAAI 2027 push. It groups
the paper source, the experiment evidence, the dataset locations, and the
code paths that are expected to keep changing. The goal is to avoid
reconstructing project state from memory every time writing or experiments
resume.

## 1. Paper source of truth

Primary manuscript:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`

Rendered output:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`

Main framework figure:

- current paper target:
  - `SchrodingerBridge/aaai_submission/framework_lbm_main_v5.png`
- older vector / legacy figure surfaces kept for recovery:
  - `SchrodingerBridge/aaai_submission/framework_figure.pdf`
  - `SchrodingerBridge/aaai_submission/framework_lbm_main.png`

Distinct5 Pareto figure for the paper:

- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.pdf`
- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.png`

Figure generation scripts:

- `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_pareto.py`
- `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_full_transfer_pareto.py`
- `SchrodingerBridge/aaai_submission/scripts_collect_distinct5_full_transfer_points.py`
- `SchrodingerBridge/aaai_submission/scripts_gen_aaai2027_figures.py`
- `SchrodingerBridge/aaai_submission/figures/README.md`

## 2. Current writing direction

High-level target:

- `SchrodingerBridge/goal.md`

Current paper update plan:

- `SchrodingerBridge/docs/experiments/2026-06-02-aaai2027-paper-update-plan.md`

Continuous reviewer lane:

- `SchrodingerBridge/docs/reviews/aaai2027_review_protocol.md`
- `SchrodingerBridge/docs/reviews/aaai2027_review_registry.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_review_score_log.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_reviewer_roster.md`
- `SchrodingerBridge/docs/reviews/aaai2027_review_packet_template.md`
- `SchrodingerBridge/docs/reviews/aaai2027_review_consensus_20260602_r2.md`
- `SchrodingerBridge/docs/reviews/aaai2027_review_consensus_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_claim_safety_memo_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_agent_ops_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_weak_reject_pressure_memo_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_rewrite_hit_list_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_boundary_alignment_pass_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_boundary_followup_overclaims_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_gate_b_runtime_anomaly_policy_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_gate_status_and_next_experiment_priority_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_gate_c_time_to_parity_audit_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_post_tightening_recheck_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_tokenizer_timing_reaudit_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_weak_reject_rerun_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_gate_b_pair_reaudit_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_l_family_postlanding_tex_reread_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_l_family_postedit_tex_check_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_adversarial_gate_refresh_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_tokenizer_execution_alignment_l_family_reread_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_tokenizer_l_family_paper_gate_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_tokenizer_probe_successor_family_reread_20260603.md`
- `SchrodingerBridge/docs/reviews/distinct5_idt_noop_claim_boundary_20260603.md`
- `SchrodingerBridge/docs/reviews/endpoint_metric_claim_boundary_20260603.md`
- `SchrodingerBridge/docs/reviews/endpoint_metric_review_gate_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_claim_matrix_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_claim_evidence_matrix_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_code_vs_output_geometry_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_localization_claim_boundary_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_localization_outcome_claim_map_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_execution_alignment_l_family_theory_boundary_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_probe_checkpoint_reselection_policy_20260603.md`
- `SchrodingerBridge/docs/reviews/tokenizer_representation_theory_queue_20260603.md`

Related-work / citation gap notes:

- `SchrodingerBridge/docs/references/literature_intel_memo_20260603.md`
- `SchrodingerBridge/docs/references/related_work_gap_candidates_20260603.md`
- `SchrodingerBridge/docs/references/evaluation_pathology_noop_memo_20260603.md`
- `SchrodingerBridge/docs/references/related_work_and_intro_gap_recheck_20260603.md`
- `SchrodingerBridge/docs/references/related_work_framing_patch_priorities_20260603.md`
- `SchrodingerBridge/docs/references/tokenizer_representation_related_work_refresh_20260603.md`

Core evaluation warning about CLIP-style / no-op:

- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/README.md`

Cross-dataset comparison report:

- `SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.pdf`
- `SchrodingerBridge/docs/experiments/2026-06-03-exp-surface-classification.md`
- `SchrodingerBridge/docs/experiments/aaai2027_experiment_logging_contract_20260603.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-exploratory-image-prune.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-smoke-surface-prune.md`

Cleanup / worktree hygiene notes:

- `SchrodingerBridge/docs/cleanup/worktree_triage_20260603.md`
- `SchrodingerBridge/docs/cleanup/paper_surface_audit_20260603.md`

## 3. Experiment families to cite separately

Do not mix these into one undifferentiated result pool.

### A. Historical strict-750

Use for the main legacy benchmark table against SaMST / S2WAT / StyleID /
AdaIN-family baselines.

Key evidence:

- `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json`
- `SchrodingerBridge/archives/old_root_files/training_times_documentation.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/selected_style_metrics_historical_merged.csv`

### B. WikiArt512 five-style convergence reference

Use for convergence behavior and the metric-hacking discussion, not as the
primary paper table.

Key evidence:

- `SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.md`
- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`

### C. Distinct5-512 stress benchmark

This is the current main 512-based stress benchmark.

Key evidence:

- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/`
- `SchrodingerBridge/docs/experiments/2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.md`
- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/README.md`

Key tables:

- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv`
- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.csv`

### D. AAAI 2027 claim-closing ablations

Use these only for bounded claim closure, not for the main quality table.

Key evidence:

- `SchrodingerBridge/docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-flow-loss-metric-ablation/repaired_endpoint_metric_ablation_packet_20260603.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-flow-loss-metric-ablation/repaired_endpoint_metric_launch_manifest_20260603.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-saswd-axis-ablation/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-execution-alignment-protocol.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-execution-alignment/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-execution-alignment-l-family/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization-probe-protocol.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/launch_manifest_20260603.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-path-stability-protocol.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/distinct5_time_to_parity_points.csv`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_clip_style.pdf`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_lpips.pdf`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_delta_idt.pdf`
- `SchrodingerBridge/docs/reviews/endpoint_metric_claim_boundary_20260603.md`
- `SchrodingerBridge/docs/reviews/endpoint_metric_review_gate_20260603.md`

## 4. Datasets and split roots

Unified dataset roots under the repo:

- `Dataset/legacy256_overfit50`
- `Dataset/wikiart512_5style`
- `Dataset/distinct5_512`

Legacy in-project dataset helper root:

- `SchrodingerBridge/datasets/horse2zebra`
  - legacy local sample only
  - not part of the current AAAI 2027 benchmark surface

SchrodingerBridge-side references:

- `SchrodingerBridge/docs/experiments/comparison_20260602/README.md`
- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/README.md`

Remote / local raw-image and latent provenance still matters for reproducibility:

- Distinct5 raw images / latents are documented in
  `SchrodingerBridge/docs/experiments/2026-06-02-distinct5-512-lancet-representation-summary.zh.md`

## 5. Code paths that matter for the next experiment cycle

Model / representation:

- `SchrodingerBridge/src/model.py`
- `SchrodingerBridge/src/style_tokenizer.py`
- `SchrodingerBridge/src/lancet_runtime.py`

Loss / transport:

- `SchrodingerBridge/src/losses.py`
- `SchrodingerBridge/src/ot_cost.py`

Training / evaluation:

- `SchrodingerBridge/src/trainer.py`
- `SchrodingerBridge/src/utils/run_evaluation.py`
- `SchrodingerBridge/tools/eval_selected_style_metrics.py`

Config family currently used for tokenizer exploration:

- `SchrodingerBridge/configs/tokenizer_t01_*`
- `SchrodingerBridge/configs/README.md`
- `SchrodingerBridge/configs/archive/20260603_local_wsl_wikiart512/`
- `SchrodingerBridge/configs/archive/20260603_refactor_legacy/`

## 6. Remote 3060 usage

Formal experiments should prefer the remote 3060 machine when they are more
than a smoke test.

Current policy reminders:

- Prefer remote formal runs for Distinct5 / tokenizer / paper-facing evidence.
- Keep formal training around the established VRAM target rather than low-VRAM
  smoke settings.
- Use local runs for fast smoke checks, figure generation, and analysis scripts.

## 7. Temporary and runtime paths that should stay out of version control

The following are local/runtime support paths, not durable research artifacts:

- `Dataset/`
- `SchrodingerBridge/_codex_tmp/`
- paper build byproducts such as `aaai_submission/*.aux` and `*.log`

Permanent artifacts should be promoted into `docs/`, `aaai_submission/figures/`,
or stable config/script locations before being referenced in the paper.

## 8. Immediate gaps before the next writing pass

1. The repaired endpoint-metric trio is now negatively closed and reviewed; its
   claim boundary must stay narrow and must not be re-expanded into a broad
   latent-metric theorem.
2. The fixed-base semantic-vs-random SA-SWD axis ablation is now negatively
   closed rather than open: both matched arms landed, but the random arm is
   admissible only as quality-only evidence because of severe runtime anomaly,
   and the landed pair does not support a positive semantic-axis superiority
   claim.
3. Time-to-parity plots should be cited with explicit scope and timing
   definitions, not only as prose summaries.
4. Worktree cleanup should separate durable paper/docs changes from local temp
   files and data mirrors before broader commits.
5. After Gate B negative closure, the tokenizer code-to-execution alignment
   lane is no longer the only live mechanism question. The original reviewed
   `H`-family packet remains blocked on payload recovery, while the landed
   `L`-family successor and the matched tokenizer-localization packet both now
   define bounded representation-side evidence rather than open launch tasks.
6. The next highest-priority remote mechanism closure lane is the matched
   Distinct5 path-stability / weakened-kinetic packet documented in:
   - `SchrodingerBridge/docs/experiments/2026-06-03-path-stability-protocol.md`
   - `SchrodingerBridge/configs/aaai2027/path_kinetic_h_base_seed42_b44_base.json`
   - `SchrodingerBridge/configs/aaai2027/path_kinetic_h_base_seed42_b44_k025.json`
   - `SchrodingerBridge/configs/aaai2027/path_kinetic_h_base_seed42_b44_k000.json`
   This is the cleanest remaining route to support the bounded kinetic /
   path-energy story with current Distinct5 evidence.
7. The matched Distinct5 tokenizer-localization packet is now a completed
   evidence packet, not the live blocking remote lane. Its current working
   truth is in:
   - `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/README.md`
   - `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/readout_20260603.csv`
   Keep its claims localized to the `L`-family frozen-direction result.
8. Every major checkpoint should trigger a fresh three-lane review round and a
   registry entry, rather than relying on a single frozen review memo.
9. Config and archive surfaces should stay indexed so the paper-facing rerun
   path is obvious and local timing/refactor residue does not pollute the main
   experiment surface.
10. Manuscript tightening around the landed `L`-family tokenizer evidence has
   now been applied and reviewer-checked once; the next highest-value action is
   a stronger mechanism-side experiment rather than another local wording pass.
11. The earlier legacy256 tokenizer-`t01` localization route is superseded and
   must not be relaunched as the active paper-facing packet. The current live
   remote mechanism-prep note for the next packet is now the Distinct5
   path-stability protocol rather than another `t01` restart.
