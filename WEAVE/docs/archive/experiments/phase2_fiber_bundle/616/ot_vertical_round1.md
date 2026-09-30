# OT Vertical Round 1

Date: 2026-06-16

## Scope

First strict 616 scratch launch.

Enabled mechanisms:

- structure-aware OT cost
- unbalanced Sinkhorn coupling
- pure vertical target projection
- training-time fast transfer eval
- white-box OT and fiber leakage probes

Held fixed:

- current backbone attention family
- current tokenizer family
- current solver family

This round is the foundation bundle only. It is intentionally the first 616 run because solver and tokenizer conclusions are not trustworthy until matching and target geometry are cleaned.

## Config

- [phase616_ot_vertical_scratch_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_ot_vertical_scratch_b8a2_e24.json)
- [run_phase616_ot_vertical_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_ot_vertical_round1.sh)
- [monitor_pid_gpu.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/monitor_pid_gpu.py)
- [build_full_eval_runtime_table.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_full_eval_runtime_table.py)

## Resolved Base

This lane is a strict 616 OT and target-geometry run on top of the retained `k070 -> k085_appalign` family, not a full post-purge clean-room rebuild.

Resolved active path:

- `solver_family = solver_i2sb`
- `transport_prediction_mode = endpoint`
- `endpoint_parameterization = absolute`
- `backbone_attention_family = legacy_semantic_crossattn`
- `tokenizer_family = pure_latent_spatial`
- `style_delta_mode = none`
- `proximal_mode = off`
- `output_appearance_alignment_mode = tokenizer_latent_affine`
- `objective_mode = i2sb_endpoint`
- `coupling_solver = sinkhorn_unbalanced`
- `coupling_structure_cost_mode = lowedge`
- `training_target_projection_mode = pure_vertical_flow`
- `w_content_lowpass_anchor = 0.0`
- `w_content_edge_anchor = 0.0`
- `cycle_consistency_weight = 0.0`
- `semantic_supervision_family = legacy_terminal_swd`
- `w_kinetic = 0.85`

Interpretation rule:

- positive evidence from this lane is evidence for `OT repair + pure_vertical_flow` on the retained legacy base
- it is not yet evidence for the fully purged 616 clean base, because output appearance alignment and the legacy semantic supervision stack are still present

## Expected Observability

Primary curve:

- transfer `CLIP-S`
- transfer `LPIPS`

White-box probes:

- `ot_target_gini`
- `ot_target_max_mass`
- `ot_structure_cost_mean`
- `ot_appearance_cost_mean`
- `base_structural_drift`
- `fiber_energy_ratio`
- `low_freq_leak`
- `target_base_shift`

Runtime probes:

- `avg_optimizer_step_time_sec`
- `epoch_time_sec`
- `samples_per_sec`
- `cuda_peak_allocated_gb`
- `cuda_peak_reserved_gb`
- `gpu_vram_used_gb_mean/peak`
- `gpu_util_mean/peak`
- `gpu_power_w_mean/peak`

## Runtime Artifacts

- training CSV:
  `exp/aaai2027_phase616_ot_vertical_scratch_b8a2_e24/logs/training_*.csv`
- per-checkpoint eval runtime table:
  `exp/aaai2027_phase616_ot_vertical_scratch_b8a2_e24/logs/full_eval_runtime.csv`
- launcher-side GPU sampler:
  `docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1/*.gpu_metrics.csv`
  `docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1/*.gpu_summary.json`

For the already-running scratch lane, do not restart only to pick up runtime logging. Use the standalone GPU sampler and the retrospective eval-runtime builder to complete the record.

Instrumentation caveat:

- this live process was launched before the final training-CSV schema extension landed
- therefore the current lane writes reliable `CLIP-S / LPIPS`, train loss, and runtime rows
- but the full 616 OT white-box columns are not guaranteed to be present in its existing `training_*.csv`
- the next matched lane must carry the finalized schema so `ot_target_gini`, `ot_target_max_mass`, `ot_plan_entropy`, `ot_target_mass_entropy`, `base_structural_drift`, `fiber_energy_ratio`, `low_freq_leak`, and tokenizer rank/cosine probes become first-class closure signals

## Closure Status

Stopped manually on 2026-06-16 before closure.

Reason:

- this lane was too slow for a first OT white-box diagnostic pass
- it was still on the retained legacy base rather than the strict `phase616` clean base
- the finalized probe schema landed mid-run, so the most important OT closure columns were incomplete

Decision:

- keep its partial evidence as a diagnostic-only record
- move the next OT step to the clean-base fast probe pair:
  - `lowedge` matched control
  - `self_affinity_gw` matched candidate
- do not promote or demote any OT conclusion from this lane alone

Last observed runtime state before stop:

- remote PID `36886`
- live GPU band about `4.27 GiB` VRAM with high util and about `123-124 W`
- current full-eval directories present: `epoch_0001`, `epoch_0002`, `epoch_0003`

Latest transfer evals captured so far:

| epoch | CLIP-S | LPIPS | full-eval wall |
|---|---:|---:|---:|
| 1 | 0.6906 | 0.3866 | 91.6 s |
| 2 | 0.6732 | 0.3657 | 27.2 s |
| 3 | 0.6783 | 0.3214 | 26.8 s |

Latest completed train-epoch runtime rows:

| epoch | loss | samples/s | epoch wall | CUDA alloc peak | CUDA reserved peak |
|---|---:|---:|---:|---:|---:|
| 1 | 5.4676 | 14.31 | 1320.2 s | 2.50 GiB | 3.01 GiB |
| 2 | 2.9076 | 14.09 | 1340.3 s | 3.16 GiB | 3.83 GiB |
| 3 | 2.3564 | 13.79 | 1370.0 s | 3.17 GiB | 3.68 GiB |

Current read:

- the lane showed that `pure_vertical_flow + lowedge + unbalanced Sinkhorn` can pull LPIPS down quickly
- it did not provide a clean enough answer on OT matching quality because probe coverage was partial and style had not stabilized
- the speed/runtime profile is not acceptable as the primary diagnostic lane

## Decision Rule

Promote only if:

- curve improves against current clean control band
- and OT hubness decreases
- and low-frequency leakage does not rise in a way that predicts late LPIPS failure
