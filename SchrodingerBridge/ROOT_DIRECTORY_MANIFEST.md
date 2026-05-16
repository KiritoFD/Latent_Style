# Root Directory Manifest

Reviewed on `2026-05-16`.

This file records what each top-level directory in `SchrodingerBridge/` is for, whether it is active, and whether it should stay visible at the root.

## Active Core

| Path | Role | Keep at root |
|---|---|---|
| `src/` | Main implementation | Yes |
| `maths/` | Theory, derivations, corrections | Yes |
| `paper_orchestra_workspace/` | Canonical paper workspace | Yes |
| `docs/` | Reports, registries, repro notes | Yes |
| `S-add__K-1_C-0_W-20_Col-0/` | Baseline checkpoint lineage | Yes |
| `ablation_destructive_7epoch/` | Strongest causal evidence on module necessity | Yes |

## Evidence / Historical Runs Worth Keeping Visible

| Path | Role | Keep at root |
|---|---|---|
| `weight_sweep_40/` | Large target-distribution sweep with summary tables | Yes, but historical |
| `kinetic_sweep/` | Flow-matching regularization sweep | Yes, but historical |
| `theory_switch_validation/` | Validation of optional switches | Yes, but historical |
| `manual_k1_k2_8epoch/` | Manual reproduction runs | Yes, but historical |
| `review_additional_experiments/` | Review-time supplemental experiments | Yes, but historical |

## Staging / Legacy Workspaces

| Path | Role | Keep at root |
|---|---|---|
| `paper_refine_v2/` | Legacy paper refinement workspace | For now |
| `paper_aaai_draft/` | Older draft workspace | For now |
| `paper_cn/` | Chinese paper / notes workspace | For now |
| `omf_sweep/` | OMF staging configs | For now |
| `omf_sweep2/` | OMF staging configs | For now |
| `omf_sweep3/` | OMF staging configs | For now |
| `next_round_80/` | Screening/staging suite definitions | For now |
| `full_dimensional_orthogonal_sweep_20/` | Sweep definition set | For now |
| `pareto_probe_4/` | Older probe configs / summaries | For now |
| `screening_grid_3epoch_108/` | Search configuration set | For now |
| `path_kinetic/` | Path-kinetic branch workspace, includes nested duplication | Needs dedicated cleanup |

## Cache / Archive / Packaging

| Path | Role | Keep at root |
|---|---|---|
| `archives/` | Local archive bundles moved out of root clutter | Yes |
| `eval_cache/` | Evaluation cache | Yes |
| `eval_results/` | Collated result outputs | Yes |

## Notable Size Hotspots

Approximate size snapshot from the 2026-05-16 audit:

- `weight_sweep_40/`: about `17.3 GB`
- `exp/`: about `16.3 GB`
- `archives/`: about `11.3 GB`
- `grid_search_3epoch/`: about `8.4 GB`
- `review_additional_experiments/`: about `3.2 GB`
- `kinetic_sweep/`: about `2.6 GB`
- `theory_switch_validation/`: about `1.3 GB`

## Cleanup Principle

We are optimizing for clarity, not minimum size. If a directory is heavy but still a stable evidence source, it remains visible. If it is heavy and purely packaging, it belongs in `archives/`.
