# SchrodingerBridge Documentation

**Last updated:** 2026-07-13
**Current status:** paper bundle is coherent and committed; broader research worktree is active and dirty.

This directory contains both current evidence and historical archives. Treat the documents below as the source map for the current AAAI v4 work.

## Current Source Map

| Path | Role |
|---|---|
| `delivery/DELIVERY_SUMMARY.md` | Current delivery-level conclusion, best checkpoints, and status caveats. |
| `713/HANDOFF_2026-07-13.md` | Current complete handoff: repo/remote audit, remote run workflow, probe results, conclusions, and next plan. |
| `713/README.md` | Short entry point for the 713 handoff. |
| `713/METHOD_EXPLORATION_AND_CKPT_2026-07-13.md` | Current checkpoint and method exploration history extracted from older non-713 docs. |
| `713/REPO_REMOTE_AUDIT_2026-07-13.md` | Detailed local/remote audit. |
| `713/HF_ARCHITECTURE_PROBE_2026-07-13.md` | Detailed HF-route probe diagnosis. |
| `713/EXPERIMENT_SUMMARY_FOR_METHOD_AND_NEXT_PLAN.md` | Method-facing experiment summary and next plan. |
| `model_probe/target_hf_delta_eval_summary.json` | Aggregate numeric evidence for HF route probes. |
| `model_probe/HF_DELTA_DIAGNOSIS_2026-07-13.md` | Earlier diagnosis of the missing target-HF condition path. |
| `model_probe/generation_only_timing_summary.json` | Generation-only timing on RTX 3060. |
| `latent_migration/final_metrics_table.md` | Resolution/baseline metric table from the latent migration work. |
| `archive/713_external_legacy/` | Archived non-713 legacy docs mined into the current 713 method exploration note. |
| `archive/` | Historical documents. Use for traceability only, not as current claims without re-checking. |

## Paper Bundle

The current paper bundle lives in:

```text
aaai2027_v4/
```

Important files:

| File | Role |
|---|---|
| `paper.tex`, `paper.pdf` | Main paper source and compiled PDF. |
| `supplement.tex` | Formal comprehensive supplement source. |
| `SUPPLEMENTARY_MATERIAL.md` | Compact supplement map aligned to `supplement.tex`. |
| `make_radar_metric_blocks.py` | Source for the current metric-block radar. |
| `radar_metric_blocks_A_clip_dinos_robustbreak.png` | Current radar figure. |

Commit `0867d43d7` updated the paper bundle with the radar and first formal supplement. The expanded supplement and the latest delivery docs are newer working-tree updates and should be committed separately from large source/config cleanup.

## Current Cleanup Reality

The repository is not clean. A recent audit found:

| Status | Count |
|---|---:|
| Deleted | 473 |
| Modified | 44 |
| Untracked | 299 |
| Total | 816 |

These changes include historical config/tool deletions, new probe evidence, supplement build scratch, source changes, and logs. Do not infer from older docs that all cleanup has been committed.

## Remote Reality

Remote host:

```text
ssh -p 2222 administrator@100.115.18.62
```

Important remote facts:

| Path | Status |
|---|---|
| `I:\Github\Latent_Style\SchrodingerBridge` | Exists; synchronized experiment/code tree; **not a git repo**. |
| `I:\checkpoints` | Does not exist on the audited remote. |
| `I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe` | Main remote location for 2026-07-13 HF route runs. |
| `I:\latent_style_remote_curated` | Historical archaeology index, not current source of claims. |

Use local git commits for source control and remote files for experiment evidence.

## Current Method Takeaway

The latest probe conclusion is:

> The training target was already style-heavy in high-frequency bands; the bottleneck was the condition route. The useful fix is to pass target-image HF information into HF velocity heads through pooled, coordinate-free subband codes while keeping LL protected.

The best current architecture probe is `target_hf_subband_ft6`, with DINO-S `0.488624`, DINO-C `0.798123`, CLIP-S `0.720880`, and LPIPS `0.296553` under the D5 probe protocol.

## Maintenance Rules

1. Update `713/HANDOFF_2026-07-13.md` or a new dated handoff before making broad cleanup claims.
2. Keep paper-facing files in `aaai2027_v4/` separate from scratch build products.
3. Use `model_probe/target_hf_delta_eval_summary.json` as the numeric source for HF-route claims.
4. Move obsolete narrative docs to `archive/` instead of leaving them beside current claims.
5. Commit documentation updates separately from large source/config cleanup.
