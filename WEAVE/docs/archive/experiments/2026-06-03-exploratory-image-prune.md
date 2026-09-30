# Exploratory Image Prune

Date: 2026-06-03

Purpose:

- reduce disk usage in frozen exploratory experiment families;
- preserve summaries, ledgers, checkpoints, and logs;
- avoid breaking current paper-facing citations and review artifacts.

## Rule used

This prune followed the same backing-store policy as:

- `docs/experiments/2026-06-03-exp-surface-classification.md`

Operational rule:

1. delete only generated `images/` directories;
2. keep:
   - `summary.json`
   - `metrics.csv`
   - ledgers / CSV frontiers
   - checkpoints
   - train / eval logs
   - config snapshots
3. do not touch families whose generated images are directly referenced by the
   current paper figures or visual-alignment manifests.

## Pruned families

Deleted all nested `images/` directories under these exploratory or
diagnostic-only experiment families:

- `exp/style_representation*`
- `exp/diagnostics`
- `exp/frontier`
- `exp/vae_backend`
- `exp/diffeomorphic_tangent_sweep`
- `exp/tokenizer_adain_gate_calibration`

Why these families were safe:

- current docs and ledgers reference their summaries, frontier CSVs, or notes,
  not the generated image payloads;
- no current paper-facing visual artifact path depends on these `images/`
  subdirectories;
- the families are either frozen exploratory probes or utility / diagnostic
  backing stores rather than active figure sources.

## Size change

Deleted payload:

- `142` image directories
- about `38,725` generated image files
- approximately `530.85 MB` reclaimed

Post-prune verification:

- remaining nested `images/` directories under the targeted families:
  - `0`

## Git/worktree effect

This prune did not dirty tracked source files by itself because the removed
payloads were runtime artifacts, not tracked research-source files.

The tracked changes for this pass are only the cleanup notes and index updates
that document the deletion.
