# Experiments Index

This directory stores experiment notes, comparison reports, diagnostic audits,
and benchmark-specific artifacts for `SchrodingerBridge`.

For the current AAAI 2027 push, start here:

- project-wide working index:
  `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- current paper/experiment plan:
  `SchrodingerBridge/docs/experiments/2026-06-02-aaai2027-paper-update-plan.md`
- unified experiment ledger:
  `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`

## Recommended entrypoints

### Paper-facing comparison

- `comparison_20260602/README.md`
- `comparison_20260602/comparison_report.md`

### Distinct5-512 stress benchmark

- `2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `distinct5_512_20260602/`

### No-op / metric-hacking diagnosis

- `metric_hacking_noop_20260602/README.md`
- `noop_comparison_across_datasets_20260602/`
- `idt_eval_20260602/`

### Historical gap tracking

- `2026-06-01-main-table-gap-analysis.md`

### Repo cleanup / archive hygiene

- `2026-06-03-repo-cleanup-and-archive-pass.md`
- `2026-06-03-exp-surface-classification.md`
- `2026-06-03-timing-artifact-prune.md`
- `2026-06-03-exploratory-image-prune.md`
- `2026-06-03-smoke-surface-prune.md`

### Logging / provenance contract

- `aaai2027_experiment_logging_contract_20260603.md`

### Current claim-closing packets

- `2026-06-03-flow-loss-metric-ablation-protocol.md`
- `2026-06-03-flow-loss-metric-ablation/README.md`
- `2026-06-03-saswd-axis-ablation/README.md`
- `2026-06-03-time-to-parity/README.md`
- `2026-06-03-tokenizer-execution-alignment-protocol.md`
- `2026-06-03-tokenizer-execution-alignment/README.md`
- `2026-06-03-tokenizer-execution-alignment-l-family/README.md`
- `2026-06-03-tokenizer-localization-probe-protocol.md`
- `2026-06-03-tokenizer-localization/README.md`
- `2026-06-03-tokenizer-localization/launch_manifest_20260603.md`
- `2026-06-03-tokenizer-localization-remote-preflight.md`
- `2026-06-03-path-stability-protocol.md`
- `2026-06-03-path-stability-launch-status.md`
- `2026-06-03-path-stability-probe/README.md`

### Current mechanism-closure packet status

The matched Distinct5 same-family path-stability / weakened-kinetic packet is
now landed.

Use these files together:

- protocol and launch chain:
  - `2026-06-03-path-stability-protocol.md`
  - `2026-06-03-path-stability-launch-status.md`
- retained probe readout:
  - `2026-06-03-path-stability-probe/README.md`
- base config:
  - `../../configs/aaai2027/path_kinetic_h_base_seed42_b44_base.json`
- weakened/no-kinetic configs:
  - `../../configs/aaai2027/path_kinetic_h_base_seed42_b44_k025.json`
  - `../../configs/aaai2027/path_kinetic_h_base_seed42_b44_k000.json`
- experiment ledger rows:
  - `aaai2027_master_experiment_log.csv`

Current state:

- tokenizer-localization has already landed as a bounded `L`-family packet;
- path-stability has now landed as a bounded same-family `H`-packet with
  retained `base`, `k025`, `k000`, and probe artifacts;
- do not promote the kinetic/path-energy story beyond its current bounded form
  until the manuscript absorbs this landed packet and a fresh review cycle
  re-checks the claim boundary.

### Archive / cleanup boundary

For cleanup or archive questions, do not guess from raw directory names alone.
Use:

- `2026-06-03-exp-surface-classification.md`
- `2026-06-03-repo-cleanup-and-archive-pass.md`
- `../cleanup/worktree_triage_20260603.md`
- `../cleanup/paper_surface_audit_20260603.md`
- `../../archives/README.md`

## Directory intent

- dated markdown files:
  decisions, run summaries, and writing plans
- dated subdirectories:
  benchmark-specific artifacts, plots, tables, and evaluation bundles
- `comparison_*`:
  cross-model consolidated reports
- `*_noop_*` / `idt_*`:
  no-op identity baselines and metric-trap audits

When adding a new experiment block, prefer:

1. a dated note for the narrative summary, and
2. a dedicated dated directory for the raw tables/plots/artifacts.

When a formal `aaai2027_*` packet changes phase from `planned`, `queued`, or
`running` into `completed`, it must also trigger:

1. a ledger update in `aaai2027_master_experiment_log.csv`, and
2. a fresh review-cycle check via `../reviews/aaai2027_review_protocol.md`.
