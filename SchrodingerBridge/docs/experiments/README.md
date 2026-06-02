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

### Logging / provenance contract

- `aaai2027_experiment_logging_contract_20260603.md`

### Next paper-closing protocol

- `2026-06-03-flow-loss-metric-ablation-protocol.md`

### Current claim-closing packets

- `2026-06-03-flow-loss-metric-ablation/README.md`
- `2026-06-03-saswd-axis-ablation/README.md`

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
