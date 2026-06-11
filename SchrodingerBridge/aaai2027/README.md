## AAAI-27 Workspace

This directory is the active AAAI-27 paper workspace derived from the earlier
`9a4b99dfa` snapshot and then realigned to the official AAAI-27 Author Kit.

Current paper entrypoints:

- `paper_aaai2027.tex`
- `paper_aaai2027.pdf`
- `build_paper.bat`

Current figure workflow:

- page-1 summary:
  - `scripts_gen_distinct5_page1_summary.py`
  - `figures/fig_distinct5_page1_summary.pdf`
- full Distinct5 operating-point landscape:
  - `scripts_gen_distinct5_all_points_big.py`
  - `fig_distinct5_all_points_big.pdf`

Strongest-point ledger:

- `G:\GitHub\Latent_Style\best.csv`

Runtime artifact policy:

- authoritative round-1 fast-eval evidence lives under
  `round1_*_remote_full_eval_pull/`
- authoritative closure notes and plan logs live under
  `docs/experiments/round1_full_sweep/`
- detached watcher logs, remote packet tar wrappers, and loose root-level
  checkpoint drops are scratch and should stay ignored or be moved into
  snapshot/archive buckets instead of accumulating in the active root

The goal of this directory is no longer archival reproduction. It is the live
paper-facing surface for the AAAI-27 rewrite, figure updates, and later review
passes.
