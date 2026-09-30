# Paper Figure Inventory

Updated: 2026-06-03

This directory is the preferred home for active paper-facing figures and
vector plots.

## Active current-paper figures already stored here

- `fig_distinct5_pareto.pdf`
- `fig_distinct5_pareto.png`
- `clip_style_vs_1lpips_full_lancet_samam_noop.pdf`
- `clip_style_vs_1lpips_full_lancet_samam_noop.png`
- `clip_style_vs_1lpips_transfer_lancet_samam_noop.pdf`
- `clip_style_vs_1lpips_transfer_lancet_samam_noop.png`

The first two files are cited directly by `paper_aaai2026.tex`. The CLIP-style
vs. `1-LPIPS` plots are active supporting assets for review memos and
experiment notes under `docs/experiments/`, even though they are not in the
current manuscript body.

## Current paper still depends on top-level aaai_submission assets

The manuscript currently includes these files from the parent
`aaai_submission/` directory:

- `framework_lbm_main.png`
- `fig_qual_grid_ours_vs_samst.png`
- `fig_zoom_ours_vs_samst.png`
- `fig_ablation_pareto.png`
- `fig_weight_sweep_summary.png`
- `fig_train_efficiency_pareto.png`

These should be normalized into one active figure surface in a later cleanup
pass, but they remain in place now to avoid breaking older references and
presentation materials.

## Generation scripts

- `../scripts_gen_distinct5_pareto.py`
- `../scripts_gen_distinct5_full_transfer_pareto.py`
- `../scripts_collect_distinct5_full_transfer_points.py`
- `../scripts_gen_aaai2027_figures.py`

## Exploratory outputs that should not stay in the active surface

If regenerated for inspection, keep them transient unless the manuscript starts
referencing them explicitly:

- `../fig_eval_landscape.pdf`
- `../fig_eval_landscape.png`
- `../framework_figure.png`
- `../framework_lbm_main_saswd.png`

## Figure hygiene rule

For new paper-facing plots:

1. prefer vector `pdf` first,
2. keep a `png` fallback,
3. store the generating script next to the paper source,
4. ensure the source CSV or table lives under `docs/experiments/`.
