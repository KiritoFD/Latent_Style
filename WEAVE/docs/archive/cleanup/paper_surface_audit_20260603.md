# Paper Surface Audit - 2026-06-03

Purpose:

- identify which paper files under `aaai_submission/` are active inputs versus
  parallel copies;
- separate safe cleanup candidates from currently referenced figure assets;
- avoid deleting files that are still used by `paper_aaai2026.tex` or the
  `final/` export surface.

## 1. Current active manuscript surface

Primary paper source:

- `aaai_submission/paper_aaai2026.tex`
- `aaai_submission/refs.bib`

Directly referenced figure files in `paper_aaai2026.tex`:

- `framework_lbm_main_v5.png`
- `fig_qual_grid_ours_vs_samst.png`
- `fig_zoom_ours_vs_samst.png`
- `figures/fig_distinct5_pareto.pdf`
- `fig_ablation_pareto.png`

Implication:

- the manuscript currently mixes root-level assets and curated `figures/`
  assets;
- referenced files are not safe deletion candidates until the manuscript is
  migrated to one stable surface.

## 2. Parallel figure surfaces

Tracked parallel figure copies currently exist in three places:

1. root `aaai_submission/`
2. `aaai_submission/figures/`
3. `aaai_submission/final/`

Observed roles:

- root `aaai_submission/`
  - active manuscript build inputs for `paper_aaai2026.tex`
- `aaai_submission/figures/`
  - curated paper-figure output surface and vector/raster bundle
- `aaai_submission/final/`
  - export / package surface for the separate final-paper bundle

Implication:

- these are not accidental duplicates in the narrow sense;
- they are parallel publication surfaces with overlapping payload;
- cleanup should prefer convergence by reference-path migration, not blind
  deletion.

## 3. Safe cleanup already applied

Removed from git tracking as pure build byproducts:

- `paper_aaai2026.blg`
- `final/paper.blg`
- `final/paper.fdb_latexmk`
- `final/paper.fls`
- `final/paper.out`

These files had no manuscript, figure, or experiment-evidence role.

## 4. Safe next cleanup candidates

### A. Root manuscript build byproducts

Keep ignored and untracked only:

- `paper_aaai2026.aux`
- `paper_aaai2026.fdb_latexmk`
- `paper_aaai2026.fls`
- `paper_aaai2026.log`
- `xelatex_pass*.log`

These are compile-runtime artifacts, not paper assets.

### B. Reference-path convergence candidate

Longer-term cleanup opportunity:

- migrate `paper_aaai2026.tex` to consume the curated `figures/` directory
  consistently;
- after that migration, reassess whether the root-level PNG copies can be
  removed or regenerated on demand.

This is not a safe same-commit deletion today because the current paper source
still references the root-level PNG names directly.

## 5. Unsafe cleanup candidates right now

Do not delete yet:

- `framework_lbm_main_v5.png`
- root-level figure PNGs currently referenced by the manuscript
- `figures/fig_distinct5_pareto.pdf`
- `final/` assets used by the packaged export surface

Reason:

- each still participates in at least one active paper build or export path.

## 6. Recommended next paper-surface cleanup step

If paper-surface convergence becomes a priority, do it in this order:

1. rewrite `paper_aaai2026.tex` to reference only one curated figure surface;
2. rebuild and verify the PDF;
3. only then remove redundant copies and update the package/export path.
