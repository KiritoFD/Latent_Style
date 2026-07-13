# WEAVE — AAAI 2027 paper bundle (`aaai2027_v4`)

This folder is **self-contained**: it holds the LaTeX source, the style/class
files, the bibliography, the figures, and the figure-generation scripts with
their bundled inputs. Copying this folder alone is enough to compile the paper
and to regenerate or edit every figure.

## Layout

```
aaai2027_v4/
├── paper.tex            # main source
├── aaai2027.sty         # AAAI 2027 style (submission)
├── aaai2027.bst         # AAAI 2027 bibliography style
├── refs.bib             # bibliography
├── build.ps1            # compile paper.tex -> paper.pdf (pdflatex + bibtex)
├── gen_figures.ps1      # regenerate every figure from the scripts below
├── *.png / *.pdf        # committed figures referenced by paper.tex
├── aaai_arch_diagram_v16_staggered_bundle.drawio.png  # architecture diagram
├── fig_teaser_comparison.png                          # teaser figure
├── plot_page1_summary.py                              # fig_distinct5_page1_summary.pdf
├── make_radar_metric_blocks.py                        # radar_metric_blocks_A_clip_dinos_robustbreak.png
└── fig_data/            # bundled inputs for the scripts (no external paths)
    ├── dino_main.json
    ├── curve_metrics_hf.csv
    └── teaser_*.jpg / *.png
```

## Compile

```powershell
.\build.ps1
```

Requires a TeX Live / MiKTeX install with `pdflatex` and `bibtex` on PATH.

## Regenerate figures

```powershell
.\gen_figures.ps1
```

The page-1 summary and radar scripts read their inputs from `./fig_data` and
write the figure next to `paper.tex`, so editing a script and re-running
reproduces the exact file the paper includes. The metric-block radar chart
(`make_radar_metric_blocks.py`) reads the same values reported in Table 1 of
`paper.tex`; update the table there first, then re-run.

## Notes

- `framework_sfm_main.png` is produced by `gen_framework_figure.py` (the base
  diagram). Earlier manual lightening was a one-off aesthetic tweak; the script
  now writes directly to the file name used by the paper.
- Build artifacts (`paper.aux`, `paper.bbl`, `paper.blg`, `paper.fls`,
  `paper.fdb_latexmk`, `paper.synctex.gz`, `paper.log`) are regenerated on every
  build and are not part of the source bundle.
