# WEAVE — IEEE ICME paper bundle (`icme2027`)

ICME rewrite of the AAAI 2027 submission in `../aaai2027_v4/` (which stays frozen).
Self-contained: copy this folder alone to compile.

| File | Role |
|---|---|
| `weave.tex` → `weave.pdf` | Main paper, IEEEtran conference class, **6 pages including references** |
| `weave_supp.tex` → `weave_supp.pdf` | Supplementary material (upload zipped, ≤20 MB) |
| `fig_arch.tex` | Architecture figure (TikZ, replaces the stale draw.io PNG) |
| `table_main.tex` | Table I, **generated** — do not edit |
| `data/main_table.csv` | Source of record for Table I, with provenance for every cell |
| `tools/make_main_table.py` | CSV → `table_main.tex` (rankings, †/‡ marks computed from IDT/TGT) |
| `refs.bib`, `IEEEtran.cls`, `IEEEbib.bst` | Bibliography and official ICME 2026 template files |
| `figures/` | Figures copied from `../aaai2027_v4/` |

## Build

```bash
python tools/make_main_table.py      # only after editing data/main_table.csv
latexmk -pdf weave.tex weave_supp.tex
```

Windows: `.\build.ps1`. Requires TeX Live/MiKTeX with `pdflatex`, `bibtex`, TikZ.

## ICME format checklist

- Letter paper, IEEEtran `conference`, all fonts embedded (check with `pdffonts weave.pdf`).
- ≤ 6 pages **including references**; abstract 100–150 words with no math or symbols (currently 148).
- Author field `Anonymous ICME submission`; no identifying links or acknowledgments.
- Supplement referenced in the paper; the paper must stand on its own.

## Changes relative to the AAAI version

- Restructured for the 6-page limit: condensed related work, method, and discussion; ethics,
  reproducibility, protocol details, sweeps, audits, and probe ledgers moved to the supplement.
- Corrected Table 1 cells against raw DINO sidecars (Z-STAR P2A, StyleID R5), made the †/‡ marks
  follow their definition, marked the Z-STAR time as an estimate, fixed the R5 description
  (five random WikiArt families, not 20), and made the ablation caption state that retrained
  variants are reported at their best-DINO-S epoch (stopping regret in supplement Table S3).
- New TikZ architecture figure consistent with the implementation (1.04M parameters, AdaIN at
  every Euler step, pooled LH/HL codes, no HH head).
- Added citations for StyleShot, Seedream, DINOv2, and CLIP.

Provenance of every number: `../../docs/experiments/02_results_of_record.md`.
