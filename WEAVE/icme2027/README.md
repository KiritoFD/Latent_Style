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

## ICME format checklist (confirmed 2026-09-30)

ICME 2027 (Xiamen, 13–17 July 2027) has not yet published its call; the rules below are the
latest official ones (ICME 2026 author instructions and template). Re-check when the 2027 site
opens; the regular-paper deadline has historically been in December.

| Requirement | Source | Status |
|---|---|---|
| ≤ 6 pages including all text, figures, **and references** | author instructions | 6 pages |
| Letter-size PDF, all fonts embedded, Times encouraged | author instructions | letter, `pdffonts` shows all embedded, IEEEtran Times |
| IEEE conference template (IEEEtran `conference`, IEEEbib) | official ICME 2026 LaTeX zip | template files copied unchanged |
| Abstract 100–150 words, identical to the CMT abstract; no math, symbols, or footnotes in title/abstract | author instructions + template | 146 words, plain text |
| Double blind: author block exactly "Anonymous ICME submission"; no identifying acknowledgments, links, or supplement titles | author instructions | done; cite own prior work in the third person |
| Supplement: single zip (site says ≤ 50 MB, template says ≤ 20 MB; use 20 MB); reviewers need not read it, so the paper must stand alone | author instructions + template | supplement PDF ≈ 0.9 MB |
| One primary subject area (+ up to 2 secondary) | CMT form | suggest *Multimedia analysis and generation*; secondary *Multimedia quality assessment and metrics*, *Image and video processing* |
| **Dual submission**: no substantially overlapping paper may be under review elsewhere during the ICME review period | author instructions | **Action needed:** the AAAI 2027 version must be withdrawn or have received its decision before the ICME submission |
| Rebuttal: 1 page, ICME CVPR-style template, seen only by area chairs | author instructions | template: ICME-2026-Rebuttal-Template.zip |

Review criteria (reviewer guidelines): relevance to ICME, novelty, technical correctness,
experimental validation and reproducibility, clarity, reference to prior work.

## Changes relative to the AAAI version

- Narrative reframed around the *identity shortcut*: an audit showing 8 of 13 methods leave the
  IDT–TGT sandwich on at least one benchmark (new *Valid* column in Table I), a quantified spectral
  cause (LL carries 69.5% of the gradient energy but has the lowest style separability, 0.12 vs 0.56
  for HH), and a headline restricted to valid methods (highest DINO-C on all three benchmarks).
  New title: *WEAVE: Escaping the Identity Shortcut in Lightweight Style Transfer with Wavelets*.

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
