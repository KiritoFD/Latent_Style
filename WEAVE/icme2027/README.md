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
| `tools/make_main_table.py` | CSV → `table_main.tex` (rankings, †/‡ marks and *Valid* column computed from IDT/TGT) |
| `tools/make_figures.py` | Regenerates Fig. 1 (`fig_sandwich.pdf`), Fig. 2 (`fig_spectral.pdf`) and Fig. 4 (`fig_qualitative.pdf`) from `data/` |
| `data/artfid_d5.csv`, `data/probe_*.csv` | ArtFID audit (canonical manifest only) and Haar-band probe data behind Figs. 1–2 |
| `refs.bib`, `IEEEtran.cls`, `IEEEbib.bst` | Bibliography and official ICME 2026 template files |
| `figures/` | Generated figures (above) plus supplement figures copied from `../aaai2027_v4/` |

## Build

```bash
python tools/make_main_table.py      # only after editing data/main_table.csv
python tools/make_figures.py         # needs matplotlib, numpy, pillow
latexmk -pdf weave.tex weave_supp.tex
```

Windows: `.\build.ps1`. Requires TeX Live/MiKTeX with `pdflatex`, `bibtex`, TikZ.

## ICME format checklist (confirmed 2026-10-01)

ICME 2027 (Xiamen, 13–17 July 2027) has not yet published its call; the rules below are the
latest official ones (ICME 2026 author instructions and template). Re-check when the 2027 site
opens; the regular-paper deadline has historically been in December.

| Requirement | Source | Status |
|---|---|---|
| ≤ 6 pages including all text, figures, **and references** | author instructions | 6 pages |
| Letter-size PDF, all fonts embedded, Times encouraged | author instructions | letter; `pdffonts` shows all fonts embedded (Times in text, STIX TrueType in figures, no Type 3) |
| IEEE conference template (IEEEtran `conference`, IEEEbib) | official ICME 2026 LaTeX zip | template files copied unchanged |
| Abstract 100–150 words, identical to the CMT abstract; no math, symbols, or footnotes in title/abstract | author instructions + template | 146 words, plain text |
| Double blind: author block exactly "Anonymous ICME submission"; no identifying acknowledgments, links, or supplement titles | author instructions | done; cite own prior work in the third person |
| Supplement: single zip (site says ≤ 50 MB, template says ≤ 20 MB; use 20 MB); reviewers need not read it, so the paper must stand alone | author instructions + template | supplement PDF ≈ 0.9 MB |
| One primary subject area (+ up to 2 secondary) | CMT form | suggest *Multimedia analysis and generation*; secondary *Multimedia quality assessment and metrics*, *Image and video processing* |
| **Dual submission**: no substantially overlapping paper may be under review elsewhere during the ICME review period | author instructions | AAAI 2027 version was rejected (per authors), so there is no concurrent submission |
| Rebuttal: 1 page, ICME CVPR-style template, seen only by area chairs | author instructions | template: ICME-2026-Rebuttal-Template.zip |

Review criteria (reviewer guidelines): relevance to ICME, novelty, technical correctness,
experimental validation and reproducibility, clarity, reference to prior work.

## Changes relative to the AAAI version

- **Tone (2026-10-01).** Assertive rewrite with the defensive hedging removed; every strengthened claim
  was re-verified against `data/main_table.csv`: 8 of 12 baselines fail the sandwich; all four compact
  baselines (<10M trainable parameters: CUT, SaMST, SaMam, Latent-WCT) fall below IDT in style on at
  least one benchmark; WEAVE is Pareto-optimal against all 12 baselines in every style–content pair on
  D5 and R5; 11× smaller and ≥11× faster than every other valid learned model; 28× faster training than
  the fastest learned baseline. New title: *WEAVE: Breaking the Identity Shortcut in Lightweight Style
  Transfer with Wavelets*.
- **Narrative.** Reframed around the *identity shortcut*: a dedicated diagnosis section (IDT–TGT
  sandwich as a formal validity test; Haar-band gradient dominance), an audit showing 8 of 13
  methods fail on at least one benchmark (*Valid* column in Table I), and a headline restricted to
  valid methods (highest DINO-C on all three benchmarks; Pareto-optimal on D5 and R5).
- **Analysis.** Proposition 1 with proof: the source-anchored endpoint requests only a channel-wise
  affine LL change and bounds its energy by α² of the direct target, which lowers LL's share of the
  displacement energy from 69.5% to at most 17.0% (derivation in supplement Sec. S3).
- **Figures.** Fig. 1 is regenerated from `data/main_table.csv` (the AAAI page-1 figure plotted an
  outdated base-model point and stale baseline values, and its ArtFID panel mixed in methods
  evaluated on a different source manifest); Fig. 2 annotates the band shares and the bound; the
  qualitative figure adds 3× close-ups that show the texture change; the TikZ architecture figure
  matches the implementation (1.04M parameters, AdaIN at every Euler step, pooled LH/HL codes, no
  HH head).
- **Honesty fixes.** Table cells corrected against raw DINO sidecars (Z-STAR P2A, StyleID R5);
  †/‡ marks follow their definition; Z-STAR time marked as an estimate; R5 is five random WikiArt
  families; retrained ablations are reported at their best-DINO-S epoch; the P2A gap is stated
  (SaMam dominates WEAVE there); WEAVE's test-time reference is disclosed as the TGT image, and the
  reference-pool test shows the DINO-S margin does not depend on it; seed s.d. used as the noise
  yardstick for ablations.
- **Bibliography.** Rebuilt and verified: fixed wrong author lists (Z-STAR, AesPA-Net, StyTr²),
  wrong venues (WCT is NeurIPS, StyleAligned is CVPR, published versions of SaMam, FreqFlow, ADD,
  flow matching, P2P, DSB, ArtFID); added FDD, AesFA, MicroAST, ArtFlow, LBM and the NST evaluation
  review.
- Ethics, reproducibility details, sweeps, audits, and probe ledgers live in the supplement.

Provenance of every number: `../../docs/experiments/02_results_of_record.md`.
