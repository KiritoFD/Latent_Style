# AAAI-27 Format Notes

Official sources checked on 2026-06-07:

- conference page: `https://aaai.org/conference/aaai/aaai-27/`
- author kit: `https://aaai.org/authorkit27/`

Current alignment in this directory:

- main paper source:
  - `paper_aaai2027.tex`
- official style files copied in:
  - `aaai2027.sty`
  - `aaai2027.bst`
  - `aaai2027.bib`
- build chain switched to:
  - `pdflatex`

Key AAAI-27 template requirements now reflected here:

- use `\usepackage[submission]{aaai2027}`
- use `aaai2027.bst`
- compile with `pdfLaTeX`, not `XeLaTeX`
- do not load `times`, `helvet`, or `courier` manually
  - the 2027 style loads the official font stack itself
- `TemplateVersion` should be `2027.1`
- submission mode hides authors and inserts the new anonymized-submission notice

Notable differences from the old AAAI-26 snapshot:

- `aaai2027.sty` is not just a filename bump
- it adds a PDFLaTeX engine guard
- it auto-loads `newtxtext`, `helvet`, and `courier`
- it adds a float barrier before references
- it has updated title-footnote and copyright handling
- it includes a `preprint` option and improved submission notice text

Practical rule for this workspace:

- do not edit `aaai2027.sty`
- keep paper edits inside `paper_aaai2027.tex`, figures, and snippets
- if a future build script or helper still points at `paper_aaai2026.tex` or `aaai2026.sty`, treat that as stale
