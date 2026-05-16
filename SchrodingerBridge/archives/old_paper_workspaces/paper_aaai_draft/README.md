# AAAI Paper Draft

This directory contains an initial AAAI-oriented paper draft for the current SchrodingerBridge / Latent Style project.

Files:

| File | Purpose |
|---|---|
| `main.tex` | Main paper draft in English |
| `references.bib` | BibTeX references used by the draft |

Notes:

- The current draft is written in a portable LaTeX `article` style so it can compile without the official AAAI template.
- When the target AAAI template is available, migrate the body sections from `main.tex` into the official `aaai.sty` skeleton.
- Quantitative results are copied from `../docs/repro_report_zh/00_总览与核心结论.md` and `../docs/repro_report_zh/02_实验数据与结果汇总.md`.
- The claim is intentionally conservative: the method is positioned as fast latent-space multi-style artistic transfer with favorable style-content and artifact-quality trade-offs, not as universally outperforming every baseline on raw CLIP-style.

