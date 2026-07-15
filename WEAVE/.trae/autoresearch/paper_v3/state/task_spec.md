# Paper v3 Continuous Polish Task Spec

## Goal
Continuously polish the AAAI 2027 v2 paper to Strong Accept level. Target: 7 pages main text + ~2 pages references + checklist. Emphasize mathematical theory depth, Method and Experiments sections. Core narrative: training in minutes on RTX 3060, real (better than IDT) style transfer.

## Hard Constraints
- Zero interaction during iteration; resolve ambiguity and log reasoning
- All data must be verified against train.log / summary.json / docs/baseline/
- No fabricated concepts; no internal naming; formal, objective, reviewer-proof
- AAAI 2026 format (letterpaper, submission mode)
- Main text 7 pages; references + checklist ~2 pages; total ~9 pages
- Method and Experiments are the focus; math theory must be deep
- Language: concise, clear, simple sentences; no invented terminology

## Milestones
- M1: Evaluate current paper gaps (which sections need expansion to reach 7 pages)
- M2: Deepen mathematical theory in Method section
- M3: Enhance Experiments section (more analysis, more tables/figures)
- M4: Add checklist + verify reference page count
- M5: Compile PDF, verify page count, git commit

## Success Criteria
- paper.pdf has 7 pages of main text + 2 pages references/checklist
- Method section has rigorous math (attractors, fiber flow formal definition, convergence argument)
- Experiments section has per-domain analysis, ablation insight, efficiency analysis
- All cited data verifiable in train.log / summary.json / docs/baseline/README.md
- 0 LaTeX warnings, 0 undefined refs
- Git commit with detailed message

## Verification Data Sources
- T11 train.log: g:\GitHub\Latent_Style\SchrodingerBridge\exp\FCSB\local_t\630_local_t11_stochastic_dwt_p08\train.log
- T11 summary.json: ...\630_local_t11_stochastic_dwt_p08\full_eval\epoch_0005\summary.json
- Baseline data: g:\GitHub\Latent_Style\SchrodingerBridge\docs\baseline\README.md
- SaMam real values: 0.5816 / 0.2434 (v5 corrected)
- Per-domain matrix: in summary.json matrix_breakdown

## Anti-Fabrication Rules
- Every number in paper.tex MUST be traceable to a log/json/md file
- If a number cannot be traced, run the experiment or read the log to fill it
- Citations must exist in refs.bib; if missing, add the bibtex entry
- No new concepts invented; only formalize existing phenomena
