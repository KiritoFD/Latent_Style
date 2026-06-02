# AI Research Writing Prompt Usage

Updated: 2026-06-02

## Sources

Installed skills:

- `Orchestra-Research/AI-Research-SKILLs/20-ml-paper-writing/ml-paper-writing`
- `Orchestra-Research/AI-Research-SKILLs/20-ml-paper-writing/academic-plotting`

Downloaded prompt reference:

- `Leey21/awesome-ai-research-writing`
- Temporary local copy: `%TEMP%/ai_research_refs/awesome-ai-research-writing-README.md`

## Relevant prompt families used

- Logic check: separate completed results from pending baselines; avoid mixing distinct evaluation protocols.
- Experiment analysis: state the tested claim before presenting numbers; identify what the result does and does not prove.
- Reviewer-perspective audit: mark incomplete SaMST-512 and ongoing SaMAM continuation explicitly, rather than turning partial status into a final comparison.
- Figure recommendation: choose a data-driven Pareto scatter/line plot for `CLIP-style` vs `1-LPIPS`.
- Figure/table title discipline: captions should state what the figure/table shows and the main takeaway without relying on surrounding text.

## Local decisions

- No Gemini-generated diagrams.
- Numerical figures use deterministic matplotlib scripts and vector PDF output.
- Architecture diagrams should use TikZ or deterministic SVG/PDF when updated.
- AAAI figure sizing follows a full-width target of roughly 7.0 inches and Times-compatible serif text.

## Applied updates

- Added `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_pareto.py`.
- Added `SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.pdf`.
- Updated `SchrodingerBridge/aaai_submission/paper_aaai2026.tex` with a separated Distinct5-512 table and figure.
- Added `SchrodingerBridge/docs/experiments/2026-06-02-aaai2027-paper-update-plan.md`.
