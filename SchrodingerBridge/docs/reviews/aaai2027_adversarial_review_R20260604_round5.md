# AAAI 2027 adversarial review round 5

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Figure script touched: `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_visual_panel.py`
- Reviewers: Feynman/Godel, Kant, Wegener, Cicero

## Reviewer scores

- Feynman/Godel, writing / AC lens: 7.0/10.
- Kant, experiments / statistics: 6.5/10.
- Wegener, method / math: 6.5/10.
- Cicero, figures / layout: 7.0/10.

## Shared diagnosis

- The paper is now strongest when framed as an evaluation-standard paper with LBM as a compact proof point.
- IDT is compelling, but it must be framed as a falsification contract rather than an obvious baseline.
- Distinct5 is credible as a fixed CLIP-separated stress test, but claims must stay scoped to CLIP-S-based art-to-art evaluation until additional fixed-rule splits land.
- SaMAM point estimates are useful evidence but should not carry finished-baseline language until Dalton returns a complete aligned packet.
- The method section must avoid implying solved Schrödinger bridge, full flow matching, path action, or unbiased SWD.
- Figure 3 remains the main visual risk because LBM examples are conservative/pale; metric labels help, but stronger examples or improved outputs are still desirable.

## Applied after round 5

- Abstract now says the Distinct5 result changes conclusions under CLIP-S-based evaluation and that the result motivates a stricter reporting standard, rather than claiming a universal standard outright.
- Contributions now position LBM as a proof point for the IDT reporting contract.
- Method wording was tightened:
  - optional online Sinkhorn is marked as a non-headline variant;
  - inactive `transport` wording was renamed to optional endpoint residual penalty;
  - intermediate Euler queries are described as deterministic execution steps, not trained path samples;
  - style control is split into style carrier `T_phi(s)` and content-conditioned routing `R_phi(s,z_0)`;
  - terminal loss is renamed an asymmetric semantic-projection discrepancy, not a metric or unbiased SWD;
  - kinetic regularization is kept as endpoint velocity-magnitude regularization.
- Table 1 was clarified:
  - `row CI` became `row-resamp.`;
  - `Train (min)` became `ret-train (min)`;
  - the footnote now states row-resampled intervals are exploratory sign checks and not clustered confidence intervals.
- Distinct5 protocol text now says the primary gate first tests whether a method moves beyond IDT.
- SaMAM result language now says the estimates provide current evidence for the failure mode, not a completed baseline conclusion.
- Historical cost caption now separates strict-750 operating-point cost from Distinct5 retained-checkpoint footprint.
- Figure 3 was regenerated with in-cell CLIP-S / LPIPS labels to make the qualitative panel less dependent on the caption.
- Figure 1 caption now points to Figure 3 for visual examples.
- Figure 2 caption was aligned with the `style-control interface` framing.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no undefined references/citations, no fatal errors, and no overfull boxes. Remaining warnings are XeLaTeX font-substitution warnings.
- Rendered pages inspected: page 1, page 4, page 6, and page 8.

## Remaining risks

- Figure 3 still does not visually prove strong stylization; it now exposes the evidence more honestly but still shows LBM as conservative/pale.
- Dalton should finish the SaMAM aligned packet before the paper treats SaMAM as a closed baseline result.
- Clustered source/style bootstrap remains needed before the row-resampled intervals can be replaced by formal uncertainty.
- Two additional fixed-rule WikiArt stress splits remain the best defense against the CLIP-selected split objection.
- Fixed-tokenizer/fixed-executor swaps remain the clean causal experiment for the representation claim.

## Experiments to delegate

- Dalton: finish SaMAM final/converged aligned packet with full and transfer CLIP-S, LPIPS, targetwise ArtFID, per-image rows, timing, and same IDT slots.
- Dalton: compute clustered bootstrap by source image and source-target direction for CLIP-S delta, LPIPS, ArtFID, and direct LBM-vs-IDT / LBM-vs-SaMST comparisons.
- Dalton: evaluate at least two additional fixed-rule WikiArt stress splits under the same 30 x 5 x 5 IDT protocol.
- Dalton: finish SaMST e5/e10/e15 aligned packet with targetwise ArtFID, not just CLIP-S/LPIPS convergence.
- Faraday: run fixed-tokenizer/fixed-executor swaps to separate representation from routing/queue changes.
- Faraday: run semantic-axis vs random-axis terminal matching with ArtFID to bound the SA-SWD claim.
- Dalton/Faraday: produce same-scope cost packet for LBM, SaMST, and SaMAM: train wall time, inference ms/img, eval wall time, hardware, and stopping rule.
