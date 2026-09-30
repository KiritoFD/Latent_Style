# AAAI 2027 adversarial review round 4

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Figure script touched: `SchrodingerBridge/aaai_submission/scripts_gen_framework_claimsafe.py`
- Reviewers: Feynman, Kant, Wegener, Cicero

## Reviewer scores

- Feynman, writing / AC lens: 6.0/10.
- Kant, experiments / statistics: 6.5/10.
- Wegener, method / math: 6.5/10.
- Cicero, figures / layout: 7.1/10.

## Applied in this writing pass

- Re-centered the paper around IDT-calibrated Style-ID transfer rather than presenting LBM as a generic transport-method paper.
- Rewrote the abstract into four moves: wrong null, IDT calibration, baseline reversal, and LBM's low-damage positive-movement operating points.
- Changed Distinct5 wording to a fixed CLIP-separated WikiArt stress split used as an evaluator-stress test, not a universal benchmark.
- Marked SaMAM as current reproduced checkpoint estimates where the retained per-image packet is incomplete.
- Changed Distinct5 table wording from selected-checkpoint to retained-checkpoint footprint and changed `row 95% CI` to `row CI`.
- Reworded row uncertainty as exploratory row-resampled intervals and explicitly deferred clustered source/style bootstrap.
- Tightened the main result sentence: retained LBM-F/K operating points occupy the low-damage positive-movement region; SaMST has higher transfer CLIP-S but much higher LPIPS and targetwise ArtFID.
- Shortened the Distinct5 visual-panel caption so the figure explains the failure modes without defensive prose.
- Repaired method notation to match the active implementation:
  - distinguished training endpoint `hat z_1 = z_0 + v_theta(z_0,1,s)` from multi-step inference endpoint `Phi_{theta,K}`;
  - applied terminal matching to the training endpoint in the method text;
  - renamed kinetic regularization to endpoint velocity-magnitude regularization;
  - replaced the patch-size SWD terminal formula with the active semantic-axis spatial sorted-projection estimator;
  - kept patch-size/random/dilated SWD as a related endpoint-cost family for queues and ablations;
  - expanded tokenizer notation to include optional style code, style spatial prior, and content-conditioned routing.
- Regenerated the framework figure so the inset now says `sem-proj`, `train endpoint`, `target endpoint`, and `routing axes`, avoiding the old `z_K patches` mismatch.
- Replaced the first Figure 3 qualitative row from manifest row 0 to row 1. The previous row cleared metrics largely through a near-white LBM output; the replacement keeps positive-IDT LBM-F/K movement while showing a more legible image transformation.

## Current PDF check

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no undefined references/citations, no fatal errors, no overfull boxes found by the checked patterns.
- Font check: no Type 3 fonts; only XeLaTeX template font-substitution warnings remain.
- Rendered pages inspected: first page, framework/method pages, main table + visual page.

## Remaining risks

- Figure 3 still contains conservative/pale LBM examples. It is acceptable for the low-displacement claim, but a stronger visual row should replace at least one example if we can find a case where LBM visibly moves style without whitening.
- After the row-1 replacement, Figure 3 is less vulnerable than the previous version but still conservative. A stronger LBM visual should be generated or selected before treating qualitative evidence as a major selling point.
- Distinct5 remains a CLIP-S stress split selected by CLIP prototypes and evaluated with CLIP-S. The paper now states this, but stronger generality needs fixed-rule follow-up splits.
- SaMAM rows remain point estimates until Dalton returns a final/tuned paired packet with transfer CLIP-S, LPIPS, targetwise ArtFID, timing, and per-image rows.
- Row-resampled intervals are only a diagnostic. Clustered source/style bootstrap is still needed for stronger statistical language.
- Tokenizer causality is still bounded: Table 6 mixes representation, routing, and queue changes. Fixed-tokenizer/fixed-executor swaps remain the clean causal test.

## Next writing gate

Do not run another broad review loop until one of these lands:

- a better Distinct5 qualitative row for Figure 3;
- Dalton's SaMAM final/tuned packet;
- SaMST e5/e15 targetwise ArtFID packet;
- at least one additional fixed-rule WikiArt stress split;
- clustered bootstrap or fixed-tokenizer/fixed-executor ablation.

## Follow-up writing polish, same round

Applied after the round-4 review:

- Rewrote the abstract from a metric listing into the paper's core claim chain: hidden art-to-art null, IDT calibration, baseline reversal, and LBM as low-damage executed movement.
- Tightened the introduction so the first page states why raw target-style affinity is unsafe before presenting LBM.
- Reframed the method overview around an endpoint-level execution contract rather than defensive transport disclaimers.
- Rewrote the experiment protocol language as gates rather than a single leaderboard.
- Strengthened the Distinct5 result narrative: SaMAM is the ArtFID-without-target-movement failure mode; SaMST is the target-movement-with-high-damage failure mode; LBM occupies the low-damage positive-movement region.
- Recompiled `paper_aaai2026.pdf`; page count remains 11. Log scan found no undefined references/citations, fatal errors, or overfull boxes. Remaining warnings are XeLaTeX font-substitution warnings.
- Rendered and inspected page 1 after several line-break fixes; the abstract no longer starts the right column with broken words.
