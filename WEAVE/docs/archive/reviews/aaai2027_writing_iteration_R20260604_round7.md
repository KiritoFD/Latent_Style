# AAAI 2027 writing iteration round 7

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Figure script: `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_visual_panel.py`

## Writing changes

- Retitled the paper from "Should Beat" to "Must Beat" to make the evaluation contract explicit.
- Rewrote the abstract around the CLIP-S-scoped null hypothesis: an art-to-art Style-ID method that cannot beat copying has not demonstrated target-style movement under that evaluator.
- Reframed Distinct5-512 as a falsification test rather than a leaderboard.
- Made the SaMAM/SaMST/LBM contrast sharper:
  - SaMAM can improve targetwise ArtFID while falling below IDT in transfer CLIP-S.
  - SaMST clears IDT but pays high LPIPS and targetwise ArtFID.
  - LBM clears IDT in the low-displacement region with minute-scale retained-checkpoint time.
- Rewrote the introduction so the argument is: raw target-style scores are underspecified, IDT supplies the counterfactual, Distinct5 exposes failure modes, and LBM is the compact proof point.
- Rewrote method overview and style-control prose around executable representation: tokenizer/code geometry matters only if it survives the content-conditioned renderer.
- Reorganized experiment prose into gates: IDT movement, cost of movement, and legacy compatibility.
- Tightened discussion and conclusion around the reporting standard: movement over IDT first, then LPIPS, ArtFID, and compute as the price of that movement.

## Figure changes

- Regenerated Figure 3 with two audited manifest examples:
  - Impressionism -> Minimalism as a target-movement case.
  - Rococo -> Ukiyo-e as a lower-displacement contrast case.
- Rewrote Figure 3 caption to avoid per-example overclaiming; aggregate claims remain anchored in Table 1.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no unresolved citations/references, no fatal errors, no overfull boxes.
- Remaining warnings: XeLaTeX font substitution warnings only.
- Rendered pages inspected: page 1, page 6, page 9, page 10, page 11.

## Reviewer pass

- Godel, area-chair/writing lens: 7.8/10.
- Kant, statistics/experimental-validity lens: 7.0/10.
- Wegener, method/math lens: 6.8/10.
- Cicero, figures/layout lens: 7.0/10.

Applied follow-up fixes:
- Scoped the abstract opener to CLIP-S-based art-to-art evaluation.
- Changed SaMAM abstract/intro/related-work mentions to point estimates pending paired packets.
- Softened the retained-checkpoint timing claim by removing the direct "hundreds of baseline minutes" contrast from the abstract.
- Renamed `row check` to `row sign` and clarified that row-resampled signs are not confidence intervals over independent samples.
- Added the ArtFID clarification that lower ArtFID is not target-direction evidence.
- Added an explicit stop-gradient empirical endpoint sampler equation, `tilde z_1 ~ q_s(.|z_0)`.
- Renamed optional local `L_trans` to `L_res` and removed "transport loss" wording from the active method path.
- Replaced residual "terminal SWD" and "w/o kinetic" labels with terminal projection and endpoint velocity penalty wording.
- Removed tiny per-cell metric labels from Figure 3 and regenerated the panel with a more diagnostic target-movement row.
- Rebuilt the PDF after follow-up edits; the final round-7 continuation remains 11 pages with no unresolved references/citations, no fatal errors, and no overfull boxes.
- Removed temporary `tmp_round7_review` and `tmp_round7_visual_review` directories.

## Remaining risks

- Distinct5 is still one CLIP-separated WikiArt stress split; Dalton/Faraday should complete additional fixed-rule splits before the paper claims broad benchmark generality.
- SaMAM remains a point-estimate baseline until Dalton provides a full aligned packet.
- Row-resampled intervals remain exploratory; clustered source/style bootstrap is still needed for formal uncertainty.
- Figure 3 is now more diagnostic, but LBM remains visually conservative in some target directions; this should be discussed as low-motion transfer rather than hidden.
