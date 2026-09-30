# AAAI 2027 Writing Iteration R20260604 Round 14

Date: 2026-06-04

## Scope

This round focused on writing quality and claim hygiene after the SaMAM 3k Distinct5 packet became usable. No new experiments were launched by the main thread. Dalton remains responsible for sidecar experiment monitoring.

Main artifacts:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_page1_summary.pdf`
- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_visual_alignment_grid_panel.jpg`

## Reviewer Pass

Four adversarial reviewer agents were asked to review the rebuilt draft.

| Reviewer lens | Score before fixes | Main risk found |
|---|---:|---|
| Area-chair / writing | 7.2/10 | Strong thesis, but Table 1 and Figure 3 still smelled like internal audit artifacts. |
| Method / math | 7.6/10 | IDT sounded too much like a formal statistical null; bridge/OT and tokenizer claims needed tighter scope. |
| Statistics / validity | 7.2/10 | SaMAM 3k is closed for metrics/train wall, not pure generation timing; LBM-F vs SaMAM is mainly ArtFID/cost, not CLIP/LPIPS. |
| Figures / layout | 7.2/10 | Page 1 works, but Figure 1 label choreography and Figure 3 SaMAM mismatch weakened persuasion. |

## Applied Fixes

1. Reframed IDT from a broad "null hypothesis" to an operational CLIP-S floor.
   - Abstract now starts with the CLIP-S-based art-to-art Style-ID scope.
   - Introduction/conclusion use "operational floor" and "calibration control."

2. Tightened bridge/OT/math wording.
   - Related work now says OT provides diagnostic language, not endpoint-map estimation.
   - The Sinkhorn diagnostic explicitly is not evidence of transport optimality.
   - Terminal loss wording is now "projection-statistic matching" rather than broad distribution matching.

3. Reworked Table 1 into paper prose rather than lab packet shorthand.
   - Removed `Pkt.` / `cl.` / `part.` / `sel.` column.
   - Added a table note: SaMAM 3k is closed for aligned metrics and training wall; pure generation timing remains unavailable.
   - Timing caption now states operating-point wall time, not search-inclusive or hardware-normalized time-to-result.

4. Removed the mismatched SaMAM visual column from Figure 3.
   - The prior visual column used a 2.25k audited crop while Table 1 reports 3k metrics.
   - Figure 3 now compares Source / IDT / SaMST / LBM-F / LBM-K only.

5. Improved Figure 1.
   - Added a faint below-IDT region.
   - Cleaned IDT label and LBM/SaMAM labels.
   - Kept training-time labels inside ArtFID bars.

6. Updated the next-review gate.
   - The gate now records SaMAM 3k metric+train packet closure and the missing generation-timing limitation.

## Current Claim State

Safe headline:

- IDT is an operational CLIP-S floor for art-to-art Style-ID transfer.
- SaMAM 3k clears IDT, but with much higher targetwise ArtFID and much longer recorded checkpoint-training time than LBM-F.
- LBM-F matches SaMAM 3k in observed transfer CLIP-S/LPIPS while occupying the lower targetwise-ArtFID and minute-scale operating region.
- SaMST clears IDT more strongly in CLIP-S but pays high LPIPS and high targetwise ArtFID.

Unsafe headline:

- Do not claim SaMAM fails to move.
- Do not claim same-scope SaMAM inference speed for 3k.
- Do not claim LBM-F is a statistically significant CLIP/LPIPS win over SaMAM 3k.
- Do not describe LBM as an OT-map or Schrödinger-bridge estimator.

## Verification

- Rebuilt `paper_aaai2026.pdf` successfully.
- `pdfinfo`: 11 pages, letter paper, PDF 1.7.
- LaTeX log scan found no `Overfull`, undefined references, undefined citations, fatal errors, or missing glyphs.
- Rendered pages with `pdftoppm`; inspected page 1, Table/Figure 3 page, artifact page, discussion/reference pages.

## Remaining Risks

1. Figure 3 still makes LBM-F/K look visually pale in the chosen examples. It is now claim-safe, but not maximally persuasive.
2. Figure 2 is readable but still visually equal-weights style request, execution, and training-only pressure. A stronger version should make the blue inference path dominant.
3. Distinct5 is a CLIP-selected and CLIP-evaluated stress split; the paper states this, but additional fixed-rule splits would strengthen generality.
4. The SaMAM 3k packet lacks pure generation ms/img, so any inference-speed comparison must stay out of the main claim.
5. The abstract is stronger than before but still dense; future polishing can reduce numeric load if Table 1 and Figure 1 carry more of the data.

## Next Writing Targets

1. Improve Figure 3 example selection using audited rows where LBM shows clearer target movement without increasing damage.
2. Redraw Figure 2 with a more dominant inference path and lighter training-only band.
3. Compress late ablation prose so the paper reads less like an experiment log after the main result.
4. If Dalton supplies new closed packets, update Table 1 only after aligned metrics, targetwise ArtFID, and timing scope are documented.
