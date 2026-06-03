# AAAI Figure Audit Memo - 2026-06-03

Scope: figure-only adversarial audit for `SchrodingerBridge/aaai_submission/paper_aaai2026.tex` and the currently referenced figure assets.  
Non-goals: no paper-text edits, no experiment requests, no metric re-interpretation beyond what the figures visibly support.

Inputs inspected:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- `SchrodingerBridge/aaai_submission/framework_figure.pdf`
- `SchrodingerBridge/aaai_submission/fig_qual_grid_ours_vs_samst.png`
- `SchrodingerBridge/aaai_submission/fig_zoom_ours_vs_samst.png`
- `SchrodingerBridge/aaai_submission/fig_ablation_pareto.png`
- `SchrodingerBridge/aaai_submission/fig_weight_sweep_summary.png`
- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.pdf`
- `SchrodingerBridge/aaai_submission/figures/fig_distinct5_time_context.pdf`

## Bottom line

The current paper has enough figure material to tell the story, but the story is not landing cleanly at AAAI review speed. The strongest reviewer attack surface is not "bad plotting taste"; it is that the main figure and the main result figure both ask the reader to decode too much at too small a scale, and several captions are compensating for things the graphics should already make obvious.

## Ranked reviewer attacks

1. `P0 - Figure 1 is overloaded, shrunken by its own aspect ratio, and not self-sufficient.`
   - Location: `fig:framework`, `paper_aaai2026.tex:79-83`, asset `framework_figure.pdf`.
   - Why a reviewer will attack it: this is the paper's method figure, but it currently tries to do three jobs at once: core inference/training graph, SA-SWD intuition, and tokenizer-probe summary. In the compiled PDF it becomes a long horizontal strip with tiny internal text and weak visual hierarchy.
   - Information-density problem: a large page footprint is spent on borders, spacing, repeated rounded boxes, and the bottom legend strip, while the truly important path is compressed.
   - Faithfulness problem: the most important boundary in the method, namely "target-domain latents are training-side only and are not inference-time conditioning inputs," is not obvious from the graphic alone.
   - Caption backfilling symptom: the caption has to explicitly explain the inference/training boundary and arrow semantics because the picture does not settle that question on first read.
   - Likely reviewer phrasing: "The main figure looks poster-like, but I still cannot tell what is actually used at inference."

2. `P0 - Figure 4 is declared as the main Distinct5 result view, but at single-column scale it is too dense to audit and too selective to trust at a glance.`
   - Location: `fig:distinct5`, `paper_aaai2026.tex:381-385`; the text explicitly calls it "the main result view" at `paper_aaai2026.tex:388`; asset `figures/fig_distinct5_pareto.pdf`.
   - Why a reviewer will attack it: the figure packs two panels, four method families, a dashed no-op baseline, operating-point labels, and a "Pareto region" subset into one narrow column. In the compiled PDF, labels and legend become borderline microscopic.
   - Information-density problem: the semantic load is high, but the readable payload is low once the figure is shrunk.
   - Faithfulness problem: panel (b) looks curated rather than inspectable. A reader cannot tell from the graphic alone why these exact LBM points are labeled, why SaMAM is a partial trajectory while LBM is shown as selected operating points, or what rule defines the displayed Pareto subset.
   - Caption backfilling symptom: the caption explains the dashed line and point families, but the asymmetry of what is shown versus omitted still has to be learned from surrounding prose.
   - Likely reviewer phrasing: "This is the main result figure, but the table is doing the real evidentiary work."

3. `P1 - The artifact-quality claim depends on Figure 2 plus Figure 3, but Figure 3 does not visually prove that its crops are matched, representative, or fairly chosen.`
   - Location: `fig:qual_grid` and `fig:zoom`, `paper_aaai2026.tex:342-353`; assets `fig_qual_grid_ours_vs_samst.png` and `fig_zoom_ours_vs_samst.png`.
   - Why a reviewer will attack it: the paper makes a perceptual claim about muddy, grain-like texture failure. The grid suggests it, but the zoom figure is where the paper asks the reader to believe it. Those crops are low-resolution, not box-linked back to the parent images, and not annotated as exemplars versus representative samples.
   - Information-density problem: Figure 2 is fine as a broad qualitative panel, but Figure 3 throws away context while adding almost no audit trail.
   - Faithfulness problem: "matched outputs" and "centered texture crops" are not encoded in the image itself.
   - Caption backfilling symptom: the caption is doing the work of certifying crop alignment and task relevance.
   - Likely reviewer phrasing: "These crops may support the claim, but I cannot verify that they were chosen in a non-cherry-picked way."

4. `P1 - Several figures are burning space on long-strip or low-yield layouts, so the paper pays page cost without getting corresponding evidentiary clarity.`
   - Main cases:
     - Figure 1: extreme horizontal spread plus tiny interior text.
     - Figure 4: two-panel frontier squeezed into one column.
     - Figure 7: much of the plot area is empty because the log-time axis and sparse operating points leave large holes.
   - Reviewer-facing risk: the page looks busy, but the usable information per square inch is lower than it should be.

5. `P2 - Figure 5 exposes hidden selection logic that the paper caption does not acknowledge.`
   - Location: `fig:ablation`, `paper_aaai2026.tex:465-469`; asset `fig_ablation_pareto.png`.
   - Why a reviewer will attack it: the figure title says "Selective ablation (6 of 12 points, +-3 from D0)", which openly reveals filtering, while the paper caption simply says "Destructive ablations."
   - Faithfulness problem: the graphic is more honest than the manuscript wrapper. That mismatch invites a cherry-picking accusation even if the selection was reasonable.

6. `P2 - Figure 6 and Figure 7 read as caption-supported side claims rather than independently valuable main-text figures.`
   - Locations: `fig:sweep` at `paper_aaai2026.tex:475-479` and `fig:timecontext` at `paper_aaai2026.tex:482-486`.
   - Why a reviewer will attack them:
     - Figure 6 reduces a 40-run sweep to two truncated-axis bars, which visually inflates a narrow comparison and hides the shape of the sweep.
     - Figure 7's caption explicitly says it is bounded timing context rather than a closed normalized claim; that is a tell that the figure is not carrying enough standalone evidentiary weight for main-text real estate.
   - Action judgment:
     - Figure 6 should be removed from the main paper, not polished. Its payload fits in one sentence of text and does not justify the page cost.
     - Figure 7 may remain only if the paper still needs bounded timing context in the main text after the next tightening pass.

## What is most likely to trigger AAAI pushback

- `Main-figure ambiguity`: the reader cannot immediately tell what the actual inference contract is.
- `Main-result illegibility`: the paper's declared main result figure is too small and too selective for confident visual auditing.
- `Caption dependence`: the most important qualitative and conceptual clarifications live in captions rather than in figure structure.

## Figure-by-figure keep / fix judgment

- `Figure 1`: keep only if remade. Current version is high-risk.
- `Figure 2`: keep; it is the strongest current qualitative asset.
- `Figure 3`: keep only if crop provenance is made visually explicit.
- `Figure 4`: keep only if promoted to a more legible layout or simplified.
- `Figure 5`: usable, but currently vulnerable on selection honesty.
- `Figure 6`: low priority; currently reads as compressive summary, not convincing evidence.
- `Figure 7`: borderline appendix material in its current form.

## Checklist for the main-figure remake agent

- Make the main figure answer one question first: `what enters at inference, what is learned at training, and where SA-SWD acts`.
- Split off tokenizer probes and diagnostic evidence from the method graph unless they can be shown in a tiny inset without stealing hierarchy.
- Remove the current long-strip footprint; redesign for two-column print readability instead of poster-width storytelling.
- Ensure every label inside the figure is still comfortable at compiled paper scale, not just in the source PDF.
- Encode the "no target image/latent at inference" boundary directly in the graphic, not only in the caption.
- Use one dominant left-to-right path for inference; demote training-only losses and supervision links to a secondary rail or sideband.
- Reduce legend burden: prefer direct labeling over a separate bottom legend strip when possible.
- Eliminate large empty regions and decorative spacing that do not carry method meaning.
- Make visual hierarchy obvious: core path first, auxiliary training signals second, diagnostics last.
- Before finalizing, verify the remade figure inside the compiled `paper_aaai2026.pdf`, not only as a standalone asset.
