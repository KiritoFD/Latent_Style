# AAAI 2027 adversarial review round 2

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Review roles: theory/method (Kant), experiments (Faraday), figures/layout (Wegener), writing (Cicero)

## Consensus

The paper now has a viable main thesis: art-to-art Style-ID transfer should first beat the unchanged-image control. The remaining acceptance risk is not lack of a core result, but whether the paper overstates the scope of a single CLIP-separated stress split and whether the method prose implies stronger transport theory than the implementation proves.

## Applied in this round

- Rewrote the abstract around IDT as an evidence contract rather than a raw metric table.
- Bounded LBM theory language: endpoint-supervised latent editing, not solved stochastic bridge or supervised continuous-time transport.
- Replaced over-strong path language with velocity/execution budget language.
- Recast terminal SA-SWD as a W1-style sorted-projection patch discrepancy, not an unbiased standard SWD estimator.
- Clarified that residual MSE/Huber/L1 variants are not active in headline Distinct5 rows.
- Added a `Changed` column to the representation/routing/queue table to avoid attributing queue or routing gains solely to tokenizer capacity.
- Moved the Distinct5 visual panel immediately after the main table so the primary qualitative evidence supports the main Distinct5 claim.
- Regenerated the page-1 figure so the right panel explicitly says targetwise ArtFID.
- Rewrote the conclusion as a methodological statement: the unchanged image is the null hypothesis, and tokenizer quality is executability after rendering.

## Still open for Dalton

These are queued to Dalton and should not block the current writing pass:

1. Final/tuned SaMAM Distinct5 packet with full/transfer CLIP-S, LPIPS, targetwise ArtFID, per-image rows, train time, and inference time.
2. Clustered bootstrap for Distinct5, at least source-image / direction clustered intervals for CLIP-S and LPIPS.
3. Two additional fixed-rule WikiArt stress splits with IDT + LBM-F/K first.
4. Semantic-axis vs random-axis SA-SWD matched run.
5. Fixed executor/tokenizer swap or factorization ablation.
6. SaMST e5 standalone targetwise ArtFID artifact closure if e5 remains convergence evidence.

## Current risk after patch

- Single-split criticism remains the main experimental risk until the additional fixed-rule stress splits land.
- SaMAM remains a current reproduced checkpoint estimate until Dalton produces a complete paired packet.
- Figure 5 is still large for a compatibility figure, but it no longer precedes the Distinct5 qualitative evidence.
- The framework figure remains serviceable but could still benefit from a small sorted-projection / terminal matching inset.

## Next review gate

Run another four-reviewer pass only after one of these happens:
- Dalton returns a final SaMAM packet.
- Additional stress split results are available.
- The framework figure is redesigned.
- A major method/tokenizer experiment changes the claimed mechanism.
