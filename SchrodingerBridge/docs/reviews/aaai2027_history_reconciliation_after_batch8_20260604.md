# AAAI 2027 History Reconciliation After Batch 8

Date: 2026-06-04

## Purpose

This note re-audits the active manuscript after the recovery, structure,
mechanism, writing-tone, and manuscript-mouth passes already logged in
`docs/reviews/aaai2027_revision_tracker_20260604.md`.

Scope:

- `SchrodingerBridge/history.md`
- `SchrodingerBridge/docs/reviews/aaai2027_revision_tracker_20260604.md`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`

The goal is concrete: distinguish which June 4 history items are now reflected
in the live manuscript, which remain risky at the paper surface, and what still
belongs to the final Phase-5 style review pass.

## Reconciled Items

- `SaMAM 2250` is the active manuscript boundary throughout the current paper.
  The abstract, page-1 summary caption, main Distinct5 table, and main
  Distinct5 paragraph all exclude later SaMAM checkpoints from the live claim
  surface.
- Distinct5 is now the primary experiment surface, with historical strict-750
  demoted to contextual support and mechanism / negative closures placed later.
- The historical main table is trimmed back to the four core metrics
  (`CLIP-S`, `CLIP-C`, `LPIPS`, `EC`), while artifact-sensitive metrics live in
  the dedicated follow-up table.
- The body no longer carries the rejected Distinct5 scatter or the rejected
  512 qualitative panel. The live qualitative surface is the user-approved 256
  `ours vs SaMST` grid plus the matching zoom crop.
- The mechanism section now uses the bounded endpoint-trained / OMF-side
  transport object rather than reopening the rejected `SaMAM 3000` closure
  surface.
- The metric explanation now distinguishes the paper-facing Distinct5
  `targetwise ArtFID` column from the auxiliary aggregate ArtFID diagnostic,
  which is no longer allowed to leak into the main comparison surface.
- The `SaMST e5/e15` plateau story is back in bounded form: CLIP-S / LPIPS are
  near plateau by e5, while e15 remains the safer conservative endpoint because
  targetwise ArtFID is lower.
- Residual internal workflow phrasing has been reduced substantially: the live
  paper now presents one OT-coupled latent-transport identity rather than
  mixing `stochastic bridge` headline wording back into the active method path.

## Remaining Risks

- The main remaining wording risk is not the old June 4 history mismatch
  anymore; it is reviewer-facing scope tension. The paper now keeps IDT mostly
  split-scoped, but the final adversarial pass should still check whether any
  sentence over-extends that rule beyond the separated art-to-art setting that
  is actually evidenced here.
- The paper surface still intentionally uses the user-selected historical
  256 qualitative packet rather than the older planned Distinct5 qualitative
  panel. This is no longer a missing history item; it is an explicit manuscript
  choice and should only be reopened by direct user instruction.

## Still Open

- Phase 5 is not yet closed. The tracker still calls for a final build/layout
  audit plus the adversarial four-lane review pass after the manuscript is
  coherent.
- The stronger sidecar evidence packets mentioned in `history.md`
  (`SaMAM` final aligned packet, extra stress splits, clustered bootstrap, and
  performance-follow-up work) remain separate from the active paper surface and
  should not be mixed into the next manuscript-only cleanup commit.

## Recommended Next Slice

- Make one narrow manuscript-only commit that:
  - captures the refreshed reconciliation note;
  - records the final wording-tightening pass in the tracker;
  - rebuilds the PDF;
  - logs readiness for the final four-lane review pass.

- Keep experiment expansion, new remote packets, and performance follow-up in
  separate later slices so the active manuscript boundary stays stable.
