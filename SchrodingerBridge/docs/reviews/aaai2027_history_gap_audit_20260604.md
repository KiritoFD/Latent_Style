# AAAI 2027 History Gap Audit

Date: 2026-06-04

## Scope

This note compares the current active manuscript surface in
`SchrodingerBridge/aaai_submission/paper_aaai2026.tex` against the June 4
history/review trail that already describes paper edits as landed:

- `SchrodingerBridge/history.md`
- `SchrodingerBridge/docs/reviews/aaai2027_claim_evidence_ledger_20260604.md`
- `SchrodingerBridge/docs/reviews/aaai2027_writing_iteration_R20260604_round18_nonreview.md`
- the four audit lanes run on 2026-06-04

The purpose is narrow: identify which historically completed manuscript changes
did not survive into the current paper, then order the next edit batches.

## Current Boundary Snapshot

Two important recovery steps are already closed and should not be reopened:

- the active paper path excludes the anomalous `SaMAM 3000 / 10.2h / 394.8`
  branch from the main manuscript boundary;
- commit `2bc88d8ef` synced the page-1 Distinct5 summary figure to the active
  `SaMAM 2250` boundary.

The remaining work is not about finding new experiments first. It is mostly
about restoring already-decided paper wording, structure, figure surfaces, and
claim boundaries.

## Open Gaps

### P0. Results Order Is Still Not Distinct5-First

History says the paper was already reorganized around:

1. primary Distinct5 evidence;
2. contextual historical strict-750 evidence;
3. mechanism and negative closures.

The current experiments section still opens with `Historical strict-750
comparison`, then `Artifact-sensitive diagnosis`, and only later reaches
`Distinct5-512 stress benchmark`. That keeps contextual evidence in front of
the primary claim surface and is the highest-level structural mismatch still
present.

Required edit batch:

- move Distinct5 to the front of the experiments section;
- demote strict-750 to contextual support;
- keep mechanism/negative closures after the main Distinct5 result surface.

### P0. The SaMAM Boundary Is Still Broader Than The Active Manuscript Policy

The current manuscript boundary is `SaMAM valid through 2250 only`, with later
closure packets excluded. That policy still has not propagated cleanly through
the paper text.

Open regressions:

- the abstract still says `reproduced SaMAM checkpoints`, which reads broader
  than the active row policy;
- the Distinct5 figure caption still describes a `SaMAM curve` rather than a
  bounded measured-point surface;
- the main Distinct5 table still contains both `SaMAM 2000` and `SaMAM 2250`
  rows instead of a single active manuscript row;
- the timing figure still refers to a `currently indexed partial curve`;
- the main Distinct5 paragraph still says SaMAM stays below IDT `throughout the
  evaluated run`, which is broader than the current `through 2250` policy.

Required edit batch:

- rewrite all SaMAM mentions as bounded measured checkpoints or point
  estimates;
- drop the `2000` row from the active main table;
- remove `curve` and `throughout the evaluated run` phrasing from the active
  manuscript path.

### P0. Round-18 Cleanup Regressions Are Back In The TeX

The June 4 round-18 cleanup explicitly removed stale wording and layout-risk
markers from the active manuscript. Several of those edits have regressed:

- `no-op-adjusted` is back in multiple places;
- `Euler steps` is back in the manuscript;
- `z_K` is back in the manuscript;
- the `Internal convergence reference` paragraph is still present even though
  the history says it had been removed from the live paper path;
- `\FloatBarrier` is back near the end of the paper even though round 18
  removed it to prevent the blank-page regression.

Required edit batch:

- restore the round-18 text cleanup exactly;
- rerender the PDF and verify there is no blank or half-empty page regression
  after removing the stale float barrier again.

### P1. The Closed SaMST e5/e15 Plateau Packet Is Missing From The Results Story

The review trail says the active paper had already been tightened to the closed
`SaMST e5/e15` packet:

- transfer CLIP-S and LPIPS differ by less than `0.004 / 0.002`;
- `e15` lowers targetwise ArtFID from `465.7` to `444.5`;
- the conclusion remains narrow: SaMST can clear the IDT floor, but only in a
  high-damage regime.

The current paper still surfaces only a single `SaMST e15` operating point plus
generic high-damage wording. The plateau narrative did not survive into the
main Distinct5 paragraph or the efficiency/timing discussion.

Required edit batch:

- restore the closed `e5/e15` wording in the Distinct5 paragraph;
- restore the same bounded plateau wording in the efficiency section;
- keep the claim narrow and avoid calling SaMST fully converged.

### P1. Distinct5 Targetwise ArtFID Labeling And Table Surface Are Only Partially Restored

The history trail says the paper had already tightened Distinct5 surface labels
around `targetwise ArtFID` and a more explicit main-table header surface.

Current misses:

- the page-1 caption still says only `high ArtFID`;
- the main Distinct5 table still uses a generic `ArtFID` column header;
- the main table does not yet surface the later `tw-ArtFID` / status-style
  caveat language that history says had already been pushed into the table
  surface itself;
- the `one CSV source` table regeneration path is still only partially reflected
  in the active paper surface.

Required edit batch:

- normalize the visible paper language to `targetwise ArtFID` where the history
  says that distinction was already landed;
- update the main table surface before doing any broader prose rewrite around
  ArtFID.

### P1. The Distinct5 Audited Qualitative/Artifact Figure Surface Regressed

History says the older strict-750 qualitative grid and separate zoom crop were
first merged into an artifact-diagnosis figure and later replaced by the
audited Distinct5 visual panel used in the June 4 paper-safe surface.

The current paper still carries the older pair:

- a full `5-by-5` qualitative grid figure;
- a separate centered texture-crop figure.

The audited Distinct5 visual-alignment panel did not survive into the live
manuscript surface.

Required edit batch:

- replace the old grid-plus-zoom pair with the reviewed Distinct5 artifact
  figure surface that the June 4 writing trail treats as active.

### P1. The Mechanism Section Still Mixes In Superseded Bridge/Object Language

The claim ledger narrowed the safe object to an endpoint-trained latent
renderer / OMF-style path for the active Distinct5 story. The current draft
still drifts back toward stronger bridge wording.

Open regressions:

- the core method framing still says the model transports latents along a
  stochastic bridge;
- the endpoint-only negative closure is narrated inside a flow/transport object
  that blurs the narrower endpoint-supervision claim;
- the tokenizer-localization packet has been diluted into generic successor
  prose instead of the landed local `L e1` executor-vs-style-branch outcome;
- the same-family Distinct5 path-stability packet is not yet the main mechanism
  evidence surface even though the review trail says that packet landed;
- the SA-SWD semantic-vs-random section keeps the narrow conclusion but drops
  the provenance caveat that the random arm is `quality_only` and unusable for
  fair wall-clock comparison.

Required edit batch:

- rewrite the mechanism wording to the active endpoint-trained renderer object;
- restore the localized tokenizer and same-family path-stability conclusions in
  their bounded forms;
- keep SA-SWD framed as a retained design choice rather than a cleanly closed
  semantic-axis win.

### P1. Checklist And Layout Fixes Did Not Survive

Several small but already-landed paper fixes are still absent from the current
source:

- the bibliography-to-checklist `\clearpage` removal did not survive;
- the checklist still contains the stale formal-section reference;
- the significance checklist item regressed from `Partial` back to `Yes`;
- the historical strict-750 main table still uses the oversized
  `\resizebox{\textwidth}{!}` surface that history says had already been
  removed.

These are not the central scientific gaps, but they are easy regressions and
should be cleared early because they affect the rendered paper surface.

Required edit batch:

- fix these source-level regressions in the same batch as the round-18 cleanup
  pass;
- rebuild and re-render the PDF immediately afterward.

## Ordered Edit Batches

### Batch 1: Cleanup And Safety Restoration

- restore round-18 cleanup-only removals;
- fix checklist/layout regressions;
- keep the active PDF stable after rebuild.

### Batch 2: Distinct5 Main Claim Surface

- enforce the `SaMAM 2250 only` policy everywhere;
- restore the SaMST `e5/e15` plateau wording;
- tighten visible Distinct5 `targetwise ArtFID` labeling.

### Batch 3: Results Structure And Figure Surface

- reorder experiments to Distinct5 first;
- replace the stale qualitative/zoom pair with the reviewed Distinct5 artifact
  figure surface;
- finish the main table/header cleanup.

### Batch 4: Mechanism And Negative-Closure Boundaries

- remove superseded bridge/object language;
- restore the localized tokenizer result;
- surface the landed same-family path-stability packet in bounded form;
- keep endpoint-only and SA-SWD conclusions narrow.

## Recommendation

Do not start by polishing sentences globally. The correct next move is to clear
Batch 1 and Batch 2 first, because they restore the active manuscript boundary
and the main claim surface before any larger structural rewrite.
