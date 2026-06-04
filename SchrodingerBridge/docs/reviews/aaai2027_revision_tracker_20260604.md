# AAAI 2027 Revision Tracker

Date: 2026-06-04

## Objective

Recover the intended AAAI manuscript state after a destructive git operation,
then continue the paper rewrite in small, reviewable phases with frequent git
checkpoints.

## Recovery Baseline

Authoritative recovery sources:

- `SchrodingerBridge/aaai_submission_recovery_20260604/sources/paper_aaai2026.before_blob_restore.tex`
- `SchrodingerBridge/aaai_submission_recovery_20260604/sources/recovered_blob_c190c5c2.tex`
- `SchrodingerBridge/aaai_submission_recovery_20260604/sources/recovered_blob_e90d.tex`
- `SchrodingerBridge/docs/reviews/aaai2027_writing_iteration_R20260604_round18_nonreview.md`
- `SchrodingerBridge/docs/reviews/aaai2027_claim_evidence_ledger_20260604.md`

Current recovery conclusion:

- The stash is not the relevant paper source.
- The `aaai_submission_recovery_20260604/sources/` directory contains real
  recovered local manuscript states from the object database.
- The current `aaai_submission/paper_aaai2026.tex` is a mixed state:
  it already includes the recovered IDT-calibrated rewrite, but it also
  reintroduces a later `SaMAM 3000 / 10.2h / 394.8` branch that the June 4
  writing cleanup explicitly rejected for the active manuscript.
- Until a clean rerun supersedes the boundary, the active manuscript must keep
  `SaMAM 2250` as the valid Distinct5 row.

## Phase Plan

### Phase 0: Restore the Intended Manuscript Baseline

Goal:

- Reconstruct the active manuscript state that matches the June 4 writing
  cleanup and claim ledger.

Tasks:

- Use `paper_aaai2026.before_blob_restore.tex` as the main recovery reference.
- Remove mixed-in superseded `SaMAM 3000` manuscript content from the active
  paper unless a newer clean packet is intentionally adopted later.
- Reconcile the current paper with the stated June 4 active boundary:
  `SaMAM 2250` only.

Verification gate:

- No active-manuscript text contains `SaMAM 3000`, `SaMAM 2500`, `10.2h`, or
  `394.8` unless the active evidence policy is explicitly changed.
- Table 1 / main Distinct5 paragraph match the 2250 boundary.

Commit checkpoint:

- `checkpoint: recover intended June 4 manuscript baseline`

### Phase 1: Unify the Metric Mouth

Goal:

- Make the ArtFID / IDT story internally consistent.

Tasks:

- Main text uses `targetwise ArtFID` for Distinct5 target-style diagnosis.
- Keep `aggregate ArtFID ~= 1.0` only as an art-domain / identity diagnostic.
- Make `IDT` the unchanged-image floor in the wording, not a throwaway baseline.

Verification gate:

- Numbers and captions agree with the intended metric type.
- No paragraph or caption mixes `aggregate ArtFID` and `targetwise ArtFID`
  as if they mean the same thing.

Commit checkpoint:

- `checkpoint: unify IDT and ArtFID manuscript wording`

### Phase 2: Align Figure and Table Surface

Goal:

- Make the paper-facing figures and tables match the active evidence boundary.

Tasks:

- Keep the first-page Distinct5 transfer-only summary aligned with the current
  active row policy.
- Regenerate Table 4 / main Distinct5 table from one CSV source rather than
  hand-filled values.
- Merge the qualitative grid and zoom crop into one artifact-diagnosis figure
  if that figure is still missing or inconsistent.

Verification gate:

- Figure 1, Table 4, and the underlying CSV rows agree.
- No figure or caption implies a broader SaMAM closure than the active packet.

Commit checkpoint:

- `checkpoint: align page-1 figure and main table`

### Phase 3: Rewrite the Experiment Structure

Goal:

- Recast the results section around the evidence boundary rather than scattered
  claims.

Tasks:

- Organize the experiments into three layers:
  - primary Distinct5 evidence
  - contextual strict-750 evidence
  - mechanism / negative closures
- Keep negative evidence narrow:
  - endpoint-only pointwise supervision does not recover the current frontier
  - semantic-vs-random SA-SWD is a tested design choice, not a positive novelty
    closure
  - tokenizer localization remains a local executor-vs-style-branch boundary
- Keep SaMST wording at the June 4 bounded version:
  e5/e15 plateau on CLIP-S / LPIPS, e15 lower targetwise ArtFID, still
  high-damage.

Verification gate:

- Introduction, experiments, discussion, and conclusion all use the same claim
  boundary.
- No section implies full convergence or universal superiority where the ledger
  forbids it.

Commit checkpoint:

- `checkpoint: rewrite experiment narrative to evidence-bound structure`

### Phase 4: Integrate Path-Stability Only if It Passes the Gate

Goal:

- Decide whether the same-family H-family path-stability packet is admissible
  as active mechanism support.

Tasks:

- Audit the landed packet under
  `docs/experiments/2026-06-03-path-stability-probe/`.
- Check whether the exact mechanism statement in the paper matches the packet's
  actual closure status.
- If the packet is weaker than the paper's wording, demote the paper claim to a
  bounded observation instead of stretching the evidence.

Verification gate:

- Same-family mechanism wording appears in the paper only if the linked packet
  cleanly supports it.
- Otherwise the paper keeps the mechanism read narrow.

Commit checkpoint:

- `checkpoint: settle path-stability evidence boundary`

### Phase 5: Build, Layout, and Final Adversarial Pass

Goal:

- Rebuild the paper, scan layout, then run the four-agent review loop only
  after the manuscript is coherent again.

Tasks:

- Run `build_paper.bat`.
- Render and inspect the first 10 pages plus any pages with large figure/table
  floats.
- After the paper is stable, run the four review lanes:
  experiment auditor, theory/claim auditor, figure/layout auditor, hostile AAAI
  reviewer.

Verification gate:

- No stale forbidden text, TODOs, placeholder markers, or blank-page
  regressions.
- Review feedback is integrated in small follow-up commits.

Commit checkpoint:

- `checkpoint: manuscript rebuild and adversarial review pass`

## In-Progress Log

- 2026-06-04: Recovery audit found the real manuscript trail in
  `aaai_submission_recovery_20260604/sources/`, not in the stash.
- 2026-06-04: The current paper was identified as a mixed state that already
  carries rejected `SaMAM 3000` active-manuscript content.
- 2026-06-04: Active baseline chosen for the next edit pass:
  June 4 writing-cleanup boundary plus recovered pre-restore manuscript text.
- 2026-06-04: Commit `3e746a1db` fixed the June 4 manuscript boundary, and the
  multi-lane history gap audit is now running.
- 2026-06-04: Commit `2bc88d8ef` synced the page-1 Distinct5 summary figure to
  the active `SaMAM 2250` boundary.
- 2026-06-04: Wrote `docs/reviews/aaai2027_history_gap_audit_20260604.md` to
  order the next edit batches: cleanup/checklist regressions first, then the
  Distinct5 `SaMAM 2250` plus `SaMST e5/e15` claim surface, then structure and
  mechanism rewrites.
- 2026-06-04: Batch 1 source cleanup removed the stale
  `no-op-adjusted`/`Euler steps`/`z_K` strings, dropped the regressed
  `FloatBarrier` and checklist `\clearpage`, restored the checklist
  significance item to `Partial`, and rebuilt the paper to a `13`-page PDF
  without the end-of-paper blank-page regression.
- 2026-06-04: Batch 2 tightened the Distinct5 claim surface: the abstract and
  main Distinct5 paragraph now use the trusted `SaMAM 2250` boundary, the
  closed `SaMST e5/e15` plateau wording is back, the main table dropped the
  `SaMAM 2000` row and relabeled `tw-ArtFID`, and `fig_distinct5_pareto.*`
  was regenerated so the SaMAM surface is a single trusted `2250` point rather
  than a manuscript-visible checkpoint curve.
- 2026-06-04: Batch 3 figure pass followed user feedback on information value:
  the body Distinct5 two-panel scatter was removed as redundant with the
  page-1 summary, the rejected 512 visual-alignment panel was discarded, the
  historical 256-resolution `ours vs SaMST` grid plus texture crops were
  restored for the artifact-diagnosis section, and the rebuilt PDF returned to
  a stable 13-page layout.
- 2026-06-04: Batch 4 structure pass moved `Distinct5-512 stress benchmark`
  ahead of the historical strict-750 packet as the primary experiments
  surface, rewrote the protocol-hierarchy lead to make that ordering explicit,
  and trimmed the historical main table back to the four core metrics because
  the artifact-sensitive rows now live only in the dedicated follow-up table.
- 2026-06-04: Batch 5 mechanism-boundary pass rewrote the method object toward
  an OT-coupled endpoint-trained renderer, restored the landed local `L e1`
  reading that executor-side refresh beats style-side refresh alone on the
  matched localization packet, absorbed the bounded same-family `H`
  path-stability packet as local kinetic/path support, and marked the
  semantic-vs-random SA-SWD control as quality-only rather than positive
  novelty closure.
- 2026-06-04: Batch 6 writing-tone pass rewrote the core method narrative as
  `CFM + OT coupling + adaptive-projection terminal matching`, reduced the
  tokenizer graveyard feel in the method section, upgraded Theorem 1 and timing
  prose from defensive caveats to principled bounded claims, strengthened the
  IDT-floor community framing in the introduction and conclusion, and tightened
  the 256 qualitative captions around muddy grain artifacts versus cleaner
  brush continuity.
- 2026-06-04: Batch 7 manuscript-mouth pass separated targetwise ArtFID from
  the auxiliary aggregate ArtFID diagnostic directly in the metric explanation,
  removed more internal workflow wording from the method/discussion/conclusion,
  and wrote `aaai2027_history_reconciliation_after_batch8_20260604.md` to mark
  which June 4 history items are now recovered versus what remains for the
  final Phase-5 review gate.
- 2026-06-04: Batch 8 identity-tightening pass removed the last live
  `stochastic bridge` headline phrasing from the active method path and aligned
  the conclusion with the same OT-coupled latent transport model identity used
  in the body.
- 2026-06-04: Batch 9 accept-surface pass consolidated the 256 artifact
  evidence into a matched red-box crop panel so the full-view boxes and zoomed
  excerpts point to the same failures, inspected but rejected a switch to a
  `SaMam-256`-led main qualitative surface because it weakened the direct
  `ours vs SaMST` artifact diagnosis, tightened the Distinct5 / cost /
  conclusion wording around the compact efficient regime rather than open-ended
  SOTA language, and confirmed a rebuilt
  `aaai_submission/paper_aaai2026.pdf` remains available as the current
  12-page manuscript artifact. Next gate: rerun the hostile figure-and-claims
  review to decide whether the new crop choices and bounded efficiency story
  are finally strong enough for the accept pass.
- 2026-06-04: Batch 10 claim-and-layout pass further tightened claim hygiene
  around the Distinct5 scope, timing wording, and historical claim bounds,
  added a compact Distinct5 pair-level sanity table, changed artifact panel
  crop B to the `Hayao -> van Gogh` line-boundary contamination case, switched
  the artifact insertion to `figure*`, and verified that the full zoom row is
  visible on a single page in the rebuilt manuscript. The four review agents
  are now back in re-review against this tightened figure/table/claim surface.
- 2026-06-04: Batch 11 final review gate reached `ACCEPT` from Gibbs
  (layout), Nietzsche (claim hygiene), Boyle (hostile reviewer), and Leibniz
  (visual artifact inspector) after adding `SaMST e15` to the Distinct5
  pair-level same-split sanity table and finalizing the artifact `figure*`
  panel.
