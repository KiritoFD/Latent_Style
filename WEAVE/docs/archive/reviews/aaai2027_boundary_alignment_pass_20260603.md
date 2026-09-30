# AAAI 2027 Boundary Alignment Pass

Date: 2026-06-03  
Scope: manuscript-boundary pass after the adversarial rewrite hit list.  
Target manuscript: `aaai_submission/paper_aaai2026.tex`

## Purpose

This note records the first explicit manuscript-alignment pass after:

- the repaired endpoint-only trio was closed negatively;
- the semantic arm of the matched SA-SWD axis packet completed;
- the matched random-axis control remained open;
- normalized time-to-parity evidence remained open.

The goal of this pass is not to finish the paper. It is to remove the most
obvious places where the manuscript outran the current evidence.

## Sections changed

### 1. Abstract

Adjusted:

- removed the strongest cross-method timing rhetoric from the abstract;
- kept the strict-750 inference runtime;
- rewrote the Distinct5 sentence so it is explicitly about:
  - `Distinct5-512`,
  - the unchanged-image prior,
  - and `currently reproduced points`.

Reason:

- Gate C is still open;
- Distinct5 cannot be written as a universal benchmark win.

### 2. Contributions list

Adjusted:

- downgraded the formal-analysis bullet from `direct empirical validation` to
  `empirical checks where available` with explicit scope limits;
- downgraded the SA-SWD bullet from closed novelty language to a proposed
  semantic-aligned terminal design;
- replaced the old comparative efficiency bullet with explicit operating-point
  cost accounting language;
- scoped the Distinct5 frontier bullet to `currently reproduced points`;
- scoped the S2WAT bullet to the reproduced historical comparison.

Reason:

- Gate B is still open;
- Gate C is still open;
- theorem support is design-grounding, not full empirical closure.

### 3. Method

Adjusted:

- rewrote the Eq. (5) / local-loss paragraph so it no longer says the current
  evidence `clearly closes` the endpoint-side story;
- changed it to the narrower statement that current evidence favors the OT +
  `W1`-style mainline over isolated endpoint-only pointwise supervision;
- changed the complementary-roles paragraph from a settled semantic-axis
  statement to an `ongoing matched control` statement;
- rewrote the SA-SWD mechanism prose as the design used in the current
  mainline, not as a proven necessity;
- softened the latent-metric paragraph so it no longer reads like a broad
  theorem about latent-space MSE;
- changed the formal-analysis bridge sentence from `direct empirical
  validation` to `empirical checks where available`.

Reason:

- the repaired endpoint trio is a negative endpoint-only closure, not a proof
  of universal `W1` optimality;
- the semantic-vs-random axis result is not closed yet.

### 4. Historical strict-750 experiment text

Adjusted:

- removed `retraining cost` from the core interpretation paragraph under the
  main historical table;
- narrowed the claim there to a quality/frontier reading within the reproduced
  historical table;
- changed the cost-table caption to explicit `recorded operating-point` wording;
- softened the cost paragraph so it reports practical footprint rather than a
  generalized comparative speed win.

Reason:

- Gate C is still open.

### 5. Distinct5 subsection

Adjusted:

- rewrote the subsection opening so Distinct5 is first presented as a
  `metric-stress benchmark rather than a universal art benchmark`;
- pinned the frontier language to `currently reproduced points`;
- kept `idt` / no-op front and center;
- moved timing language into explicit `recorded operating-point wall clock`
  framing rather than normalized efficiency language.

Reason:

- Distinct5 is a stress split, not a general AST leaderboard;
- Gate C is still open.

### 6. Mechanism-ablation interpretation

Adjusted:

- rewrote the SA-SWD ablation interpretation so it says terminal distribution
  alignment is a primary style driver in the current mainline, not a proof of
  semantic-axis novelty.

Reason:

- destructive ablations close `terminal matching matters`;
- they do not close `semantic axis selection is necessary`.

### 7. Discussion, limitations, conclusion, checklist

Adjusted:

- added scope pins to the main Distinct5 frontier wording in Discussion and
  Conclusion;
- weakened the Distinct5 limitations sentence from `already decisive` to
  `informative on this split`;
- changed the reproducibility-checklist theoretical-contribution sentence from
  `proofs and experimental validation` to `proofs and partial empirical
  support`.

Reason:

- theorem closure is still partial;
- the paper must not read broader than the current experiment state.

## What remains intentionally unresolved after this pass

1. **Gate B is still open.**
   - The matched random-axis control is still running.
   - The paper therefore still cannot claim that semantic axis selection is
     proven necessary.

2. **Gate C is still open.**
   - No normalized time-to-parity artifact exists yet.
   - Timing language remains deliberately downgraded to operating-point
     bookkeeping.

3. **Gate D is only partially addressed.**
   - This pass removes several obvious overclaims.
   - A second manuscript pass is still required after Gate B closes and after
     the timing artifact is available.

## Practical result

After this pass, the manuscript should be substantially harder to attack on
three fronts:

- broad latent-metric overclaim,
- premature semantic-axis novelty closure,
- and unfair speedup rhetoric.

It is **not** yet review-safe enough to move beyond `weak_reject` on its own,
because the missing matched SA-SWD control and the missing normalized timing
artifact are still real evidence gaps.
