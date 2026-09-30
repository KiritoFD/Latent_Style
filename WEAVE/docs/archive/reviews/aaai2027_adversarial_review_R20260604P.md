# AAAI 2027 Adversarial Review Round R20260604P

Date: 2026-06-04

## Scope

This review round starts after the Distinct5-512 evidence pass that added
paired IDT bootstrap intervals for retained transfer rows and cleaned the
SaMAM/IDT wording boundary.

Review target:

- Manuscript: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Writing gate: `SchrodingerBridge/docs/reviews/aaai2027_writing_gate_R20260603O.md`
- Bootstrap packet: `SchrodingerBridge/docs/experiments/distinct5_512_20260602/bootstrap/README.md`

## Pre-review Verification

- `cmd /c build_paper.bat` succeeds.
- PDF remains 13 pages.
- LaTeX log search finds no undefined references/citations, no overfull boxes,
  and no float errors.
- Pages 1, 7, 8, 9, and 13 were rendered for visual inspection.
- Table 4 was checked after the compact-header resize fix.
- Bootstrap script compiles with `py -3 -m py_compile`.
- Bootstrap CSV contains six IDT-aligned transfer rows: LBM-F/H/K and SaMST e5/e15.

## Reviewer Assignments

| reviewer | agent | lens | status |
|---|---|---|---|
| A | Ramanujan | method / theory / novelty | complete |
| B | Ptolemy | experiments / fairness / statistics | complete |
| C | Ampere | writing / structure / layout | complete |
| D | Euler | hostile baseline reviewer | complete |

## Results

### Reviewer A / Ramanujan

Score: `5/10`.

Main critique:

- The manuscript still overstated OT as an active headline mechanism.
- The reported Distinct5 F/H/K configurations are OMF-family runs with
  `w_flow=0.0`; the local flow / OT endpoint residual branch in `losses.py` is
  therefore inactive for those headline rows.
- The Distinct5 rows use target-domain endpoint selection through a
  prototype-aware pairing cache / queue schedule, not an active minibatch
  Sinkhorn objective at every training step.
- The inference path was described incorrectly in one place: target-domain
  latents and endpoint selection are training-side supervision, not
  inference-time inputs.
- SA-SWD should be framed as the selected terminal distribution-matching
  implementation, not as independently proven superiority over random axes.

Immediate safe fixes:

- Remove `OT-coupled` from the title and headline method claim.
- Rewrite the abstract and method overview around IDT-calibrated latent
  transport, training-side endpoint selection, vector-field execution, terminal
  SWD, and kinetic regularization.
- State explicitly that zero-weight terms are inactive and that Distinct5
  headline rows should be read as pairing-cache / terminal-SWD / kinetic OMF
  objectives.
- Keep Sinkhorn/OT as a design-family endpoint-assignment route and
  design-grounding check, not as a claimed active force for every reported row.

### Reviewer B / Ptolemy

Score: `4/10`, weak reject.

Main critique:

- Distinct5 selection is still the largest fairness risk because the original
  full ranked CLIP-prototype list is not retained.
- Baseline convergence remains incomplete for SaMAM and only partially supported
  for SaMST.
- The paired bootstrap is valid only for method-minus-IDT transfer CLIP-S on
  retained aligned rows; it is not a model-vs-model test and does not cover
  SaMAM, LPIPS, ArtFID, or selection-after-sweep uncertainty.
- Table 4 should make ArtFID scope clear and avoid mixing full-scope metrics
  with transfer-only deltas without explanation.
- Timing should be framed as selected-checkpoint wall time, not time-to-parity.

Immediate safe fixes:

- Add explicit bootstrap boundary language in the Distinct5 paragraph.
- Make Table 4 caption/headers clarify full-scope and transfer-only quantities.
- Replace abstract/contribution language that implies normalized efficiency
  with selected-checkpoint wall-time language.

### Reviewer C / Ampere

Score: `6/10`.

Main critique:

- The IDT/Distinct5 story is coherent, but the draft still sounds too much like
  an internal rebuttal memo.
- Distinct5 is the primary matched evidence but appears after the historical
  strict-750 surface.
- Abstract is overloaded and should state the spine more directly.
- Figure 1 caption undersells the result.
- Method prose should describe the clean evaluated algorithm instead of
  foregrounding implementation caveats.
- Table 4 remains cramped; if kept single-column, it needs clearer labels.

Immediate safe fixes:

- Rewrite the abstract toward a cleaner IDT-calibrated story.
- Strengthen Figure 1 caption.
- Replace defensive method framing with a clean four-component LBM description.
- Keep caveats in Experiments/Limitations instead of Method.

### Reviewer D / Euler

Score: `4/10`, reject-leaning.

Main critique:

- The paired bootstrap materially strengthens the narrow Distinct5-vs-IDT
  claim, but does not answer the hostile baseline objection: one CLIP-selected
  split, selected operating points, incomplete SaMAM convergence, and no
  normalized time-to-parity.
- Several claims still sound broader than the evidence: mechanism controls,
  tokenizer capacity, target-queue mechanism, and selected-checkpoint training
  cost.

Immediate safe fixes:

- Soften abstract/contribution wording around training cost and mechanism
  controls.
- Make tokenizer conclusions explicitly variant-limited.
- Avoid causal language such as "reduces target-distribution noise" unless
  clearly framed as interpretation.

## Integration Plan

The intended integration rule is:

- Fix writing/layout issues immediately when they are evidence-preserving.
- Weaken or qualify claims when reviewers identify a real evidence mismatch.
- Do not add new positive claims without landed artifacts.
- Do not integrate SaMAM convergence or significance until a complete paired
  per-image packet is available.

Current integrated action set:

- Rewrite the abstract and Figure 1 caption.
- Clarify Table 4 scope and column labels.
- Add a sentence that bootstrap intervals are method-minus-IDT transfer CLIP-S
  only and are not model-vs-model tests.
- Soften selected-checkpoint cost and tokenizer/mechanism language.
- Align the method/objective claim with the active Distinct5 configs:
  `w_flow=0.0`, no active flow endpoint residual, pairing-cache endpoint
  selection, terminal SWD, and kinetic control.
- Preserve the SaMAM failure definition as target-style movement failure:
  reproduced SaMAM checkpoints visibly edit images but remain below the
  transfer-only IDT CLIP-S floor.
