# AAAI 2027 Adversarial Review Round R20260604Q

Date: 2026-06-04

Scope:

- Manuscript: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Gates consulted: `aaai2027_writing_gate_R20260603O.md`, `aaai2027_adversarial_review_R20260604P.md`
- Config disclosure: `docs/experiments/distinct5_512_20260602/resolved_headline_config.md`

This round used four independent adversarial reviewer roles. They did not edit
files. The main manuscript was subsequently patched only for evidence-preserving
wording and table-label fixes.

## Fixed Correction

The SaMAM Distinct5 failure is not a high-LPIPS claim.

The supported statement is:

- reproduced SaMAM checkpoints visibly edit images;
- their measured transfer-only CLIP-S remains below the IDT floor;
- therefore the edits are not stably target-style-directed under the
  IDT-calibrated CLIP-S criterion;
- LPIPS records displacement and is not the rejection criterion for SaMAM.

This correction is now reflected in the manuscript, the Distinct5 experiment
index, and the experiment audit note.

## Reviewer Votes

| reviewer lens | score | decision tendency |
|---|---:|---|
| Method / active objective consistency | 5/10 | weak reject |
| Experiments / fairness / statistics | 4/10 | weak reject |
| Writing / layout / AAAI narrative | 6/10 | borderline |
| Hostile baseline / reproducibility | 4/10 | reject-leaning |

Current aggregate expectation: borderline-to-weak-reject unless the writing is
tightened further and the remaining evidence gaps are either filled or explicitly
scoped down.

## Main Risks

1. Active objective mismatch remains the highest method risk. The paper must not
   imply that the Distinct5 F/H/K headline rows optimize an active online
   minibatch Sinkhorn / flow-residual objective. The resolved headline configs
   use `objective_mode=omf`, `w_flow=0.0`, `terminal_swd_weight=20.0`, and
   `w_kinetic=1.0`. The safe wording is pairing-cache / terminal-SWD / kinetic
   OMF.

2. Distinct5 selection is still attackable. The split is a valid WikiArt stress
   case, but the complete original CLIP-prototype ranked list is not retained.
   The paper should describe a materialized CLIP-separated stress split plus a
   fixed-rule selector for follow-up splits, not a universal benchmark.

3. Baseline convergence is incomplete. SaMAM rows remain measured checkpoints,
   not a convergence endpoint or paired-bootstrap packet. SaMST has e5/e15
   evidence but not a full budget-controlled frontier.

4. Bootstrap support is narrow. Current intervals validate method-minus-IDT
   transfer CLIP-S for retained aligned rows. They do not prove model-vs-model
   superiority, LPIPS/ArtFID significance, or selection-after-sweep uncertainty.

5. Timing must stay as operating-point footprint. The 1.2-minute LBM Distinct5
   number excludes evaluation and search. It can be reported because it is
   measured, but not as normalized time-to-parity.

6. ArtFID must be treated as diagnostic. Targetwise ArtFID can support the
   displacement/artifact story, but cannot override the IDT style-movement test
   when CLIP-S remains below IDT.

## Safe Claims After This Round

- Distinct5-512 is a CLIP-separated WikiArt stress case, not a universal art
  benchmark.
- IDT is a strong and necessary baseline on separated art-to-art transfer.
- LBM-F and LBM-K have positive transfer-only CLIP-S deltas over IDT with retained
  paired-bootstrap support.
- SaMAM's current Distinct5 issue is target-style direction failure
  (`CLIP-S < IDT`), despite visible edits and nonzero displacement.
- SaMST clears IDT, but in the measured operating point it does so with much
  larger LPIPS and high targetwise ArtFID.
- The Distinct5 LBM headline rows should be described as resolved
  pairing-cache / terminal-SWD / kinetic OMF runs.

## Edits Applied Immediately

- Replaced abstract/contribution/discussion phrasing of `selected-checkpoint`
  with checkpoint-training wall time in the reproduced setup.
- Softened the Distinct5 split construction wording from a retained fixed ranked
  screen to a materialized CLIP-separated WikiArt stress split plus fixed-rule
  follow-up selector.
- Renamed the Table 4 ArtFID header to `tw-ArtFID` and made IDT capitalization
  consistent.
- Updated experiment index and audit language to state that SaMAM is rejected
  by `CLIP-S < IDT`, not by LPIPS.
- Reordered the Experiments section so the primary Distinct5/IDT evaluation
  appears before the historical strict-750 contextual surface.
- Rephrased historical training cost as checkpoint wall time / checkpoint
  footprints rather than stronger time-to-parity language.

## Necessary Next Experiments

Highest priority:

1. Complete SaMAM final packet on Distinct5: transfer-only CLIP-S, LPIPS,
   targetwise ArtFID, aligned per-image metrics, and paired bootstrap where
   possible.
2. Evaluate at least two additional fixed-rule WikiArt stress splits with IDT,
   LBM, SaMAM, and SaMST under the same transfer-only protocol.
3. Add clustered or source-pair-aware bootstrap if time permits.

Useful but secondary:

4. Add a non-CLIP target-style check, such as a held-out style classifier,
   VLM rubric, or small human preference audit.
5. If SA-SWD is kept as more than an implementation choice, rerun
   semantic-vs-random projection axes with artifact diagnostics.
6. If OT endpoint construction is to remain a headline method claim, either run
   a true active Sinkhorn / flow-residual mainline or keep the current
   pairing-cache wording throughout.

## Writing Priorities

1. Make Distinct5/IDT the spine of the experiments.
2. Keep the historical strict-750 table as contextual support only.
3. Replace defensive caveats with compact, explicit scope statements.
4. Avoid claims that require missing convergence packets.
5. Keep the aggressive message focused: raw style metrics can be worse than
   useless without the unchanged-image floor.
