# AAAI 2027 Next Review Gate

Date: 2026-06-04

This file defines when the next four-reviewer adversarial pass should happen.
It prevents repeated review loops before the draft has new evidence or a
meaningful new writing state.

## Current State

- Current paper: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Latest full reviewer pass:
  `SchrodingerBridge/docs/reviews/aaai2027_writing_iteration_R20260604_round16.md`
- Latest local non-review cleanup:
  `SchrodingerBridge/docs/reviews/aaai2027_writing_iteration_R20260604_round18_nonreview.md`

The draft is positioned as an evaluation-contract paper:

- IDT is the slotwise CLIP-S falsification contract.
- Distinct5-512 is one fixed-rule CLIP-separated WikiArt stress split, not a
  broad robustness benchmark.
- LBM is an endpoint-supervised Style-ID latent renderer, not an OT-map,
  Schrodinger-bridge estimator, or learned continuous-time stochastic path.
- Multi-step fixed-step recurrence is an endpoint-refinement heuristic.
- Table 1 reports observed operating points, not selection-corrected method
  estimates or hardware-normalized time-to-parity.
- Headline Distinct5 claims use transfer-only metrics; audit-full columns are
  for scope checking.

## Current SaMAM Boundary

The active manuscript treats SaMAM as valid only through step 2250.

Current paper row:

| point | transfer CLIP-S | transfer LPIPS | targetwise ArtFID | delta-IDT | train min |
| --- | ---: | ---: | ---: | ---: | ---: |
| SaMAM 2250 | 0.5523 | 0.3605 | 148.2 | -0.0877 | 458.6 |

Outputs after 2250 are excluded from manuscript evidence as reproduction-chain
failures unless a clean independent rerun closes a new aligned packet. The
paper should not describe post-2250 SaMAM outputs as positive-IDT evidence.

Authoritative boundary:
`SchrodingerBridge/docs/experiments/2026-06-04-samam-distinct5-valid-through-2250.md`.

## Trigger a New Four-Reviewer Pass Only After One of These

1. A clean independent SaMAM Distinct5 rerun supersedes the 2250 boundary:
   - exact checkpoint status and stop reason;
   - full and transfer-only CLIP-S/LPIPS;
   - targetwise ArtFID;
   - same-scope timing, or an explicit statement that pure generation timing is
     still missing;
   - retained IDT-aligned per-image packet or explicit missing-row report.

2. At least one additional fixed-rule WikiArt stress split has a closed packet:
   - IDT outputs and summary;
   - LBM-F/K full and transfer summaries;
   - targetwise ArtFID;
   - train/eval/inference timing;
   - paired or clustered uncertainty packet where aligned rows exist.

3. The main paper receives a substantial structural rewrite:
   - new abstract/contribution framing;
   - new main table;
   - replacement of Figure 1 or Figure 3;
   - upgraded SaMAM/SaMST wording from partial evidence to a closed baseline.

4. A mechanism claim is upgraded:
   - semantic-axis terminal projection beats random-axis under matched settings;
   - endpoint-trained vs path-trained field check changes the method section;
   - tokenizer/executor factorization produces a causal representation result.

## Reviewer Roles for the Next Pass

- Area-chair / writing reviewer: judge story coherence and claim scope.
- Statistics reviewer: judge IDT, row/cluster intervals, ArtFID, and cost
  accounting.
- Method/math reviewer: judge endpoint sampler, terminal projection,
  endpoint-velocity penalty, and representation claims.
- Figures/layout reviewer: judge page 1, Table 1, Figure 2, Figure 3, and
  whether visuals support the aggressive claims.

## Do Not Trigger a New Reviewer Pass For

- purely local spelling edits;
- PDF rebuilds without content changes;
- isolated wording changes that do not alter claims;
- remote runs that have not completed full_eval plus targetwise ArtFID;
- partial SaMAM/SaMST logs without aligned metrics;
- reviewer feedback that repeats earlier issues without new evidence or new
  figures.

## Immediate Handoff

Dalton should use
`SchrodingerBridge/docs/experiments/dalton_aaai2027_experiment_backlog_20260604.md`.
The main thread should not run another four-reviewer adversarial pass until one
of the triggers above occurs.

Until then, the active paper should keep SaMAM at `2250`, keep IDT as the
primary reporting contract, and avoid reviewer loops for local wording cleanup.
