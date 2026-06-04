# Dalton AAAI 2027 Experiment Backlog

Updated: 2026-06-04

Owner: Dalton remote sidecar.

Rule: do not edit the main paper from the sidecar. Deliver closed or explicitly
partial experiment packets only; the main thread decides what enters the draft.

## Current Paper State To Preserve

The paper uses Distinct5/IDT as the primary evidence spine.

Current paper-safe claims:

- SaMAM is valid only through step 2250 in the current manuscript.
- SaMAM 2250 remains below IDT in transfer CLIP-S while lowering targetwise
  ArtFID.
- Post-2250 SaMAM outputs are excluded from manuscript evidence as
  reproduction-chain failures unless a clean independent rerun closes a new
  aligned packet.
- LBM-F/K clear IDT with low checkpoint-training wall time in the reproduced
  setup.
- SaMST clears IDT but in a high-displacement, high-targetwise-ArtFID region.
- LPIPS records displacement; it is not the sole rejection criterion for any
  baseline.

Current SaMAM row:

| point | transfer CLIP-S | transfer LPIPS | targetwise ArtFID | delta-IDT | train min |
| --- | ---: | ---: | ---: | ---: | ---: |
| SaMAM 2250 | 0.5523 | 0.3605 | 148.2 | -0.0877 | 458.6 |

Authoritative boundary:
`SchrodingerBridge/docs/experiments/2026-06-04-samam-distinct5-valid-through-2250.md`.

## Current Sidecar Audit Status

- A packet is not closed until it contains full/transfer CLIP-S and LPIPS,
  targetwise ArtFID, same-scope timing, and IDT-aligned per-image rows or an
  explicit missing-row report.
- Existing curve CSVs may still contain post-2250 SaMAM rows for audit history.
  Do not treat those rows as manuscript evidence.
- The main bottleneck is statistics/evidence closure, not another local rewrite.

## Immediate Dalton Deliverable

Create or update one packet directory that answers only this:

1. Which SaMAM Distinct5 checkpoints are visible now through step 2250?
2. Which fields are closed for the 2250 packet: full/transfer CLIP-S, LPIPS,
   targetwise ArtFID, training wall time, generation/inference time, and
   IDT-aligned rows?
3. What exact fields are missing, and are they recoverable without retraining?
4. If any post-2250 SaMAM artifact is considered for future use, can it be
   independently regenerated from a clean run? If not, it remains excluded.
5. What SaMST e5/e15 timing fields are still missing, and can they be measured
   from existing generated outputs or logs?
6. Can the paper upgrade any wording? Valid answers are:
   - `no upgrade`;
   - `SaMST timing closed only`;
   - `SaMAM 2250 packet closed only`;
   - `clean SaMAM rerun supersedes 2250`;
   - `additional split packet closed`.

Do not start a new model-training run just to improve the paper score until the
existing packet status is resolved. If the remote GPU is idle and the packet is
blocked only by a short eval/timing job, running that eval is allowed.

## Priority 0: SaMAM Distinct5 Boundary or Clean Rerun

Goal: either strengthen the current 2250 boundary with complete packet evidence
or produce a clean independent rerun that supersedes it.

Required outputs for the current 2250 packet:

- Full-scope CLIP-S and LPIPS.
- Transfer-only CLIP-S and LPIPS.
- Targetwise ArtFID.
- Train wall time.
- Inference ms/img if generation is rerun.
- IDT-aligned per-image metrics packet if available.
- A compact `summary.json`, `metrics.csv`, and `README.md` under the run root.

If running a clean independent SaMAM rerun:

- Use the same Distinct5 train/test split and 30 x 5 x 5 evaluation protocol.
- Record exact command, environment, seed, checkpoint schedule, and stop reason.
- Evaluate every retained checkpoint with full_eval plus targetwise ArtFID.
- Report whether any post-2250 transition reproduces under the clean run.

Paper gate:

- If the clean rerun remains below IDT, the paper can strengthen SaMAM as a
  converged target-execution failure on this split.
- If the clean rerun clears IDT, report the new point honestly and compare its
  LPIPS / targetwise ArtFID / wall time against LBM and SaMST.
- Do not use aggregate ArtFID alone to overturn the CLIP-S-vs-IDT criterion.

## Priority 0.5: Close SaMST Distinct5 Evidence Packet

Goal: make the SaMST comparison as auditable as the LBM/IDT packet instead of
using only the e5/e15 CLIP-S/LPIPS closeness statement.

Required outputs:

- SaMST e5 and e15 full-scope summaries.
- SaMST e5 and e15 transfer-only summaries.
- Targetwise ArtFID for e5 and e15.
- IDT-aligned per-image rows where available.
- Paired bootstrap for method-minus-IDT transfer CLIP-S if aligned rows exist.
- Train wall time and inference ms/img for both checkpoints.

Paper gate:

- If e5/e15 differ by less than `0.004` CLIP-S and `0.002` LPIPS but ArtFID
  changes materially, report ArtFID separately rather than claiming full
  convergence.
- If ArtFID and transfer metrics are also stable, the paper can keep e15 as a
  conservative endpoint and cite e5 as convergence evidence.

## Priority 1: Additional Fixed-Rule WikiArt Stress Splits

Goal: reduce the "custom Distinct5" reviewer attack.

Source selector:

- `SchrodingerBridge/docs/experiments/wikiart_stress_splits_20260603/selected_splits.json`

Run at least two splits first:

1. `Color_Field_Painting`, `High_Renaissance`,
   `Mannerism_Late_Renaissance`, `Pop_Art`, `Realism`
2. `Abstract_Expressionism`, `Baroque`, `Cubism`,
   `Northern_Renaissance`, `Post_Impressionism`

For each split:

- Materialize train/test with the same rule: 1000 train and 30 held-out test
  images per class.
- Build IDT outputs first.
- Evaluate IDT with the same 750-output evaluator.
- Train/evaluate LBM-F and LBM-K style operating points if budget permits.
- Evaluate SaMAM and SaMST only after the current Distinct5 baseline packets
  are closed or explicitly declared unrecoverable.

Required metrics:

- full and transfer-only CLIP-S
- full and transfer-only LPIPS
- targetwise ArtFID
- train wall time
- inference ms/img
- paired method-minus-IDT transfer CLIP-S bootstrap where aligned per-image rows
  are retained

Paper gate:

- Keep a split only if all methods use the same 5x5 strict-750 test set and IDT
  baseline.
- If LBM clears IDT on at least two additional fixed-rule splits, the paper can
  strengthen Distinct5 from "one stress case" to "fixed-rule stress-family
  evidence".
- If results are mixed, write them as boundary evidence, not a failed run.

## Priority 1.25: Reviewer-Requested Statistical Closure

Goal: close the current AAAI reviewer risks without changing the paper's core
claim. These are evidence packets, not new model-design work.

For Distinct5-512 and any additional fixed-rule stress split, produce:

- clustered bootstrap by source image and transfer direction;
- paired intervals for CLIP-S, LPIPS, targetwise ArtFID, and method-minus-IDT;
- where aligned packets exist, direct LBM-F/LBM-K vs SaMST comparisons;
- explicit aggregation rule for targetwise ArtFID;
- row count and missing-row report for every method.

For SaMAM/SaMST:

- include full and transfer-only summaries;
- include train curve points, even if sparse;
- mark whether each point is `measured_checkpoint`, `final_checkpoint`, or
  `tuned_checkpoint`;
- do not treat post-2250 SaMAM as manuscript evidence without a clean rerun.

Paper gate:

- If clustered intervals preserve the current signs, upgrade bootstrap wording.
- If clustered intervals weaken signs, keep only point estimates and write the
  limitation explicitly.

## Priority 1.3: Aligned Qualitative Evidence Packet

Goal: make the visual evidence match the IDT-calibrated claim instead of showing
only dense 5x5 grids.

For Distinct5-512 first, then any additional fixed-rule stress split that enters
the paper, produce a compact aligned image packet:

- 3-4 representative off-diagonal transfers.
- For each row, include:
  - source image / IDT output;
  - target style label and one held-out target reference image;
  - LBM-F output;
  - LBM-K output if visually distinct from F;
  - SaMST e15 output;
  - SaMAM 2250 output if available.
- Use exactly the same source image and target direction for every method.
- Include the row-level CLIP-S, LPIPS, and targetwise ArtFID contribution where
  available.
- Export both:
  - a paper-ready single-column panel;
  - a full-resolution contact sheet for audit.

Selection rule:

- Do not cherry-pick by visual taste alone.
- Prefer rows near each method's median transfer CLIP-S and LPIPS, plus one row
  where SaMST's high-displacement failure is visually clear.
- Record the selection script, random seed if any, and source metrics file.

Paper gate:

- If aligned outputs are missing for any method, label the missing column rather
  than fabricating a comparison.
- The panel should support the existing claim: IDT is the unchanged control,
  LBM occupies low-displacement positive-IDT movement, SaMST exposes
  high-damage target movement, and SaMAM 2250 exposes low-ArtFID non-execution.

## Priority 1.5: Matched Terminal-Axis Ablation

Goal: decide whether semantic-axis terminal projection can be claimed as more
than an implementation choice.

Run under a matched Distinct5 config/seed, preferably H or F family:

- semantic-axis terminal projection discrepancy
- random-axis terminal projection discrepancy with matched number of projections
- same endpoint cache / queue schedule
- same endpoint-velocity penalty weight
- same batch and epoch budget

Required metrics:

- full and transfer-only CLIP-S
- full and transfer-only LPIPS
- targetwise ArtFID
- endpoint/path/peak L2 if path probe is available
- train wall time and inference ms/img

Paper gate:

- If semantic-axis improves transfer CLIP-S or ArtFID without LPIPS damage, it
  can become a scoped mechanism claim.
- If random-axis matches it, keep semantic-axis projection as an implementation
  detail and do not write it as a standalone contribution.

## Priority 1.55: Endpoint-Trained vs Path-Trained Field Check

Goal: close the theory-reviewer risk that the selected Distinct5 rows train at
endpoint time while inference uses intermediate recurrence controls.

Run a matched Distinct5 H or F family check:

- current endpoint-trained field with retained fixed-step recurrence inference;
- endpoint-trained field forced to use endpoint-time query for every recurrence
  substep, if this can be implemented without changing the renderer;
- random-time or multi-time supervised variant with the same endpoint queue and
  terminal projection discrepancy, if budget permits.

Required metrics:

- full and transfer-only CLIP-S;
- full and transfer-only LPIPS;
- targetwise ArtFID;
- 1/4/8/16/256-step endpoint stability probe;
- train wall time and inference ms/img.

Paper gate:

- If intermediate-time queries match endpoint-time quality and remain stable,
  the paper can keep the current empirical recurrence-stability wording.
- If random-time supervision improves quality or stability materially, update
  the method section and consider making path training the new default.
- If intermediate queries hurt quality, remove path-style language and report
  the model as an endpoint residual renderer.

## Priority 1.56: Active Endpoint Metric Ablation

Goal: make the latent-metric discussion evidence-backed instead of only
theoretical. The headline rows have `w_flow=0`, so endpoint residual metric
claims must be tested in a branch where the residual is active.

Run a small matched packet with identical endpoint queues and terminal settings:

- `w_flow>0` with MSE residual;
- `w_flow>0` with Huber residual;
- `w_flow>0` with L1 residual;
- optional terminal-only control matching the headline objective.

Required metrics:

- full and transfer-only CLIP-S;
- full and transfer-only LPIPS;
- targetwise ArtFID;
- train wall time;
- endpoint/path/peak L2.

Paper gate:

- If Huber/L1 materially improves the active-residual branch, the method text
  can explain why endpoint pointwise metrics are fragile in latent space.
- If all residual branches underperform terminal-only, keep pointwise residual
  as inactive implementation support and avoid overclaiming.
