# AAAI 2027 Adversarial Review - 2026-06-03 R4

Reviewer lane: `standing_adversarial_reviewer`  
Scope: manuscript/evidence audit after paper-safe tightening and config-audit update  
Inputs inspected:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/reviews/aaai2027_review_consensus_20260603.md`
- `SchrodingerBridge/docs/reviews/aaai2027_review_registry.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_review_score_log.csv`
- `SchrodingerBridge/docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`

## Summary fields

- `overall_status`: `weak_reject`
- `claim_safety_band`: `narrow_only`
- `evidence_closure_band`: `open`
- `blocking_issue`: `The new wording is materially safer, but the paper still lacks matched activated closure on the endpoint-metric story, still lacks semantic-vs-random SA-SWD isolation, and still presents operating-point timing without the normalized time-to-parity figure that would close the fairness attack surface.`
- `next_action_1`: `Run the repaired endpoint-metric ablation on an actually active transport term (either OMF with w_flow > 0 or a non-OMF velocity-regression path) and treat the current mse/huber/l1 trio only as an operational near-null control.`
- `next_action_2`: `Run a fixed-base Distinct5 semantic-vs-random SA-SWD axis ablation before claiming semantic alignment as more than design decoration; if time allows, pair it immediately with the normalized time-to-parity figure.`
- `new_paper_wording_materially_safer_than_before`: `yes`

## Why the wording is safer

- The manuscript now explicitly says the active Distinct5 tokenizer family resolves to `objective_mode=omf`, which closes the previous paper/code mismatch.
- The manuscript now narrows the metric claim to endpoint-side OT plus `W1`-style terminal matching, instead of implying that all latent-space `MSE/L2` is broadly discredited.
- The efficiency language is softer than before and now reads as an operating-point observation rather than a universal speed theorem.

## Why this is still not AAAI-safe

1. `Endpoint-metric closure is still missing.`
   The experiment log and flow-loss README both say the completed `mse/huber/l1` block was invalidated by config audit (`objective_mode=omf`, `w_flow=0.0`). The paper is safer for admitting this boundary, but the headline metric story is still not closed by an activated ablation.

2. `SA-SWD novelty is still under-isolated.`
   The paper still attributes value to semantic projection-axis selection, but the required semantic-vs-random axis ablation remains planned, not completed.

3. `Efficiency fairness is still vulnerable.`
   The current paper language is better, but the evidence surface still depends on operating-point wall-clock comparisons across methods with different stopping rules. The review consensus is still correct: without a normalized time-to-parity figure, this remains attackable.

4. `Theorem-support rhetoric is safer but not fully closed.`
   The wording is more careful than before, yet the paper still says the formal results are paired with direct empirical validation while the experiment log still leaves path-level closure as planned rather than closed.

## Bottom line

The paper is meaningfully safer than the previous version and no longer commits the most obvious paper/code mismatch on the flow-loss story. That improvement is real. It is still not submission-safe because the current strongest claims are now mostly phrased correctly but remain experimentally under-closed in exactly the places the review consensus already identified.
