# AAAI 2027 Review Packet Template

Use this packet when sending a new cycle to an independent reviewer subagent.

## Header

- `review_cycle_id`:
- `lane`:
- `checkpoint_label`:
- `scope`:

## Required inputs

- manuscript:
  - `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- working index:
  - `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- experiment log:
  - `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- previous consensus:
  - `SchrodingerBridge/docs/reviews/...`
- focused evidence bundle:
  - `SchrodingerBridge/docs/experiments/...`

## Required outputs

1. `overall_status`
2. `claim_safety_band`
3. `evidence_closure_band`
4. `blocking_issue`
5. `next_action_1`
6. `next_action_2`
7. `support_score` (`0/1/2`)
8. `fairness_score` (`0/1/2`)
9. `artifact_path_score` (`0/1/2`)
10. `closure_value_score` (`0/1/2`)

For `scorecard`, also return:

1. `novelty`
2. `technical_quality`
3. `experimental_rigor`
4. `clarity`
5. `reproducibility`
6. `significance`

## Lane-specific prompts

### Adversarial review

- What would a strong AAAI reviewer reject first?
- Which claim must be narrowed immediately?
- Is there any fairness, provenance, or metric-hacking risk that blocks
  escalation?

### Scorecard

- Score the six AAAI criteria on a `1-10` scale.
- Which one score is the bottleneck to acceptance?
- What one action would move the mean score most?

### Experiment audit

- Which current claims are directly supported, indirectly supported, or still
  unsupported?
- What is the minimum next experiment that closes the largest evidence gap?
- What must block claim escalation right now?
