# Continuous Independent Review Upgrade

Date: 2026-06-03

This note records the change from a loose reviewer lane to a persistent,
inspectable subagent review loop.

## What changed

1. review rounds are now treated as explicit `review cycles`
2. each cycle must use three distinct independent subagents:
   - `adversarial_review`
   - `scorecard`
   - `experiment_audit`
3. each lane now writes structured fields:
   - `overall_status`
   - `claim_safety_band`
   - `evidence_closure_band`
   - `blocking_issue`
   - `next_action_1`
   - `next_action_2`
4. per-lane verdicts and scores are now tracked in:
   - `docs/reviews/aaai2027_review_score_log.csv`
5. the experiment master log now tracks:
   - `review_required`
   - `latest_review_cycle`
   - `review_status`
   - `claim_safety_band`
6. routine cycles are now compact:
   - CSV append for normal checkpoints
   - markdown memos only when the paper-safe boundary moves

## Why this matters

The paper risk was not only missing experiments. It was also missing a durable
control loop that prevents favorable metrics or isolated visuals from being
promoted before an independent reviewer lane judges them.

This upgrade makes three things auditable:

1. which agent reviewed which checkpoint
2. what score or claim-safety band it assigned
3. whether repeated blockers are actually being closed over time

## Immediate effect on the current paper state

The current paper remains below submission-safe. The new control loop is meant
to prevent drift while the next closure experiments land:

1. `flow_loss_metric_ablation`
2. `sa_swd_semantic_vs_random`
3. `normalized_time_to_parity`
