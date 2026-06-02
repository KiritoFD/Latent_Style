# Review Lane

This directory is the persistent reviewer lane for the AAAI 2027 submission.

The rule is simple: every major paper or experiment checkpoint should be judged
from at least three independent perspectives:

1. `adversarial_review`
   - asks: "why would a strong reviewer reject this paper?"
2. `scorecard`
   - asks: "how strong is the paper on each AAAI criterion?"
3. `experiment_audit`
   - asks: "which claims are actually supported by current evidence?"

## Current file naming

Use date-stamped files so reviews can accumulate over time:

- `aaai2027_adversarial_review_YYYYMMDD.md`
- `aaai2027_scorecard_YYYYMMDD.md`
- `aaai2027_experiment_audit_YYYYMMDD.md`

## What belongs here

- harsh reviewer memos
- acceptance-risk scorecards
- experiment support audits
- follow-up review deltas after paper rewrites or new baselines

## What does not belong here

- raw experiment logs
- generated figures
- training configs
- code patches

Those belong in `docs/experiments/`, `aaai_submission/figures/`, or the main
source tree.

## Usage pattern

1. Update the paper or finish a meaningful experiment block.
2. Re-run independent reviewer agents on the new state.
3. Save each memo here with a new date-stamped filename.
4. Fold only evidence-backed findings into the paper or experiment plan.

This keeps the review process continuous instead of one-shot.

When two or more independent reviews exist for the same checkpoint, add a short
consensus note such as:

- `aaai2027_review_consensus_YYYYMMDD.md`

## Operating files

Protocol:

- `aaai2027_review_protocol.md`

Round registry:

- `aaai2027_review_registry.csv`

Use the protocol to decide when to re-run reviewers and the registry to track
whether the paper is actually converging toward submission safety.

Standing claim-boundary memo:

- `aaai2027_claim_safety_memo_20260603.md`
