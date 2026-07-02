# AAAI 2027 Continuous Review Protocol

Updated: 2026-06-03

This protocol makes the reviewer lane a standing control loop instead of a
one-off paper critique. The unit of operation is a `review cycle` tied to a
meaningful experiment or writing checkpoint, not an ad hoc memo.

## 1. Trigger conditions

Run a new review cycle whenever at least one of the following is true:

1. the manuscript changes in a way that affects claims, framing, or tables;
2. a new paper-facing experiment family lands;
3. a baseline comparison changes;
4. an evaluation contract changes;
5. a new theorem, tokenizer claim, or efficiency claim is introduced;
6. a formal remote experiment block finishes or changes phase from
   `planned/queued/running` to `completed`.

Do not wait for a "finished" draft. The point is to catch drift early.

## 2. Standing independent lanes

Every cycle must contain three independent subagent reviews:

1. `adversarial_review`
   - asks why a strong AAAI reviewer would reject the paper now
   - outputs one verdict, strongest rejection reasons, and the claim that must
     be narrowed immediately

2. `scorecard`
   - scores the paper on:
     - novelty
     - technical_quality
     - experimental_rigor
     - clarity
     - reproducibility
     - significance
   - each score must include concrete reasons

3. `experiment_audit`
   - maps paper claims into:
     - directly supported
     - indirectly supported
     - unsupported / should not be written yet
   - outputs the next minimum experiment protocol needed to close the largest
     evidence gap

One agent must not play multiple lanes in the same cycle. The active agent
nickname or id for each lane must be recorded in the score log so the
independence is inspectable after the fact.

## 3. Review-cycle identity

Every cycle gets one stable id:

- `RYYYYMMDD[A-Z]`

Example:

- `R20260603A`

One cycle produces:

1. three lane memos;
2. one consensus note;
3. three lane rows in `aaai2027_review_score_log.csv`;
4. one summary row in `aaai2027_review_registry.csv`.

## 4. Minimum inputs per cycle

Every reviewer cycle should read the same minimum package:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/README.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- current comparison or benchmark evidence directories referenced by the paper
- previous consensus note, if one exists

If a cycle is about a specific claim, add the narrow evidence bundle for that
claim instead of broadening the read set arbitrarily.

## 5. Required lane outputs

Every lane must emit the following structured fields, even if some are blank:

1. `overall_status`
   - one of:
     - `reject`
     - `weak_reject`
     - `borderline`
     - `weak_accept`
     - `accept`

2. `claim_safety_band`
   - one of:
     - `unsafe`
     - `narrow_only`
     - `paper_safe`

3. `evidence_closure_band`
   - one of:
     - `open`
     - `partial`
     - `closed`

4. `blocking_issue`
   - the single highest-priority reason the paper should not escalate claims

5. `next_action_1`
6. `next_action_2`

Every lane must also emit four compact cycle scores for the target claim:

1. `support_score`
   - `0` unsupported
   - `1` bounded / indirect support
   - `2` directly supported under a matched protocol

2. `fairness_score`
   - `0` comparison attackable
   - `1` bounded but still qualified
   - `2` normalized and reviewer-safe

3. `artifact_path_score`
   - `0` broken or mixed provenance
   - `1` partial chain
   - `2` auditable chain from config and checkpoint to summary and paper table

4. `closure_value_score`
   - `0` closes no blocker
   - `1` narrows one blocker
   - `2` closes one blocking claim from the current consensus set

The `scorecard` lane must additionally emit the six criterion scores listed in
Section 2, each on a `1-10` scale.

## 6. Output contract

Default cycle writeback is compact:

1. three lane rows in `aaai2027_review_score_log.csv`
2. one summary row in `aaai2027_review_registry.csv`

Full markdown memos are required only when at least one of the following
changes:

1. the consensus status changes
2. the safe / unsafe claim boundary changes
3. blocker ordering changes
4. one lane materially disagrees with the others

Memo-mode files use:

1. `aaai2027_adversarial_review_YYYYMMDD.md`
2. `aaai2027_scorecard_YYYYMMDD.md`
3. `aaai2027_experiment_audit_YYYYMMDD.md`
4. `aaai2027_review_consensus_YYYYMMDD.md`

If multiple memo-mode cycles happen on the same date, append `_r2`, `_r3`, and
so on.

The consensus note must include:

- the review-cycle id
- the checkpoint label under review
- the source reviewer files
- the lane agents used
- shared strengths
- shared blockers
- claims that must be narrowed now
- ordered next experiments
- current submission status:
  - `reject`
  - `weak_reject`
  - `borderline`
  - `safe_to_submit`

## 7. Promotion gates

A claim may move into the abstract, contributions list, or main comparison
table only if all of the following are true:

1. `experiment_audit` marks it as directly supported;
2. `adversarial_review` does not flag it as a fairness or overclaim blocker;
3. `claim_safety_band` is not `unsafe`;
4. the evidence path is explicit in `docs/experiments/` or the paper figure
   directory.

The paper is not "AAAI-safe" until:

1. no reviewer lane calls out an unsupported headline claim;
2. the experiment audit no longer requests a blocking ablation;
3. the scorecard average is at least `7.0/10`;
4. at least two of the three lanes are no worse than `weak_accept` or
   equivalent;
5. the latest review cycle has `claim_safety_band != unsafe`.

## 8. Registry and score-log discipline

Every cycle must be recorded in:

- `SchrodingerBridge/docs/reviews/aaai2027_review_registry.csv`

Every independent lane evaluation must be recorded in:

- `SchrodingerBridge/docs/reviews/aaai2027_review_score_log.csv`

The registry is the control plane for:

- current paper risk
- claim status
- next experiment priority
- whether the paper is converging or drifting

The score log is the per-agent audit trail for:

- which agent reviewed which checkpoint
- what band and verdict it assigned
- how scorecard numbers changed over time
- whether the same paper claim is repeatedly blocked for the same reason

## 9. Experiment-log coupling

Every paper-facing row in:

- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`

must also track:

- whether it requires a review cycle
- the latest review-cycle id touching it
- its current review status
- its current claim-safety band

This prevents experiment results from outrunning reviewer judgment.

## 10. Cadence

Recommended cadence:

1. after every paper rewrite pass
2. after every new Distinct5 or strict-750 result that could change claims
3. before any large wording escalation in abstract, intro, or conclusion
4. immediately after every formal remote run that is a keep-candidate for the
   paper

If experiments are moving faster than writing, batch them into one review cycle
per meaningful checkpoint, not per checkpoint file.
