# AAAI 2027 Continuous Review Protocol

Updated: 2026-06-02

This protocol makes the reviewer lane a standing control loop instead of a
one-off paper critique.

## 1. Trigger conditions

Run a new review round whenever at least one of the following is true:

1. the manuscript changes in a way that affects claims, framing, or tables;
2. a new paper-facing experiment family lands;
3. a baseline comparison changes;
4. an evaluation contract changes;
5. a new theorem, tokenizer claim, or efficiency claim is introduced.

Do not wait for a "finished" draft. The point is to catch drift early.

## 2. Required independent lanes

Every round must contain three independent subagent reviews:

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

One agent must not play multiple lanes in the same round.

## 3. Minimum inputs per round

Every reviewer round should read the same minimum package:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/README.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- current comparison or benchmark evidence directories referenced by the paper
- previous consensus note, if one exists

If a round is about a specific claim, add the narrow evidence bundle for that
claim instead of broadening the read set arbitrarily.

## 4. Output contract

Each round creates four files:

1. `aaai2027_adversarial_review_YYYYMMDD.md`
2. `aaai2027_scorecard_YYYYMMDD.md`
3. `aaai2027_experiment_audit_YYYYMMDD.md`
4. `aaai2027_review_consensus_YYYYMMDD.md`

If multiple rounds happen on the same date, append `_r2`, `_r3`, and so on.

The consensus note must include:

- the checkpoint label under review
- the source reviewer files
- shared strengths
- shared blockers
- claims that must be narrowed now
- ordered next experiments
- current submission status:
  - `reject`
  - `weak_reject`
  - `borderline`
  - `safe_to_submit`

## 5. Promotion gates

A claim may move into the abstract, contributions list, or main comparison
table only if all of the following are true:

1. `experiment_audit` marks it as directly supported;
2. `adversarial_review` does not flag it as a fairness or overclaim blocker;
3. the evidence path is explicit in `docs/experiments/` or the paper figure
   directory.

The paper is not "AAAI-safe" until:

1. no reviewer lane calls out an unsupported headline claim;
2. the experiment audit no longer requests a blocking ablation;
3. the scorecard average is at least `7.0/10`;
4. at least two of the three lanes are no worse than `weak_accept` or
   equivalent.

## 6. Registry discipline

Every round must be recorded in:

- `SchrodingerBridge/docs/reviews/aaai2027_review_registry.csv`

This registry is the control plane for:

- current paper risk
- claim status
- next experiment priority
- whether the paper is converging or drifting

## 7. Cadence

Recommended cadence:

1. after every paper rewrite pass
2. after every new Distinct5 or strict-750 result that could change claims
3. before any large wording escalation in abstract, intro, or conclusion

If experiments are moving faster than writing, batch them into one review round
per meaningful checkpoint, not per checkpoint file.
