# AAAI 2027 Reviewer Roster

Updated: 2026-06-03

This file defines the standing independent reviewer lanes for the paper. The
goal is persistence and inspectability, not one-off commentary.

## 1. Lane ownership

Each review cycle must assign three distinct subagents:

1. `adversarial_review`
   - responsibility:
     - strongest rejection case
     - overclaim detection
     - fairness / comparison-risk detection

2. `scorecard`
   - responsibility:
     - AAAI-style scoring
     - track whether scores actually improve across cycles

3. `experiment_audit`
   - responsibility:
     - claim-to-evidence closure map
     - next minimum experiment needed for promotion

No agent may cover multiple lanes in the same review cycle.

## 2. Independence rules

1. the consensus note must not be authored by one of the three lane agents;
2. if the same agent is reused in later cycles, reuse is acceptable only across
   cycles, not across lanes within the same cycle;
3. every cycle must record the lane agent nickname or id in
   `aaai2027_review_score_log.csv`;
4. if one lane repeatedly returns the same blocker for two or more cycles, that
   blocker is treated as standing paper debt until a new experiment closes it.

## 3. Minimal lane packet

Every lane should receive:

- checkpoint label
- review-cycle id
- current manuscript path
- current working index path
- current experiment master log path
- relevant benchmark or figure evidence path
- previous consensus note

## 4. Output bands

Every lane returns:

- `overall_status`
- `claim_safety_band`
- `evidence_closure_band`
- `blocking_issue`
- `next_action_1`
- `next_action_2`
- `support_score`
- `fairness_score`
- `artifact_path_score`
- `closure_value_score`

The `scorecard` lane also returns:

- `novelty`
- `technical_quality`
- `experimental_rigor`
- `clarity`
- `reproducibility`
- `significance`

## 5. Escalation rule

A new experiment result is not paper-facing until at least one independent
review cycle has touched it and the resulting consensus has been recorded in:

- `aaai2027_review_registry.csv`

The paper is not allowed to escalate a claim based only on raw metrics or a
single favorable visual inspection.

## 6. Standing agent roster

Current named reviewer agents that may be reused across cycles:

- `Avicenna`
  - preferred lane: `adversarial_review`
- `Harvey`
  - preferred lane: `scorecard`
- `Kepler`
  - preferred lane: `experiment_audit`
- `Lorentz`
  - preferred lane: `adversarial_review`
  - note:
    - use as a second harsh reader when the current cycle needs a fresh attack
      on claims, figure logic, or experiment prioritization
