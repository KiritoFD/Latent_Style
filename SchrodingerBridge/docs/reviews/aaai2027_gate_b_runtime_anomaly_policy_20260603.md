# AAAI 2027 Gate B Runtime-Anomaly Policy

Date: 2026-06-03  
Lane: `adversarial_review`  
Question: if the abnormal `random-axis` SA-SWD run finishes and yields usable full-eval summaries, can Gate B close on quality-only evidence, or does reviewer-safe practice still require a rerun?

## Short answer

Yes, Gate B **can** close on quality-only evidence **without** an immediate rerun, but only under a narrow policy:

- the abnormality must be clearly runtime-only;
- the matched semantic/random configs must still be demonstrably aligned except for `terminal_swd_axis_source`;
- the random run must complete with clean full-eval summaries;
- the resulting semantic-vs-random conclusion must be stable enough to support a paper-safe claim boundary.

If any of those conditions fail, reviewer-safe practice requires a rerun.

## Why a quality-only closure is defensible

Current evidence already points in the right direction for a quality-only reading:

- the packet was designed as a matched Gate B ablation with only `semantic` vs `random` axis source changed;
- the semantic arm completed normally;
- the random arm has crossed into a degraded-throughput regime, but the current diagnosis ranks:
  1. random-axis path itself as the most likely cause,
  2. generic host/runtime interference as secondary,
  3. accidental config mismatch as unlikely;
- both the experiment README and the master log already quarantine the random-arm wall clock as abnormal and not formal-speed evidence.

That means the current anomaly does **not** automatically poison the quality comparison. It only poisons any speed interpretation.

## Accept criteria: Gate B may close without rerun

Gate B may close on the current abnormal random-arm run if **all** of the following are true:

1. **Matched-config integrity is preserved**
   - resolved configs remain matched except for `terminal_swd_axis_source`;
   - no evidence of accidental batch/seed/weight/objective drift;
   - no hidden fallback that changes the actual math path.

2. **Run completion is operationally clean enough**
   - the random arm reaches its intended training end;
   - full-eval summaries exist for the planned epochs;
   - there is no crash-restart chain, manual resume from a different state, or silent partial-eval artifact that breaks comparability.

3. **The anomaly is quarantined to runtime, not correctness**
   - no NaN/Inf instability;
   - no OOM-triggered automatic behavior change;
   - no evidence that the degraded run altered the optimization contract itself.

4. **The semantic-vs-random result is conclusion-stable**
   - semantic beats random on at least one headline dimension without unacceptable regression, **or**
   - random matches/beats semantic clearly enough that the paper must downgrade the SA-SWD novelty claim;
   - the conclusion is not resting on one tiny, noisy, single-epoch wiggle.

5. **The paper uses the correct boundary**
   - the result is used only to close the semantic-vs-random quality question;
   - the abnormal random-arm wall clock is excluded from any efficiency or fairness claim.

Under these five conditions, Gate B can close as a **quality-only matched ablation**.

## Reject criteria: rerun is still required

Reviewer-safe practice still requires a rerun if **any** of the following holds:

1. **Config ambiguity remains**
   - there is no trustworthy resolved-config proof;
   - the random arm may have drifted in more than just axis source.

2. **Completion is not truly matched**
   - missing epochs;
   - missing full-eval summaries;
   - resumed or hand-patched state that breaks comparability.

3. **Runtime anomaly may have changed optimization behavior**
   - OOM recovery;
   - altered precision/path behavior;
   - hidden fallback or adaptive behavior that changed the training contract.

4. **The quality result is too weak to be decision-grade**
   - semantic vs random is nearly tied;
   - the winner flips by epoch/scope with no stable story;
   - the margin is too small to justify a manuscript-level novelty claim or downgrade.

5. **The team wants to make any speed or practicality implication from this packet**
   - the current abnormal random run cannot support that;
   - a cleaner rerun would be mandatory for any runtime-side argument.

## Recommended policy

Use a **two-tier closure policy**:

### Tier 1 - quality-only Gate B closure

Accept the current abnormal random-arm run **if** it finishes cleanly and yields matched summaries under the accept criteria above.  
Then:

- close Gate B for **quality/novelty boundary only**;
- explicitly annotate the packet as:
  - `quality-usable`
  - `runtime-abnormal`
  - `not admissible for speed evidence`

### Tier 2 - optional clean rerun

Schedule a rerun only if one of these becomes necessary:

- the semantic-vs-random result is borderline/noisy;
- the paper still wants stronger semantic-axis rhetoric than the current margin supports;
- someone wants to discuss runtime/practicality implications of semantic vs random axes.

## Reviewer-safe bottom line

If the abnormal random-axis run finishes with clean, matched full-eval summaries, my recommendation is:

- **close Gate B on quality-only evidence without forcing an immediate rerun**,
- **forbid any runtime interpretation from that run**,
- **require a rerun only if the quality conclusion is ambiguous or if a speed-side claim is desired**.
