# R20260603J Adversarial Preread Memo

Date: 2026-06-03

Lane: `adversarial_review`

Scope: scoring-admission rule for the Distinct5 tokenizer-localization packet
only.

Inputs:

- `aaai_submission/paper_aaai2026.tex`
- `docs/reviews/tokenizer_localization_claim_boundary_20260603.md`
- `docs/experiments/2026-06-03-tokenizer-localization/README.md`
- `docs/reviews/aaai2027_review_protocol.md`

## Purpose

This memo defines when the tokenizer-localization packet is admissible for
`R20260603J`, and how different landing patterns should change paper-safe
claims and review scores. It is a preread rule sheet, not a manuscript edit and
not a result interpretation memo.

## Admission conditions for R20260603J

The cycle should be scored only if all of the following are true:

1. both matched arms are fully landed:
   - `stylebranch`: `epoch_0001`, `epoch_0002`, `epoch_0003`
   - `executoronly`: `epoch_0001`, `epoch_0002`, `epoch_0003`
2. both arms have an auditable path from config and launch mode to
   `remote_train.log` and `full_eval/.../summary.json`;
3. both arms are read under the same checkpoint-selection rule;
4. the primary readout is no-op-aware:
   - `delta_idt` or equivalent no-op-adjusted style gain first,
   - then `content_lpips`,
   - then `clip_dir`,
   - then any geometry diagnostic;
5. a raw `CLIP-S` rise without acceptable no-op-adjusted gain or with obvious
   `LPIPS` collapse does not count as a winner;
6. interpretation stays local to the current Distinct5 `L e1` surface.

If any item above fails, `R20260603J` should be recorded as `deferred` in
practice, and the paper should remain at the current mechanism status:

- `overall_status`: `weak_reject`
- `claim_safety_band`: `narrow_only`
- `evidence_closure_band`: `partial`

## Classification rule before scoring

Use the following decision order:

1. determine whether each arm produces a real positive no-op-adjusted gain
   under acceptable content cost;
2. only then ask whether `stylebranch` or `executoronly` wins;
3. if the arms are numerically close, separate:
   - `joint/localized tie`: both are meaningfully above the reviewed base; or
   - `negative localization`: both remain near zero, contradictory, or below
     the no-op-adjusted expectation.

This prevents "close because both failed" from being miswritten as "shared
bottleneck solved."

## Outcome-to-claim and score rules

### 1. `stylebranch` wins clearly

Allowed paper move:

- "Within the matched Distinct5 `L e1` localization packet, style-side refresh
  yields the larger recoverable gain."
- "For this packet, tokenizer-side control remains the stronger recoverable
  bottleneck candidate than executor-side refresh alone."

Disallowed move:

- any sole-bottleneck, global-tokenizer, or family-generic theorem wording.

Score effect:

- target-claim `support_score`: promote to `2`
- target-claim `fairness_score`: promote to `2` only if the matched no-op-aware
  comparison is complete; otherwise keep at `1`
- target-claim `artifact_path_score`: `2` if provenance is complete
- target-claim `closure_value_score`: promote to `2`
- paper-level status cap: may improve from `weak_reject` to `borderline`, but
  not beyond, and only if the manuscript keeps the claim local to this packet

### 2. `executoronly` wins clearly

Allowed paper move:

- "Within the matched Distinct5 `L e1` localization packet, executor-side
  refresh yields the larger recoverable gain."
- "For this packet, the reviewed style-side control appears more usable than
  the current executor allowed."

Disallowed move:

- any claim that tokenizer design no longer matters, or that execution is
  proven to be the only bottleneck everywhere.

Score effect:

- target-claim `support_score`: promote to `2`
- target-claim `fairness_score`: promote to `2` only under the same matched
  no-op-aware conditions above
- target-claim `artifact_path_score`: `2` if provenance is complete
- target-claim `closure_value_score`: promote to `2`
- paper-level status cap: may improve from `weak_reject` to `borderline` only
  if tokenizer rhetoric is tightened accordingly; stale tokenizer-forward
  overclaim keeps the paper at `weak_reject`

### 3. both arms are close, but both clear the idt-adjusted bar

Allowed paper move:

- "Within the matched Distinct5 `L e1` localization packet, neither branch
  dominates cleanly; the bottleneck remains joint."
- "The packet argues against a tokenizer-only or executor-only story on this
  surface."

Disallowed move:

- theorem-like equality claims, final-architecture closure, or universal
  statements about representation theory.

Score effect:

- target-claim `support_score`: promote to `2`
- target-claim `fairness_score`: promote to `2` if both arms were compared
  under the same no-op-aware rule
- target-claim `artifact_path_score`: `2`
- target-claim `closure_value_score`: promote to `2`
- paper-level status cap: may improve from `weak_reject` to `borderline`
  because the packet closes the single-cause localization question locally,
  even though it does not select one branch as the winner

### 4. both arms fail, or both stay below the idt-adjusted expectation

Allowed paper move:

- "Within the matched Distinct5 `L e1` localization packet, one-sided refresh
  of either branch does not materially improve the reviewed point."
- "This is negative evidence against simple one-branch localization on the
  current surface."

Disallowed move:

- any positive mechanism-closure claim,
- any "model already optimal" language,
- any "tokenizer story is now proven" language.

Score effect:

- target-claim `support_score`: at most `1`
- target-claim `fairness_score`: can still be `2` if the negative result is
  fully matched and auditable
- target-claim `artifact_path_score`: `2` if provenance is complete
- target-claim `closure_value_score`: at most `1`
- paper-level status: stays `weak_reject` unless the manuscript rewrites this
  packet explicitly as negative local evidence rather than as a positive
  tokenizer-mechanism result

## Review-policy summary

For `R20260603J`, the main score change is allowed only when the packet closes
the localization question in a matched, no-op-aware, packet-local way. A clear
winner or a real positive tie can upgrade the mechanism subclaim to direct
support. A dual failure can close the probe negatively, but it does not justify
stronger tokenizer or renderer claims and should not lift the paper out of
`weak_reject` on its own.
