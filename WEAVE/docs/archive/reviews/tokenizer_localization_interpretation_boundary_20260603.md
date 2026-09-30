# Tokenizer Localization Interpretation Boundary

Date: 2026-06-03

Scope: theory / mathematical-boundary memo for what the tokenizer-localization
packet can and cannot establish.

Inputs read:

- `docs/reviews/tokenizer_claim_matrix_20260603.md`
- `docs/reviews/tokenizer_claim_evidence_matrix_20260603.md`
- `docs/reviews/tokenizer_representation_theory_queue_20260603.md`
- `docs/reviews/aaai2027_theory_owner_audit_20260603.md`
- `docs/reviews/tokenizer_localization_claim_boundary_20260603.md`
- `docs/reviews/tokenizer_localization_outcome_claim_map_20260603.md`
- `docs/experiments/2026-06-03-tokenizer-localization/README.md`

## Global rule

The localization packet is an identification probe on one question only:

- when one branch is refreshed and the other is frozen, where does recoverable
  end-to-end gain come from on the current Distinct5 `L e1` surface?

It is **not** by itself:

- a proof that the tokenizer is good or bad in general,
- a proof that the executor is good or bad in general,
- a proof that the bottleneck is unique rather than shared,
- or a proof that the observed ceiling is not partly metric-driven.

## What the packet can prove in principle

If both arms are fully landed and read under the same evaluation contract, the
packet can support a **local comparative claim** of the form:

> on the current matched `L e1` surface, branch X yields larger recoverable
> improvement than branch Y under one-sided refresh.

That is a directional localization claim, not a full mechanism theorem.

## Four explanations that must be kept separate

### 1. Style-side weakness

Meaning:

- the style-side control object `T_phi(s)` is currently the tighter limit;
- the executor can use better style-side control if it is provided.

Minimum evidence condition:

1. `style-branch` and `executor-only` arms both fully land with the same
   metrics and the same selection rule.
2. The `style-branch` arm shows a clear end-to-end advantage over the
   `executor-only` arm on the main readout, preferably:
   - higher `delta_idt`, and
   - better or comparable `LPIPS`, not only higher raw `CLIP-S`.
3. The gain is not explainable as a trivial metric move into a clearly damaged
   region, meaning artifact-sensitive diagnostics do not collapse while style
   rises.
4. If geometry diagnostics are available, executed-output separation should
   improve in the same direction as the no-op-adjusted style gain.

What this would support:

- "For the current `L e1` packet, style-side refresh is the stronger
  recoverable source of gain."

What it would still not support:

- "The tokenizer is the sole bottleneck."
- "The current tokenizer representation is globally wrong."
- "The next tokenizer factorization is proven."

### 2. Executor-side weakness

Meaning:

- the reviewed style-side control already contains usable target information;
- the executor is currently failing to realize it.

Minimum evidence condition:

1. Both matched arms fully land under the same metric contract.
2. The `executor-only` arm shows a clear end-to-end advantage over the
   `style-branch` arm on `delta_idt` with non-catastrophic `LPIPS`.
3. The gain is not merely raw-style inflation with obvious artifact or content
   collapse.
4. If geometry diagnostics are available, executed separation or generated
   movement should improve in the same direction as the style gain.

What this would support:

- "For the current `L e1` packet, executor refresh recovers more than
  style-side refresh alone."

What it would still not support:

- "Tokenizer design no longer matters."
- "Execution is the only bottleneck everywhere."
- "Code-space geometry is irrelevant."

### 3. Shared bottleneck

Meaning:

- both sides matter materially;
- neither a tokenizer-only nor executor-only story is sufficient.

Minimum evidence condition:

1. Both matched arms fully land.
2. Both arms improve materially over the shared base, or neither arm separates
   cleanly from the other while both remain meaningfully above noise.
3. The improvements are measured on no-op-adjusted style movement together with
   content cost, not on raw style alone.
4. There is no clean monotone winner across the main operating points.

What this would support:

- "The current bottleneck is coupled across representation and execution on the
  matched `L e1` surface."

What it would still not support:

- "Both sides are equally responsible in a theorem-like sense."
- "Tokenizer theory is now closed."
- "The correct architectural decomposition is known."

### 4. Metric artifact

Meaning:

- the apparent winner is produced mainly by metric behavior rather than better
  stylization;
- localization conclusions would then be overstated.

Minimum evidence condition for raising this explanation seriously:

1. The arm that "wins" on raw `CLIP-S` does not also win on `delta_idt`, or
   gains style only by moving into a much worse `LPIPS` / artifact regime.
2. Identity / no-op baselines remain strong enough that raw style movement is
   ambiguous without subtraction.
3. Competing arms change the metric profile in contradictory ways, such as
   style gain with broad artifact deterioration.
4. There is no supporting executed-geometry evidence aligned with the claimed
   stylistic improvement.

What this would support:

- "The packet is not yet clean localization evidence because the apparent gain
  may be metric-driven."

What it would still not support:

- "The localization packet is useless."
- "All tokenizer conclusions are invalid."

## Current paper-safe reading

Given the current document state, the packet is still a queued / partial
localization line. Therefore the present safe statement is only:

- this packet is the right identifying probe for separating style-side and
  executor-side weakness on the current `L e1` surface;
- until both arms are fully landed and reviewed under the same no-op-aware
  readout, none of the four explanations above is closed.

## Minimal evidence table

| explanation | minimum evidence needed | still not proven even if satisfied |
| --- | --- | --- |
| style-side weakness | both arms landed; style-branch wins on `delta_idt` with acceptable `LPIPS` and no artifact collapse | tokenizer is sole/global bottleneck |
| executor-side weakness | both arms landed; executor-only wins on `delta_idt` with acceptable `LPIPS` and no artifact collapse | execution is sole/global bottleneck |
| shared bottleneck | both arms landed; both help materially or neither separates cleanly; read on no-op-aware tradeoff | equal causal responsibility or final architecture |
| metric artifact | winner depends on raw `CLIP-S` but not `delta_idt`, or requires severe damage/artifacts | total invalidity of tokenizer story |

## One-line rule for future interpretation

Tokenizer-localization results identify **where recoverable gain appears under
one-sided refresh**; they do not, by themselves, decide whether the true cause
is representation weakness, execution weakness, a shared bottleneck, or metric
artifact unless the minimum evidence conditions above are met.
