# Tokenizer Localization Claim Boundary

Date: 2026-06-03

Scope: paper-safe claim boundary for the Distinct5 tokenizer-localization
packet only.

Inputs aligned:

- `goal.md`
- `aaai_submission/paper_aaai2026.tex`
- `docs/reviews/aaai2027_adversarial_gate_refresh_20260603.md`
- `docs/experiments/2026-06-03-tokenizer-localization/README.md`

## Global rule

This packet can localize the current bottleneck only **within the current
Distinct5 `L e1` surface**. It cannot by itself:

- close tokenizer theory globally,
- recover the blocked `H`-family story,
- or prove the final correct tokenizer architecture.

## Outcome map

### 1. Style-branch wins clearly

Legal paper claims:

- "Within the matched Distinct5 `L e1` localization packet, refreshing the
  style-side branch yields the larger improvement."
- "For the current `L e1` surface, tokenizer-side control remains the stronger
  recoverable bottleneck candidate than executor-side refresh alone."
- "This supports keeping tokenizer / representation work active in the next
  design iteration."

Forbidden wording:

- "The tokenizer is proven to be the sole bottleneck."
- "The correct next tokenizer factorization is now identified."
- "This result closes the tokenizer mechanism story for the paper."
- "This generalizes across all families."

### 2. Executor-only wins clearly

Legal paper claims:

- "Within the matched Distinct5 `L e1` localization packet, executor-side
  refresh yields the larger improvement."
- "For this packet, the reviewed `L e1` control signal appears more usable than
  the current executor allowed."
- "This shifts immediate bottleneck suspicion toward execution rather than raw
  tokenizer-code weakness on the current surface."

Forbidden wording:

- "Tokenizer geometry no longer matters."
- "Execution is proven to be the sole bottleneck everywhere."
- "Tokenizer design is no longer a priority."
- "The landed `L` result recovers the original `H` story."

### 3. Both tie or both improve similarly

Legal paper claims:

- "Within the matched Distinct5 `L e1` localization packet, neither side
  dominates cleanly; the bottleneck remains joint."
- "This packet argues against a single-cause tokenizer-only or executor-only
  story."
- "Further mechanism work should continue to treat representation and execution
  as coupled."

Forbidden wording:

- "Both sides are equally responsible in a theorem-like sense."
- "The tokenizer theory is now closed."
- "The right architectural fix is already obvious."
- "A tie proves capacity no longer matters."

### 4. Both fail or neither improves materially

Legal paper claims:

- "Within the matched Distinct5 `L e1` localization packet, one-sided refresh
  of either branch does not materially improve the reviewed point."
- "This is negative evidence against simple one-branch localization on the
  current surface."
- "The remaining bottleneck is not cleanly resolved by refreshing only the
  style branch or only the executor."

Forbidden wording:

- "The current model is already optimal."
- "Tokenizer research is no longer needed."
- "Localization is settled in general."
- "This proves the paper's representation story."

## Wording that must stay forbidden in all four cases

No matter which pattern lands, the following must remain forbidden:

- "tokenizer theory is closed"
- "the correct next tokenizer design is proven"
- "the result recovers the blocked `H`-family continuity claim"
- "this is a family-generic theorem about style representation"
- "the localization packet alone resolves the broader latent-distance story"

## One-line drafting rule

Use this packet only to say **where the current `L e1` bottleneck appears to
sit under a matched freeze-direction probe**, never to claim universal closure
of tokenizer or representation theory.
