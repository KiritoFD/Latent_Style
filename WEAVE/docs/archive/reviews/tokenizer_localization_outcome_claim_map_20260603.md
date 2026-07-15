# Tokenizer Localization Outcome Claim Map

Date: 2026-06-03

Scope: paper-safe claim map for the Distinct5 tokenizer-localization packet
only.

Source note:

- packet docs read:
  - `docs/experiments/2026-06-03-tokenizer-localization/README.md`
  - `docs/experiments/2026-06-03-tokenizer-localization/launch_manifest_20260603.md`
- the README also references
  `docs/experiments/2026-06-03-tokenizer-localization-probe-protocol.md`,
  but that file is not currently present in the workspace; the claim map below
  therefore follows the landed packet README plus the launch manifest's success
  gate and interpretation contract.

## Pattern map

### 1. Style-branch wins clearly

Safe claims:

- "In this matched `L e1` localization packet, refreshing the style-side branch
  yielded the larger improvement."
- "For this packet, tokenizer-side control remains the stronger bottleneck
  candidate than executor-side refresh alone."
- "This outcome keeps tokenizer representation as a live mechanism target in
  the current `L` family."

Unsafe overclaims:

- "The tokenizer is proven to be the sole bottleneck."
- "The correct next tokenizer design is now identified."
- "This generalizes to all families, including the blocked `H` packet."

### 2. Executor-only wins clearly

Safe claims:

- "In this matched `L e1` localization packet, executor-side refresh yielded
  the larger improvement."
- "For this packet, the reviewed `L e1` style-side control was more usable than
  the current executor allowed."
- "This shifts the immediate bottleneck suspicion toward execution rather than
  raw tokenizer-code quality in this family."

Unsafe overclaims:

- "Tokenizer geometry is unimportant."
- "Execution has been proven to be the sole bottleneck everywhere."
- "Tokenizer design is no longer a meaningful research axis."

### 3. Both improve materially

Safe claims:

- "In this matched `L e1` localization packet, both style-side and executor-side
  refresh improve the result, so the bottleneck remains joint."
- "The packet argues against a single-cause story and supports a coupled
  representation-plus-execution reading."
- "Further localization should stay two-sided rather than collapsing to one
  branch."

Unsafe overclaims:

- "Both branches are equally responsible in a theorem-like sense."
- "The tokenizer theory is now closed."
- "The right architectural fix is obvious from this packet alone."

### 4. Neither improves materially

Safe claims:

- "In this matched `L e1` localization packet, neither one-sided refresh
  materially improves the reviewed point."
- "This packet is negative evidence for simple one-branch localization."
- "Under the current setup, the bottleneck is not cleanly resolved by
  refreshing only the style branch or only the executor."

Unsafe overclaims:

- "The current tokenizer/executor pair is already optimal."
- "Tokenizer research is no longer needed."
- "The localization question is closed in general."

## Global paper-safe rule

Regardless of which pattern lands, the packet supports only an `L`-family
mechanism reading. It must not be written as:

- recovery of the original blocked `H` story,
- a family-generic theorem,
- or proof that the next tokenizer factorization is already known.

## One-line usage rule

Use the localization packet to narrow **where the current `L`-family bottleneck
appears to sit**, not to declare a universal tokenizer or executor law.
