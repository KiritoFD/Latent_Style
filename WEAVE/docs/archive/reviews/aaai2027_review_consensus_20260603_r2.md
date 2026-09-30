# AAAI 2027 Review Consensus - R20260603K

Date: 2026-06-03

Checkpoint label:

- `current_paper_after_four_lane_reaudit_with_landed_localization_and_timing_v1`

Lane agents used:

- `Averroes` - `adversarial_review`
- `Darwin` - `scorecard`
- `Aristotle` - `experiment_audit`
- `James` - `figure_audit`

Primary inputs:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_review_score_log.csv`
- `SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/README.md`

## Consensus status

Current consensus remains:

- `weak_reject`

But the blocker stack is now clearer than before:

1. the paper's best supported story is already known;
2. several older "still pending" items are actually landed and should stop
   being described as pending;
3. the next lift is no longer broad exploration, but a smaller set of
   evidence-closure and figure-surface fixes.

## Shared strengths

All four lanes agree the following are real and worth keeping:

1. the `idt` / no-op framing is paper-worthy;
2. the Distinct5 stress-benchmark argument is the paper's sharpest evaluation
   contribution;
3. the repaired endpoint-only packet gives a valid negative closure;
4. the current paper surface is more disciplined and less overclaimed than the
   earlier versions.

## Shared blockers

### 1. Mechanism evidence is still only partially closed

Safe now:

- OT-coupled endpoint construction and `W1`-style terminal matching remain the
  strongest supported mechanism reading;
- endpoint-only pointwise supervision is negatively closed;
- SA-SWD is a retained mainline design choice, not a proven semantic-axis win.

Not safe yet:

- broad latent-metric-correction rhetoric;
- decisive `Huber/L1` language;
- tokenizer theory as a closed headline story;
- theorem-to-path closure beyond bounded local support.

### 2. Tokenizer localization is landed, but only as `L`-family-local evidence

The new cycle agrees that tokenizer localization should no longer be treated as
"still pending." The landed `L e1` packet currently supports one narrow read:

- executor-side refresh is stronger than style-side refresh alone on the
  matched `L e1` surface.

What it does **not** support:

- family-generic tokenizer closure;
- restored `H`-family continuity;
- a contribution-level statement that tokenizer theory is solved.

### 3. Efficiency is usable only as bounded timing context

The timing artifact is now good enough to support bounded paper language, but
not strong enough for reopened speedup rhetoric. The paper should either:

1. keep timing as operating-point / timing-context evidence only; or
2. pay for a stronger matched parity packet.

### 4. The evidence figures are still weaker than the paper's best claims

The figure lane's main diagnosis is that the results story is fragmented across
too many partially convincing visuals.

Most important figure action:

- remake the Distinct5 result spine as one integrated frontier-plus-cases
  figure;
- demote `figures/fig_distinct5_time_context.pdf` from the main paper first;
- merge or demote the standalone zoom figure unless crop provenance becomes
  explicit.

## Ordered next actions

1. **Paper-side boundary pass**
   - update the manuscript so tokenizer-localization and time-to-parity are
     written as landed, bounded evidence rather than as pending closure;
   - keep SA-SWD and latent-metric language at the current negative-closure /
     narrow-only boundary.

2. **Figure-surface repair**
   - demote `figures/fig_distinct5_time_context.pdf` from the main paper;
   - plan one integrated Distinct5 visual spine combining frontier, cases, and
     zoom crops.

3. **Next remote 3060 experiment**
   - launch the Distinct5 path-stability / weakened-kinetic packet;
   - this is the highest-value unblocked remote closure left.

4. **Blocked-but-high-value follow-up**
   - revisit `H`-family tokenizer execution-alignment or localization only if
     checkpoint recovery becomes policy-safe.

## Submission-state read

Current paper state:

- not submission-safe;
- no longer drifting randomly;
- one good review cycle away from `borderline` if the paper absorbs landed
  localization/timing evidence cleanly and the next remote packet closes
  kinetic/path support;
- still short of `weak_accept` until at least one central mechanism claim is
  positively closed under a reviewer-safe protocol.
