# Endpoint-Metric Review Gate

Date: 2026-06-03
Owner lane: `standby_adversarial_reviewer`
Scope: repaired endpoint trio only (`MSE / Huber / L1`)
Purpose: define the exact evidence required for this packet to move the reviewer stance above `weak_reject`, and define the result patterns that still fail.

## 1. Minimum evidence packet required

The endpoint-metric review gate is not open until all of the following exist for the repaired trio:

- all three repaired configs:
  - `endpoint_metric_h_omf_flow_mse_seed42.json`
  - `endpoint_metric_h_omf_flow_huber_seed42.json`
  - `endpoint_metric_h_omf_flow_l1_seed42.json`
- one resolved-config artifact per arm proving:
  - `objective_mode = omf`
  - `w_flow > 0`
  - `loss_type` matches the arm
  - `terminal_swd_weight = 0.0`
- one train log per arm
- one strict full-eval summary bundle per arm for all evaluated epochs
- one compact comparison table or CSV that identifies:
  - best epoch per arm
  - full `clip_style`
  - full `content_lpips`
  - transfer `clip_style`
  - transfer `content_lpips`
  - wall time
- one explicit comparison against the reviewed H mainline reference, not just arm-vs-arm comparison

Without this full packet, the reviewer treats the trio as incomplete and keeps `weak_reject`.

## 2. What evidence would change the stance

The stance may improve only if the repaired trio shows a clear, activated, reviewer-usable conclusion. Any one of the following is sufficient:

### Pass pattern A - strong positive metric result

At least one activated local-loss arm:

- beats repaired `MSE` by a visible margin on the same scope, and
- does so without collapsing LPIPS relative to the repaired packet, and
- remains competitive enough to matter against the reviewed H mainline

Reviewer reading:

- local-loss choice is not decorative;
- a non-MSE local residual has measurable value on the active endpoint path.

### Pass pattern B - strong negative but decisive result

All activated pointwise endpoint arms:

- remain clearly worse than the reviewed H mainline on LPIPS/content-preserving trade-off, and
- do so consistently across the trio, not by one noisy checkpoint

Reviewer reading:

- the packet still changes the stance because it closes the question in the opposite direction;
- the paper can safely argue that pure pointwise endpoint matching is not the source of the mainline gains.

### Pass pattern C - parity with clear null conclusion

If `MSE / Huber / L1` remain near-identical under the activated packet, the stance may still improve only if:

- the parity is demonstrated on a truly activated path, and
- the manuscript claim is narrowed to:
  - endpoint-side activation was tested directly,
  - local kernel choice did not materially change the outcome in this packet,
  - the stronger story remains OT + terminal distribution matching rather than local residual geometry

Reviewer reading:

- this does not prove Huber/L1 superiority, but it does close the paper/code mismatch and removes one direct rejection route.

## 3. Result patterns that still fail

The endpoint trio does **not** move the review above `weak_reject` if any of the following occurs:

### Fail pattern 1 - inactive or ambiguous activation

- missing resolved-config proof
- nonzero prose claims but no artifact proving `w_flow > 0`
- accidental coexistence of another active style-driving term that makes the compared endpoint term uninterpretable

### Fail pattern 2 - only arm-vs-arm, no mainline context

- the trio is compared only internally
- no direct placement against the reviewed H mainline reference

Reason:

- reviewer cannot tell whether the packet matters for the actual paper claim.

### Fail pattern 3 - noisy winner with no stable conclusion

- one arm wins by a tiny margin at one checkpoint only
- ordering flips by epoch or scope
- differences are too small to support a writing change

Reason:

- this is still weak evidence and cannot carry a claim change.

### Fail pattern 4 - all repaired arms are simply bad

- activated `MSE / Huber / L1` all land in the current repaired-MSE regime of roughly high LPIPS / weak frontier placement, and
- no clean conclusion is written down against the mainline

Reason:

- bad results alone do not help unless they are turned into a clear negative closure.

### Fail pattern 5 - overclaim after null or negative result

- the packet shows parity or failure, but the paper still escalates into
  - "latent MSE is disproven everywhere", or
  - "Huber/L1 are proven better", or
  - "local metric correction is the key driver"

Reason:

- this recreates the same rejection route under a new packet.

## 4. Reviewer decision rule

For this packet alone, the reviewer may move from `weak_reject` to `narrow_only / conditional` only if:

1. the repaired trio is fully activated and fully logged;
2. the result supports one of `Pass pattern A/B/C`;
3. the write-up uses exactly the conclusion that the packet supports, and no stronger one.

Otherwise the endpoint-metric lane remains open and the standing stance stays at `weak_reject`.
