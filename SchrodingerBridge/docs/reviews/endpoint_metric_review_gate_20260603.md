# Endpoint-Metric Review Gate

Date: 2026-06-03
Owner lane: `standby_adversarial_reviewer`
Scope: repaired endpoint trio only (`MSE / Huber / L1`)
Purpose: define the exact evidence required for this packet to move the reviewer stance above `weak_reject`, and define the result patterns that still fail.

## 0. Current packet status

Current status after the completed repaired trio:

- packet completeness: `satisfied`
- activation proof: `satisfied`
- current outcome pattern: `Pass pattern B`
- lane interpretation: `negative closure`

Reviewer reading:

- the repaired endpoint trio is now a valid activated packet;
- it closes the endpoint-only question in the negative direction;
- it removes the old paper/code mismatch and the old `inactive probe` rejection route;
- it does **not** by itself upgrade the whole paper above `weak_reject`, because the broader review still depends on the SA-SWD isolation gate and the normalized efficiency gate.

## 1. Minimum evidence packet required

The endpoint-metric review gate was not open until all of the following existed for the repaired trio:

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

This packet is now complete. Without this full packet, the reviewer would treat the trio as incomplete and keep `weak_reject`.

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

Current repaired-trio fit:

- `MSE` achieves roughly `0.6998 / 0.5170` at best and is far behind the reviewed H mainline on LPIPS;
- `Huber` and `L1` recover to roughly `0.3557` / `0.3552` LPIPS, but still remain materially behind the reviewed H mainline reference of roughly `0.6994 / 0.3213`;
- the packet therefore currently satisfies `Pass pattern B`, not `Pass pattern A`.

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

Current endpoint-only decision:

- conditions `1/2/3` are satisfied for a **negative closure** reading;
- the endpoint-metric lane can now be treated as closed in the negative direction;
- the allowed narrow claim is:
  - activated endpoint-only pointwise supervision was tested directly,
  - the repaired trio did not recover the current W1-style mainline frontier,
  - therefore pure endpoint-only pointwise supervision is not the source of the mainline gains in this family.

What is still not allowed from this packet:

- claiming that all latent-space `MSE/L2` is broadly invalid;
- claiming that `Huber` or `L1` is a proven global winner;
- claiming that the whole paper is now past `weak_reject` without the remaining review gates.
