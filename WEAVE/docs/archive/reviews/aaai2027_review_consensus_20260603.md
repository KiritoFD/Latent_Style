# AAAI 2027 Review Consensus

Date: 2026-06-03

Source reviewer memos:

- `aaai2027_adversarial_review_20260602.md`
- `aaai2027_scorecard_20260602.md`

This note extracts the overlap between independent reviewer lanes so the main
project plan is driven by repeated objections rather than by whichever memo is
longest.

## Consensus status

Current consensus is:

- the project has a real paper core;
- the current submission would still be rejected;
- the bottleneck is evidence closure and comparison hygiene, not lack of ideas.

## Shared positives

Both reviewers independently agree on three strengths:

1. the `idt` / no-op framing is real and paper-worthy;
2. Distinct5-512 is the cleanest current stress benchmark;
3. the tokenizer-vs-renderer diagnosis is more interesting than a naive
   "bigger style embedding" story.

## Shared rejection risks

### 1. The latent-metric thesis is ahead of the evidence

Consensus:

- the current paper can safely claim that style alignment is not driven by
  latent endpoint MSE;
- it can safely claim that `W1`-style terminal alignment matters;
- it cannot yet safely claim that `Huber/L1` flow residuals are a proven driver
  of the headline results.

Required closure:

- the originally launched `MSE` / `Huber` / `L1` block is no longer valid as
  theory evidence after the config audit.
- the next required closure is a \emph{repaired} endpoint-metric ablation in
  which the switched loss term actually lies on the active transport path.

Config-audit update:

- the completed `MSE` / `Huber` / `L1` Distinct5 runs resolved to
  `objective_mode=omf` with `w_flow=0.0`;
- changing `loss_type` therefore did not activate the intended comparison term;
- the resulting trio remains useful as operational stability evidence only.

### 2. The efficiency claim is still attackable

Consensus:

- current timing evidence is useful for research decisions;
- it is not yet normalized enough for aggressive paper-level speedup rhetoric.

Required closure:

- one same-clock time-to-parity figure with explicit hardware and stop
  criterion.

### 3. SA-SWD still needs a decisive novelty ablation

Consensus:

- terminal SWD clearly matters;
- semantic projection-axis selection is still not isolated strongly enough.

Required closure:

- semantic-axis vs random-axis ablation on the same Distinct5 base config.

### 4. The paper still carries too many messages at once

Consensus:

- the strongest current story is:
  1. no-op-adjusted evaluation exposes metric illusion;
  2. LBM gives a strong content-preserving frontier;
  3. tokenizer capacity alone is not the main bottleneck.

Anything outside that should be supported more narrowly or deprioritized.

## Immediate writing policy

Until new experiments land, the paper should avoid:

- broad "distance correction is proven" wording;
- universal speedup claims;
- rhetoric that implies all tokenizer questions are closed;
- treating theorem-support numbers as self-evident without artifact paths.

The paper can continue to say:

- the measured runs support the endpoint-side `W1` story;
- raw `CLIP-S` is unsafe without `idt`;
- Distinct5-512 is the current strongest content-preserving benchmark slice.

## Immediate experiment order

1. `endpoint_metric_ablation_repaired`
   - either `objective_mode=omf` with `w_flow>0`, or a non-`omf` path where
     `loss_type` is the active transport penalty
   - remote 3060 first
2. `sa_swd_axis_ablation`
   - semantic vs random axes
   - same Distinct5 config family
3. `time_to_parity_curve`
   - unify wall-clock accounting
4. `path_stability_probe`
   - theorem-support, but only after the first two are running

## Submission gate

Do not call the paper AAAI-safe until all of the following are true:

1. the metric claim is backed by a direct and activated ablation;
2. the efficiency claim is backed by a normalized time-to-parity plot;
3. the no-op insight is elevated into a formal evaluation contribution in the
   manuscript and tables;
4. the review lane is no longer converging on `reject`.
