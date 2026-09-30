# Repaired Endpoint-Metric Ablation Packet

Date: 2026-06-03

## 1) Recommended option

Recommended option: **stay on `objective_mode = omf`, but set `w_flow > 0` and
turn `terminal_swd_weight = 0.0` for the primary packet**.

Do **not** use the non-OMF path for the main repaired packet if the intended
claim is endpoint-metric behavior. In the current codebase:

- OMF with `w_flow > 0` applies `loss_type` to
  `pred_endpoint` vs `matched_target`
  (`src/losses.py`, `_compute_omf_details`, `flow_loss = self._loss(pred_endpoint, matched_target)`).
- non-OMF applies `loss_type` to
  `pred_velocity` vs `target_velocity`
  (`src/losses.py`, `flow_loss = self._loss(pred_velocity, target_velocity)`).

So:

- **OMF + `w_flow > 0`** probes the endpoint Euclidean-matching story
- **non-OMF** probes velocity-regression kernel choice instead

For the primary repaired packet, use:

- `bridge.objective_mode = "omf"`
- `bridge.w_flow = 1.0`
- `bridge.terminal_swd_weight = 0.0`

This makes the compared kernel the active endpoint-matching term instead of a
near-null side setting.

## 2) Exact config knobs

### Knobs that must differ across `mse / huber / l1`

Only these should differ across the three arms:

- `bridge.loss_type`
  - `mse`
  - `huber`
  - `l1`
- `checkpoint.save_dir`
- `ablation.name`
- `ablation.stage`

Suggested config names:

- `configs/aaai2027/endpoint_metric_h_omf_flow_mse_seed42.json`
- `configs/aaai2027/endpoint_metric_h_omf_flow_huber_seed42.json`
- `configs/aaai2027/endpoint_metric_h_omf_flow_l1_seed42.json`

### Knobs that must stay frozen

Freeze these across all three arms:

- inherited base family:
  - `distinct5_512_ema_variant_h_hard_explore_queue_e3.json`
- `bridge.objective_mode = "omf"`
- `bridge.w_flow = 1.0`
- `bridge.terminal_swd_weight = 0.0`
- `bridge.w_kinetic = 1.0`
- `bridge.kinetic_mode = "endpoint"`
- all OT coupling settings:
  - `bridge.ot_cost_mode`
  - `bridge.sinkhorn_*`
  - `bridge.identity_endpoint`
- all queue / pairing settings:
  - `data.pairing_cache_*`
- all model architecture knobs:
  - tokenizer mode
  - routing mode
  - backbone dimensions
  - style spatial mode
- training contract:
  - `training.seed = 42`
  - `training.batch_size = 44`
  - `training.num_epochs`
  - optimizer / scheduler / AMP / checkpointing flags
- eval contract:
  - dataset root
  - cache paths
  - full-eval batch knobs
  - Distinct5 test split

## 3) Success / failure interpretation rule

### Success as a valid probe

Treat the packet as a valid endpoint-metric probe only if all of the following
hold:

1. resolved config audit confirms:
   - `objective_mode = omf`
   - `w_flow = 1.0`
   - `terminal_swd_weight = 0.0`
2. the only semantic difference across arms is `loss_type`
3. training logs show a materially nonzero `flow` term
4. no hidden inheritance drift changes queue, OT, backbone, or eval protocol

If these conditions hold, then:

- a separation among `mse / huber / l1` is valid evidence about **endpoint
  pointwise matching kernels**
- a near-overlap among them is valid evidence of **practical parity on this
  isolated endpoint penalty**

### Failure / non-probing condition

Treat the packet as failed or non-probing if any of the following happens:

- `w_flow` resolves back to `0.0`
- `terminal_swd_weight` remains positive in the primary packet
- more than `loss_type` changes across arms
- the reported `flow` contribution is negligible relative to the total objective
- the run is later described as evidence for velocity-regression or global
  manifold-metric claims

## 4) Why this version actually probes the intended claim

The intended claim here is not "which kernel is better for velocity FM?" It is
closer to:

> what happens when target-style endpoint supervision is enforced through a
> pointwise Euclidean-family latent penalty?

This repaired packet actually probes that claim because:

- the compared kernel is applied directly to
  `pred_endpoint` vs `matched_target`
- the terminal SWD term is removed from the primary packet, so it cannot
  dominate the endpoint comparison
- the queue, OT plan, backbone, and eval protocol remain fixed

In short:

- the old packet was a near-null control because `loss_type` did not hit the
  active endpoint term
- the repaired packet makes `loss_type` the endpoint-matching term itself
- that is the correct object for the current endpoint-metric thesis

## Optional follow-up packet

After the primary isolated packet, a second "mixed-objective realism" packet may
restore `terminal_swd_weight = 20.0` while keeping `w_flow = 1.0`. That second
packet can answer a narrower ecological question:

> does the endpoint kernel still matter once SA-SWD is reintroduced?

But it should be reported as secondary, not as the clean theory-closing test.
