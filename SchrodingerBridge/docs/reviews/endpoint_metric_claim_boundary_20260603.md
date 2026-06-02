# Endpoint-Metric Claim Boundary

Date: 2026-06-03

Scope: claim boundary for the **repaired endpoint-metric trio** only. This memo
assumes the repaired packet described in
`docs/experiments/2026-06-03-flow-loss-metric-ablation/repaired_endpoint_metric_ablation_packet_20260603.md`:

- `objective_mode = omf`
- `w_flow > 0`
- `terminal_swd_weight = 0.0` in the primary isolated packet
- only `loss_type = mse / huber / l1` differs across arms

## What the repaired trio can support

The completed repaired trio now supports a **narrow endpoint-side negative
closure**:

- it compares `MSE`, `Huber`, and `L1` as **pointwise latent endpoint-matching
  kernels**
- it shows that these Euclidean-family endpoint penalties are practically
  distinct in the expected local way: repaired `MSE` is the most destructive on
  LPIPS, while repaired `Huber` and `L1` recover substantially better LPIPS
- it can support a bounded statement about **endpoint reconstruction kernels**
  under a frozen backbone, queue, OT plan, and eval protocol
- it supports a stronger negative conclusion that **pure endpoint-only
  pointwise supervision is not the source of the current reviewed H-mainline
  frontier**, because all three repaired arms remain materially worse on LPIPS
  than the reviewed H mainline with `W1`-style terminal matching

## What the repaired trio cannot support

Even if validly repaired, this trio still cannot support any of the following:

- not a proof that **all latent-space MSE** is bad
- not a proof that **all Euclidean latent metrics** are wrong
- not a proof that `Huber` or `L1` is universally better than `MSE`
- not evidence about the best **velocity-regression** kernel
- not evidence about the full **OT + SA-SWD + kinetic** composite objective
- not a theorem about latent-manifold geometry in general

It is an endpoint-objective ablation, not a global latent-metric verdict.

## Safe wording boundary

### `MSE / Huber / L1`

Safe:

- "In the repaired endpoint-objective ablation, `MSE`, `Huber`, and `L1` are
  compared only as pointwise latent endpoint-matching kernels."
- "This result supports a narrow statement about the behavior of endpoint
  Euclidean-family penalties on our fixed setup."
- "Within this isolated endpoint-only packet, `Huber` and `L1` are much less
  destructive than `MSE`, but none of the three matches the reviewed H
  mainline frontier."
- "The repaired trio closes in the negative direction: pure endpoint-only
  pointwise supervision is not the source of the current mainline gains."
- "The result does not generalize to all latent-space L2/MSE uses."

Unsafe:

- "`MSE` is wrong in latent space."
- "`Huber` or `L1` replaces the current mainline."
- "`Huber/L1` fixes latent geometry."
- "This ablation proves manifold-aware losses are necessary everywhere."

### `W1` terminal matching

Safe:

- "The repaired trio is isolated from the `W1`-style terminal term, so it does
  not by itself prove why SA-SWD terminal matching wins mechanistically in the
  full composite objective."
- "Current evidence for the paper's broader style-endpoint story still comes
  from the reviewed OT endpoint construction plus `W1`-style terminal matching
  mainline, and the repaired trio strengthens that story only indirectly by
  showing that endpoint-only pointwise supervision falls short."

Unsafe:

- "The repaired trio proves that `W1` terminal matching is superior."
- "The repaired trio closes the SA-SWD novelty claim."

### Latent-space metric claims

Safe:

- "The main risk is using pointwise Euclidean reconstruction as the
  **style-endpoint alignment objective**."
- "By contrast, some latent-space L2 terms remain natural, especially kinetic
  path-energy regularization and other same-chart local objectives."
- "Any claim beyond the endpoint-style object must be stated as open or
  separately validated."

Unsafe:

- "latent-space Euclidean metrics are broadly invalid"
- "all latent-space MSE terms should be replaced by `L1/Huber/W1`"

## One-sentence paper-safe takeaway

The repaired endpoint-metric trio can at most support a **narrow claim about
pointwise latent endpoint-matching kernels**, plus a negative closure that
endpoint-only pointwise supervision does not recover the current mainline
frontier; it still cannot, by itself, close a broader claim about latent-space
metrics, terminal `W1` matching, or manifold geometry in general.
