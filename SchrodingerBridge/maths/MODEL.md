# Code-Faithful Mathematical Model

Reviewed on `2026-05-16`.

This document is intentionally not minimal. Its job is to describe what the current code is actually doing, especially the `cross_attn` path, the skip path, and the loss branches.

## 1. State Space and Endpoint Map

The model works in latent space. Let

- `x in R^(C x H x W)` be the current latent state
- `s` be the target style id
- `z_1` be the predicted endpoint latent

The main public endpoint operator in `src/model.py` is

`endpoint_map(x, s) = x + v_theta(x, t=1, s) * horizon`

and the multi-step inference operator in `src/lancet_backbone.py` is

`h_(k+1) = h_k + Delta_k`

with

`Delta_k = step_size * step_scale * per_step * delta_theta(h_k, s)`

So the implementation is best viewed as a latent residual flow:

`z_1 = z_0 + integral_0^1 v_theta(z_t, t, s) dt`

approximated by one or a few learned residual steps.

## 2. Architecture Decomposition

The backbone has four important parts.

### 2.1 Content trunk

The content latent is lifted and processed into:

- a high-resolution feature path
- a body-resolution feature map at `16 x 16`
- a skip tensor at `32 x 32`

### 2.2 Style spatial prior

The model constructs a style-conditioned spatial map `style_map_proj` from style latents or cached style maps. This map is the main spatial style signal that later enters semantic cross-attention.

### 2.3 Body blocks = semantic cross-attention

This is the part that was missing from the earlier writeup.

The body blocks are instances of `SemanticCrossAttn` in `src/lancet_backbone.py`.

For content feature `x` and style map `y`, the block computes:

- normalized content features `nx`
- normalized style features `ns`
- queries from content
- keys and values from style

Formally,

`q = W_q IN(nx)`

`k = W_k IN(ns)`

`v = W_v ns`

The raw routing matrix is

`A_raw = q k^T / (sqrt(C) * temp)`

Then the routing mode chooses:

- `softmax` routing:
  `A = softmax(A_raw)`
- `sinkhorn` routing:
  `A = Sinkhorn(A_raw)`

The painted style feature is

`P = A v`

and the block returns either:

- pure painted feature if `paint_only = true`
- or a gated residual update

`x_out = x + gate_global * gate_local(x) * (1 + gamma) * P`

where:

- `gate_local(x)` is produced by `gate_conv`
- `gamma` is a learnable per-channel scaling tensor
- `gate_global` is the external scalar gate passed into the block

This matters mathematically. The cross-attention block is not just matching tokens. It is a style painting operator with learned spatial and channel gating.

### 2.4 Skip routing

After the body blocks, the network upsamples and fuses skip features.

If skip routing is enabled, the skip tensor is not simply concatenated. It is passed through `StyleRoutingSkip`, which modulates how much clean structure is retained versus re-expressed under style conditioning.

So there are two separate style injection routes:

1. body-level semantic cross-attention
2. decoder-level skip routing / skip fusion

This split is essential for interpreting the ablations.

## 3. What the Cross-Attention Statistics Mean

The code exports:

- `last_semantic_attn`
- `last_semantic_k`
- `semantic_attn_mean`
- `semantic_k_abs`
- `plan_entropy`

Interpretation:

- `last_semantic_attn` is the spatial routing matrix from content queries to style keys
- `last_semantic_k` is the normalized key bank derived from the style map
- `plan_entropy` measures how diffuse the routing is
- `semantic_k_abs` is a rough magnitude summary of style-key activation

Empirically, `semantic_attn_mean` is nearly constant in the destructive ablations and is not useful as a discriminator. `semantic_k_abs` and `plan_entropy` are more informative.

## 4. Loss Family in Code

There are two objective families in `src/losses.py`.

### 4.1 Flow-matching branch

This branch learns velocity targets from OT-matched bridge states.

It uses:

- sampled time `t`
- matched target latent from OT
- flow loss between predicted and target velocity
- optional path kinetic penalty

This branch is important historically, but it is no longer the main style-maximization path.

### 4.2 OMF branch

The active direct endpoint branch computes:

`L_total = L_kin + L_swd + L_low + L_color + L_nce + L_cycle + L_repulsive`

with each branch multiplied by its configured weight.

The main terms are:

`L_kin = w_kinetic * E ||v_theta||^2`

`L_swd = terminal_swd_weight * SWD(z_1, z_style)`

Optional branches:

- `w_low_freq`: low-frequency anchor term
- `w_color`: contextual local color term
- `w_nce`: latent patch NCE
- `w_cycle`: cycle consistency
- `w_repulsive`: anti-collapse repulsion

## 5. Two SWD Regimes

The code implements two different terminal SWD regimes.

### 5.1 Standard terminal SWD

If `swd_use_high_freq = false`, the endpoint is matched directly, optionally with semantic guidance from `semantic_k`.

This is the cleaner endpoint-matching regime and the one more closely aligned with the successful D0-style family.

### 5.2 High-frequency split regime

If `swd_use_high_freq = true`, the endpoint is split into:

- low-frequency component
- high-frequency component

Then:

- low-frequency content is anchored to a content-preserving but target-colored reference
- SWD is applied to the high-frequency branch

This creates a much more structured bias, but the ablations show that pushing too hard into micro high-frequency matching becomes a trap.

## 6. Kinetic Modes

There are at least two kinetic modes in code:

- `path`
- `time_gated`

For `time_gated`, the code uses

`gate(t) = t^alpha`

and computes

`L_kin = E[ gate(t) * ||v||^2 ]`

So time-gated kinetic does not change the target style directly. It changes where in normalized time the path is allowed to move more freely.

## 7. Why Step Count and Step Size Are Weak Levers in the Current Regime

The code supports multi-step integration, but the existing sweeps show:

- changing step size around the current baseline hardly changes final metrics
- increasing step count from `1` to `16` hardly changes final metrics

So the current model behaves much more like a learned endpoint corrector than like a delicate ODE solver that needs better numerical resolution.

Mathematically, that means:

- the dominant error is model bias in `delta_theta`
- not discretization error from too few Euler steps

This is why residual amplitude can matter a lot while step count does not.

## 8. What the Model Is Really Optimizing

The most faithful compact expression for the active branch is:

`min_theta w_swd * SWD(z_1, Z_style) + w_kin * E||v||^2 + sum_j w_j L_j`

But the data says the effective behavior is dominated by three structural mechanisms:

1. semantic cross-attention paints style into the body features
2. skip routing preserves or leaks content structure
3. endpoint SWD decides whether the final latent actually lands near the target style distribution

So the model is not "just a bridge". It is a bridge with:

- a painted semantic body
- a routed skip highway
- an endpoint distribution matcher

That is the right object to analyze mathematically.
