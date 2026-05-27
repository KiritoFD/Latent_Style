# Transport-Confidence Body Gate

Date: 2026-05-27

## Problem

The current EMA frontier is not evidence that the VAE is unusable. It is evidence that our style actuator is still poorly placed and poorly gated.

Recent negative probes are informative:

- full-train post-hoc `style_emb` tuning lowered style, so the missing lever is not a stronger global style code;
- `region_paint` kept the bodyblend style entrance but learned an almost constant region gate, so an unconstrained MLP does not discover object routing from terminal SWD alone;
- the best `region_paint` row was only `clip_style=0.71308`, `LPIPS=0.54967`, worse than the balanced EMA frontier.

Seedream's diagnostic role is to show the qualitative target: style changes are region-organized and phase-locked to objects. It must not become a teacher for the main method.

## Hypothesis

The semantic cross-attention already computes a transport plan:

```text
A_ij = P(style token j is the style match for content token i)
```

The correct body actuator should use this plan as a physical confidence field. A token should receive body-level style residual only when:

1. its attention row is sharp;
2. its selected style token is not over-subscribed by many unrelated content tokens;
3. the content location has enough structural support to absorb a residual without turning into color fog.

This replaces a free learned gate:

```text
gate = MLP(style_id, lowfreq_content_bin)
```

with a deterministic transport-confidence gate:

```text
row_conf_i = (top1(A_i) - top2(A_i)) * (1 - H(A_i) / log N)
load_j     = sum_i A_ij
uniq_i     = 1 / sqrt(E_{j~A_i}[load_j])
T_i        = normalize(row_conf_i * uniq_i)
G_i        = floor + (1 - floor) * (1 - exp(-gamma * T_i))
```

The low-frequency body residual is then gated by `G_i` and the content support gate. Mid/high residuals additionally keep the phase/support gates already used by the dual residual branch.

## Expected Signal

If the hypothesis is right:

- `body_transport_gate` should have real spatial variation, unlike `body_region_gate`;
- LPIPS should recover relative to `region_paint`;
- style should remain near or above bodyblend/bodyregion because the body-level style entrance is still active.

If it fails:

- a constant `body_transport_gate` means the semantic attention itself is too degenerate to support routing;
- good LPIPS but low style means support gating is too conservative or the body residual is the wrong carrier;
- high style and bad LPIPS means body transport confidence alone is insufficient and needs a stronger content-boundary penalty.

This is a backbone-routing test, not a scalar loss sweep.
