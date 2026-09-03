# Semantic Moment OT

Date: 2026-05-27

## Problem

The current frontier is capped around `clip_style = 0.714`.

Evidence:

- full-train `style_emb` fitting is negative, even when applied to the current
  body-transport anchor;
- phase-envelope and edge-phase losses reduce unsafe texture but do not raise
  style;
- transport-conditioned AdaIN can move channel statistics, but a style-push
  version reaches only `0.7143 / 0.5295`.

This argues that the remaining bottleneck is not raw style pressure, global
style-vector capacity, or one more body residual gate. The terminal OT target is
still too spatially literal for an unpaired transfer task.

## Hypothesis

Patch SWD on unpaired style images can reward two different things:

1. style statistics: palette, contrast, and brush-energy envelopes;
2. target-image phase: the exact high-frequency sign/location pattern of a
   different painting.

Only the first should be transferred. The second produces the observed
anti-phase high-pass haze and fractured texture. Seedream's diagnostic role is
useful here: it has large coherent regional repainting, not stronger random
high-pass energy.

## Objective

For each predicted endpoint and target style latent:

1. Partition the predicted endpoint by quantiles of the source-content low-band
   guide.
2. Partition the target style latent by quantiles of its own low-band guide.
3. For corresponding quantile ranks, match:
   - low-band channel mean and std;
   - local high-pass envelope mean and std.

The target style image contributes region statistics, not pixel phase. This is
still fully unsupervised and uses no Seedream teacher.

## Probe

Code mode:

```text
terminal_swd_mode = "semantic_moment"
```

Remote variants:

| variant | intent |
|---|---|
| `ema_semantic_moment_adain_w30_guard` | conservative region-statistic OT on the best content-safe AdaIN carrier |
| `ema_semantic_moment_adain_w38_style` | style-push version; acceptable only if style moves toward `0.72` |

Success criterion:

- primary: `clip_style > 0.72`;
- acceptable tradeoff: LPIPS may move toward `0.49-0.50`;
- reject if style remains near `0.714`, because that means region statistics
  alone are not enough and the missing actuator is deeper than the terminal OT
  target.

## Result

Remote run:

```text
exp/vae_backend/ema_semantic_moment
```

| variant | epoch | clip_style | content_lpips | EC | verdict |
|---|---:|---:|---:|---:|---|
| `ema_semantic_moment_adain_w30_guard` | 6 | 0.71325 | 0.49158 | 0.36263 | content-safe, no style lift |
| `ema_semantic_moment_adain_w30_guard` | 7 | 0.71088 | 0.49955 | 0.35575 | worse |
| `ema_semantic_moment_adain_w30_guard` | 8 | 0.71210 | 0.49768 | 0.35770 | no recovery |
| `ema_semantic_moment_adain_w38_style` | 6 | 0.71441 | 0.53082 | 0.33518 | tiny style lift, content failure |
| `ema_semantic_moment_adain_w38_style` | 7 | 0.71122 | 0.53777 | 0.32875 | worse |
| `ema_semantic_moment_adain_w38_style` | 8 | 0.71228 | 0.53623 | 0.33033 | no recovery |

Conclusion:

Semantic moment OT is a useful diagnostic but not the missing mechanism. It
separates region statistics from unpaired phase more cleanly than raw terminal
SWD, and the guard branch keeps LPIPS under `0.50`, but the style ceiling stays
near `0.713`. The style-push branch reaches only `0.71441` while LPIPS degrades
to `0.53082`.

This rejects the hypothesis that the remaining gap is mainly a terminal OT
target-design problem. Combined with the full-training-set `style_emb` negative
result, the bottleneck is now localized to the **style actuator**: the current
global embedding and body carrier can preserve content or add weak local
statistics, but they cannot produce the organized region-level repainting that
Seedream shows without damaging the transfer geometry.
