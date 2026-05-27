# Post-VAE EMA Verdict

Date: 2026-05-27

## Question

Is `sd-vae-ft-ema` actually unusable for the 256 workflow, or is the current
EMA line under-designed / under-tuned?

## Short Answer

EMA is not fundamentally unusable. The evidence says the current EMA route has
real style capacity, but the content-safe delivery mechanism is still wrong.

Do not use pre-VAE-switch SD15/MSE numbers as proof for the current EMA route.
They show the old system had capacity, but they are not a clean A/B against the
post-VAE EMA/KL-f4/SDXL experiments.

Within the post-VAE evidence currently available:

- SDXL is stable but low-style: best useful rows are around `clip_style ~0.667`
  with very low LPIPS, and style-push variants damage LPIPS without recovering
  style.
- KL-f4 is not yet competitive: current fair rows are around
  `clip_style ~0.654`, `content_lpips ~0.485`.
- EMA is the only post-VAE backend with evidence of reaching the target style
  band: `ema_guard_w20_lowwarp e7` reached `clip_style=0.7245`, but
  `content_lpips=0.5526`.
- EMA also has content-safe frontier points:
  `ema_transport_texton_w34_guard e6` reached `0.7145 / 0.4826`, and
  `ema_bodyblend_w28_guard e6` reached `0.7158 / 0.4972`.

So the correct diagnosis is:

```text
EMA latent is viable.
The current EMA carrier/tokenizer still fails to deliver enough visible style
under the LPIPS budget.
```

## What Would Prove EMA Is Fundamentally Bad?

EMA would look fundamentally bad if all content-safe variants stayed below
`clip_style ~0.70`, or if every style gain required catastrophic structure
loss. That is not what the results show.

Instead, EMA has two separate regimes:

| regime | observed result | interpretation |
|---|---:|---|
| high style pressure | `0.7245 / 0.5526` | enough style capacity exists |
| content-safe carrier | `0.7145 / 0.4826`, `0.7158 / 0.4972` | good geometry, insufficient style delivery |

This is a frontier-shaping problem, not a dead-backend problem.

## Per-Style Evidence

The best content-safe EMA carrier is not uniformly weak. For
`ema_transport_texton_w34_guard e6`, target-style means were:

| target | clip_style | content_lpips |
|---|---:|---:|
| Van Gogh | `0.7604` | `0.4878` |
| Cezanne | `0.7294` | `0.4722` |
| Monet | `0.7288` | `0.4565` |
| photo | `0.6910` | `0.4836` |
| Hayao | `0.6630` | `0.5129` |

This is decisive: EMA can express several target domains well. The real failure
mode is Hayao and, more generally, flat / contour / low-texture animation-like
style. A global VAE failure would not produce strong Van Gogh, Cezanne, and
Monet rows while Hayao alone remains low.

## MSE Comparison Status

There is no clean post-VAE CSV row that directly compares `sd-vae-ft-mse` and
`sd-vae-ft-ema` under the same current backend, same configs, same evaluator,
and same eval path. Therefore we should not claim a scientific EMA-vs-MSE
winner from old t00/t01 numbers.

Working position:

- EMA is theoretically preferred because it avoids the MSE-smoothed latent
  prior.
- The old MSE-oriented architecture may still be better matched to the old MSE
  latent statistics.
- If MSE is kept, it should be a control backend, not the explanation for the
  new mainline.

## Tokenizer Finding

The first EMA tokenizer probe did run, but it did not beat the better EMA
carriers:

| run | best observed summary |
|---|---:|
| `ema_style_vocab_texton_w34` | `clip_style=0.7084`, `content_lpips=0.5144` |
| `ema_style_vocab_hayao_w36` | `clip_style=0.7068`, `content_lpips=0.5445` |

Debug traces show the tokenizer fields are present, but the effective carrier
response is too small and too generic. The high-frequency delta remains tiny,
and the Hayao flattening path is not enough to create clean color planes. This
means the idea is not disproved, but the first readout design is weak.

Important correction: these first tokenizer runs used manual per-style
grammar/band priors. They are therefore diagnostics, not clean evidence about a
learned tokenizer vocabulary. The mainline tokenizer must start with neutral
zero grammar/band fields and a differentiable zero point, then learn the style
fields from data through the backbone/vocabulary spiral.

## Model Implication

Do not spend the next round on scalar sweeps alone. EMA needs a style operator
change:

1. Keep EMA as the main post-VAE backend.
2. Use `ema_transport_texton_w34_guard` or `ema_bodyblend_w28_guard` as the
   content-safe anchor.
3. Treat Hayao as the diagnostic style, not as a sampling-weight problem first.
4. Add an explicit macro style branch for flat color planes and edge-locked
   contours.
5. Keep the texton branch for Monet/Cezanne/Van Gogh, but do not ask it to solve
   Hayao's low-texture grammar.
6. Use Seedream only as a diagnostic reference for the missing visual operator,
   not as the main training teacher.

## Next Testable Hypothesis

The next architecture hypothesis should be:

```text
Hayao fails because the current carrier is mostly a texton / moment transporter.
It can move texture distributions, but it does not create large coherent flat
regions with phase-locked contours.
```

A useful follow-up should therefore compare:

- content-safe anchor carrier;
- anchor + macro flat-color branch;
- anchor + edge-contour branch;
- anchor + both branches.

Acceptance gate:

```text
global clip_style > 0.72
content_lpips <= 0.50 preferred, <= 0.52 acceptable for a style-push probe
Hayao target clip_style must rise, not only the global average
```
