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

### MSE Controls Completed 2026-05-28

The missing control was run on the remote 3060 with the same current
launcher/evaluator:

```text
out root:
I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_mse_controls

variants:
mse_plain4_w20_anchor
mse_dynamic_guard_w28
mse_transport_texton_w34_guard
mse_bodyblend_w28_guard
mse_guard_w20_lowwarp
```

Each MSE variant is cloned from its EMA counterpart with only:

```text
vae_model = mse
latent_root = latent-256
```

All architecture, loss, schedule, and eval settings are inherited from the EMA
variant. The launcher was also fixed so `--skip-existing-latents` accepts an
existing `.pt` latent tree even when `manifest.json` is absent.

Full 6/7/8 results:

| variant | backend | epoch | clip_style | content_lpips | EC |
|---|---|---:|---:|---:|---:|
| `mse_plain4_w20_anchor` | MSE | 6 | `0.703597` | `0.419917` | `0.408145` |
| `mse_plain4_w20_anchor` | MSE | 7 | `0.702931` | `0.422557` | `0.405903` |
| `mse_plain4_w20_anchor` | MSE | 8 | `0.703076` | `0.423843` | `0.405082` |
| `mse_dynamic_guard_w28` | MSE | 6 | `0.710273` | `0.446022` | `0.393476` |
| `mse_dynamic_guard_w28` | MSE | 7 | `0.708131` | `0.464010` | `0.379551` |
| `mse_dynamic_guard_w28` | MSE | 8 | `0.709669` | `0.459310` | `0.383711` |
| `mse_transport_texton_w34_guard` | MSE | 6 | `0.718588` | `0.483008` | `0.371505` |
| `mse_transport_texton_w34_guard` | MSE | 7 | `0.715380` | `0.491912` | `0.363477` |
| `mse_transport_texton_w34_guard` | MSE | 8 | `0.716346` | `0.489790` | `0.365487` |
| `mse_bodyblend_w28_guard` | MSE | 6 | `0.715295` | `0.485741` | `0.367846` |
| `mse_bodyblend_w28_guard` | MSE | 7 | `0.712734` | `0.495505` | `0.359571` |
| `mse_bodyblend_w28_guard` | MSE | 8 | `0.713599` | `0.493060` | `0.361751` |
| `mse_guard_w20_lowwarp` | MSE | 6 | `0.722365` | `0.544632` | `0.328942` |
| `mse_guard_w20_lowwarp` | MSE | 7 | `0.725233` | `0.553443` | `0.323858` |
| `mse_guard_w20_lowwarp` | MSE | 8 | `0.723961` | `0.550776` | `0.325221` |

Matched best-row comparison:

| family | EMA clip | EMA LPIPS | MSE best epoch | MSE clip | MSE LPIPS | delta clip | delta LPIPS | readout |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `plain4_w20_anchor` | `0.700700` | `0.421500` | 6 | `0.703597` | `0.419917` | `+0.002897` | `-0.001583` | small gain |
| `dynamic_guard_w28` | `0.707800` | `0.447700` | 6 | `0.710273` | `0.446022` | `+0.002473` | `-0.001678` | small gain |
| `transport_texton_w34_guard` | `0.714510` | `0.482610` | 6 | `0.718588` | `0.483008` | `+0.004078` | `+0.000398` | small gain |
| `bodyblend_w28_guard` | `0.715800` | `0.497200` | 6 | `0.715295` | `0.485741` | `-0.000505` | `-0.011459` | content gain, no style gain |
| `guard_w20_lowwarp` | `0.724500` | `0.552600` | 7 | `0.725233` | `0.553443` | `+0.000733` | `+0.000843` | high style, over LPIPS budget |

Artifacts:

```text
CSV:
SchrodingerBridge/exp/vae_backend_256_mse_controls/vae_backend_256_results.csv

Matched comparison:
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/mse_backend_matched_comparison.csv
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/mse_backend_matched_comparison.md

First-image grids:
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/grids/
```

Interpretation:

```text
plain4:  +0.0029 clip, -0.0016 LPIPS
dynamic: +0.0025 clip, -0.0017 LPIPS
texton:  +0.0041 clip, +0.0004 LPIPS
```

The texton carrier is the most useful MSE-positive row: it reaches
`0.718588 / 0.483008`, very close to the target while staying content-safe.
However, it still does not cross `0.72`, let alone `0.73`. The lowwarp row
proves MSE can carry high style (`0.725233`) but it remains structurally
over-budget (`LPIPS=0.553443`).

Conclusion: MSE is not globally worse, and the old claim that MSE is unusable
is too strong. Current evidence says MSE is a modestly useful diagnostic/control
backend for texton transport, not a sufficient replacement for the EMA mainline.
The next productive move is still operator/tokenizer design: recover the
lowwarp style gain without its LPIPS drift, or add a style branch that lifts the
texton carrier from `0.7186` to `>0.72` while keeping LPIPS below `0.50`.

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
