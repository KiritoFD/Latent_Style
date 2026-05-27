# Style Type as Operator Coordinates

## Thesis

A style type is not a class ID and not a single anonymous embedding vector.
For LANCET, a useful style representation must be a compact policy over
executable image operators:

```text
style s -> operator coordinates tau(s)
content latent z + tau(s) -> carrier gates, frequency allocation, warp/color actions
```

The tokenizer should therefore learn coordinates that the backbone can execute,
not hidden labels that only help a classifier identify the target domain.

## Required Coordinates

The current tokenizer split into `identity / grammar / band / residual` is the
right first abstraction, but the semantics must be made explicit during
diagnosis:

| coordinate | visual meaning | expected actuator |
|---|---|---|
| palette_strength | hue/value/saturation statistics | low-frequency color flow |
| flatness_strength | size and cleanliness of color planes | flat-region repaint / highpass suppression |
| contour_strength | line prominence and edge cleanliness | edge-locked contour branch |
| contour_width | width of stylized contour bands | contour dilation / edge support |
| shadow_simplify | removal of photo-like micro shading | low-frequency plane simplification |
| mid_texton_strength | brush / local covariance density | mid-band texton carrier |
| high_texture_strength | fine stroke/noise energy | high-band amplitude carrier |
| highfreq_suppression | how much texture should be removed | anti-fragment high-frequency gate |
| transport_softness | how hard semantic transport may move mass | semantic routing / transport gate |

These names are diagnostic coordinates, not initialization priors. The values
must be learned from data under balanced style exposure.

## Why Hayao Is The Critical Slice

Hayao is visually different from texture-heavy styles. It is not "more style"
in the same direction as Van Gogh or Cezanne. The desired operator is closer to:

```text
large flat color planes
+ clean contour preservation / reinforcement
+ suppressed high-frequency fragments
+ restrained local texton injection
```

This explains the current failure mode: a texton/SWD style carrier can raise
global style while still making Hayao broken, because it treats Hayao as if it
needed extra local texture. Manual Hayao sampling weights would hide the
diagnosis. Hayao must remain a reporting slice.

## Diagnostic Contract

Every tokenizer run must answer three questions before it is treated as
progress:

1. Does the vocabulary separate styles?
   - `style_token_grammar` should show non-trivial range across target styles.
   - `style_token_band_gains` should not stay at the same low/mid/high policy
     for every target.
2. Does the backbone read those fields?
   - `body_transport_texton_*_delta` and related actuator debug values should
     vary by target in the expected direction.
3. Does the executable direction match the style?
   - Hayao should show stronger flattening / lower high texture than the
     texture styles.
   - If Hayao fields separate but Hayao score and grid stay poor, the missing
     part is the operator, not the token value.

The diagnostic summary is:

```powershell
python tools\experiments\summarize_style_tokenizer_debug.py `
  exp\vae_backend_256_probe\<run_name> --limit-events 80
```

It writes:

```text
style_tokenizer_debug_readout.md
style_tokenizer_debug_by_style.csv
style_tokenizer_field_discrimination.csv
style_tokenizer_eval_overview.csv
style_tokenizer_checkpoint_vocab.csv
```

## Current Evidence From Earlier Tokenizer Probe

On `ema_style_vocab_texton_w34`, the debug readout shows:

- global eval around `clip_style=0.7084`, `content_lpips=0.5144`;
- Hayao cross eval around `clip_style=0.6429`, `content_lpips=0.5659`;
- grammar does separate styles;
- Hayao activates flattening more than the other styles;
- but Hayao score remains low.

Interpretation: the first tokenizer did not merely fail to create a Hayao
field. It created a weak field that the current texton carrier cannot execute
as a clean Hayao operator. The next architecture step should add or expose a
macro flat-color / contour actuator, then revisit vocabulary-only refinement.

The first clean no-prior run, `ema_style_vocab_neutral_w34`, confirms the same
mechanism without manual style weighting:

- epoch 8 global `clip_style=0.7078`, `content_lpips=0.5149`;
- Hayao cross `clip_style=0.6432`, `content_lpips=0.5664`;
- grammar normalized range is `3.443`;
- Hayao grammar and band vectors become nonzero without manual priors;
- Hayao flattening is activated, but visible style remains weak.

Therefore the next move is not a larger class embedding. It is a backbone
operator revision: add a controllable flat-plane / contour-preserving branch
and then test whether the already-learned Hayao fields can drive it.

## Spiral Protocol

The intended loop is:

```text
neutral tokenizer backbone
-> field/debug diagnosis
-> freeze backbone, refine vocabulary
-> identify missing executable operators
-> revise backbone
-> repeat
```

Success is not a single average score. The run must satisfy:

- global `clip_style > 0.72`;
- `content_lpips` preferably below `0.50`;
- Hayao cross-target score rises materially;
- Hayao grids look like clean animation-like planes and contours rather than
  fragmented texture;
- the tokenizer readout shows that this came from learned coordinates, not
  manual style weighting.
