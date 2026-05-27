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
as a clean Hayao operator. Before changing the backbone again, the tokenizer
itself should be treated as a component with explicit capacity, coverage, and
sensitivity metrics.

The first clean no-prior run, `ema_style_vocab_neutral_w34`, confirms the same
mechanism without manual style weighting:

- epoch 8 global `clip_style=0.7078`, `content_lpips=0.5149`;
- Hayao cross `clip_style=0.6432`, `content_lpips=0.5664`;
- grammar normalized range is `3.443`;
- Hayao grammar and band vectors become nonzero without manual priors;
- Hayao flattening is activated, but visible style remains weak.

Therefore the next move is not a larger class embedding. It is a backbone
operator diagnosis deferred until the vocabulary is understood. The tokenizer
phase should first ask:

```text
capacity: does the vocabulary have enough effective rank?
coverage: does every non-photo style occupy non-neutral fields?
sensitivity: does changing a field move the carrier in a measurable direction?
refitability: with the backbone frozen, can vocabulary-only optimization improve
             style without harming LPIPS?
```

If those tests fail, the problem is tokenizer design/training. If they pass and
Hayao remains weak, then the missing piece is a backbone operator that consumes
the learned fields.

## Component Metrics

The tokenizer is now judged as a component before another backbone change:

| metric | question | failure signal |
|---|---|---|
| effective rank | does the vocabulary use its nominal dimensions? | low rank in grammar or band |
| field coverage | do all non-photo styles leave neutral coordinates? | only Hayao/Cezanne active |
| sensitivity | do fields move carrier deltas? | collapsed `body_transport_texton_*` deltas |
| downstream gate | does refitting improve style without LPIPS drift? | style gain below noise or LPIPS worsens |

The current clean tokenizer fails mainly on coverage. `neutral_w34` and
`neutral_w36_stylepush` both activate grammar/band only for Hayao and Cezanne,
while Monet and Van Gogh remain near zero grammar. That means the tokenizer is
not yet a full style vocabulary even though its Hayao field is nonzero.

The next empirical step is vocabulary-only refinement with a fixed backbone:

```text
freeze backbone
freeze style_emb
freeze style_spatial_id_16
optimize grammar_vocab and band_vocab only
export style_adapter.pt
evaluate with the same full protocol
```

If this raises coverage and style, the tokenizer was under-trained. If coverage
still stays at two active styles, the field parameterization or objective is
wrong.

## Vocabulary-Only Refit Finding

The frozen-backbone vocabulary refit answered the refitability question:

| recipe | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|
| `m10_token_vocab_swd_anchor` | 0.710066 | 0.466699 | 0.618145 | 0.517782 |
| `m11_token_vocab_stylepush` | 0.710138 | 0.466697 | 0.618121 | 0.517815 |

This is not enough to claim tokenizer progress. The adapter scorecard still
shows only two active non-photo grammar/band rows, and a direct tensor diff
shows that `grammar_vocab` did not change at all. The only meaningful adapter
movement was in `band_vocab`, mostly for Cezanne.

The gradient audit explains the failure mode:

```text
grammar gradient:
  Hayao: tiny but nonzero
  Monet / Van Gogh / Cezanne: zero under the current objective

band gradient:
  nonzero for all target styles, but too small and too low-dimensional
```

Interpretation: the existing grammar coordinates are mostly non-executable in
the current network. They can be logged and diagnosed, but the training
objective cannot use them as a strong handle. This means "increase tokenizer
loss" is the wrong next move.

## Tokenizer Projector Hypothesis

The next tokenizer-only change is to let named fields produce a residual
style-code delta:

```text
base style_emb
+ code_projector(identity, grammar, band)
-> style_code consumed by the existing style-conditioned blocks
```

This keeps the backbone weights fixed but changes the tokenizer from a passive
side table into an executable controller. It tests whether the current backbone
already has unused style capacity that the old tokenizer could not address.

The adapter format now stores:

```text
style_tokenizer.project_code
style_tokenizer.code_projector.*
style_tokenizer.grammar_vocab.weight
style_tokenizer.band_vocab.weight
```

Success criteria for this route:

- the projector route must improve global style beyond `0.710` without losing
  the good LPIPS band near `0.47`;
- Hayao cross-target style must rise materially, otherwise the projector only
  adds generic style pressure;
- the component scorecard should show either higher coverage/rank or a clear
  downstream gain;
- if projector training helps, tokenizer capacity was the bottleneck; if it
  fails, the missing piece is an explicit flat-plane / contour operator rather
  than another vocabulary optimizer.

Result:

| recipe | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|
| `m12_token_projector_swd_anchor` | 0.709745 | 0.430403 | 0.614650 | 0.482358 |
| `m13_token_projector_stylepush` | 0.709595 | 0.434844 | 0.622817 | 0.488738 |

The route is therefore negative for style. It gives excellent LPIPS, but does
not move global style above the vocabulary-only `~0.710` band. The tokenizer
scorecard still reports only two active grammar/band non-photo rows, and
metric-space diagnosis shows weak token/data distance alignment. This supports
the hard-binding revision below.

## Operator-Bound Tokenizer Revision

The projector route is deliberately not the final representation. It still
mixes fields through a learned projection and can hide tokenizer collapse
inside an anonymous `style_code`. The stricter representation-learning route is
now:

```text
identity -> 1x1 pointwise color/channel operator
grammar  -> depthwise 3x3 spatial operator
band     -> direct low/mid/high residual energy gains
```

This is exposed as:

```json
{
  "model": {
    "style_tokenizer_enable": true,
    "style_token_grammar_dim": 32,
    "dynamic_style_operator_head": true,
    "dynamic_style_operator_mode": "factorized_token"
  }
}
```

The corresponding probe variants are:

```text
ema_style_vocab_factorized_w36
ema_style_vocab_factorized_w40_stylepush
```

These runs test a specific hypothesis: the old tokenizer failed because its
named fields had no enforced operator meaning. If factorized binding improves
field gradients, token/data metric correlation, and Hayao visual cleanliness,
the tokenizer problem was a representation-to-operator mismatch. If it does
not, the next missing component is likely a dedicated flat-plane / contour
operator rather than a larger embedding table.

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
