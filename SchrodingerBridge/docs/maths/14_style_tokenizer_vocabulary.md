# Style Tokenizer Vocabulary

Date: 2026-05-27

## Claim

The next style-conditioning interface should not be a single anonymous
`style_code`. It should be a small vocabulary of named fields. The backbone is
trained first to read those fields. Only after that should we freeze the
backbone and refine the vocabulary.

This avoids the failure mode seen in `m02`: a 160-dimensional table can have
low effective rank and still look numerically wide. The model should not be
asked to discover style identity, style grammar, frequency allocation, spatial
prior, and style strength inside one centroid-prone vector.

## Interface

The new vocabulary is:

```text
style_id -> StyleTokens

StyleTokens =
{
  identity,     # which target domain
  grammar,      # how this style acts
  band,         # low/mid/high carrier allocation
  spatial,      # weak spatial prior
  residual_code # legacy-compatible code path
}
```

The first backbone-training implementation keeps legacy compatibility in the
minimal sense that the old `style_emb` remains the base code:

```text
style_code = legacy_style_emb
```

The carrier reads `grammar` and `band` directly. During tokenizer-backbone
training, both anonymous shortcuts are disabled:

- `style_token_project_code = false`
- `style_blender_texton_use_style_code = false`

The optional projection back into `style_code`, and the legacy
`style_code -> texton modulation` generator, should be revisited only in the
vocabulary-refinement phase after the backbone already shows field-specific
response.

The first remote attempt also exposed a diagnostic trap: all forward
activations were finite, and the apparent `finite_ratio=0.99999994` in a large
gradient tensor was caused by float32 reduction precision when averaging an
all-True boolean mask. Gradient checks must use `finite.all()` first and only
compute a ratio after the all-finite check fails. We do not use gradient
sanitation for this route.

## Field Semantics

The current grammar layout is deliberately small:

```text
grammar = [
  palette_strength,
  flatness_strength,
  contour_strength,
  contour_width,
  shadow_simplify,
  mid_texton_strength,
  high_texture_strength,
  highfreq_suppression,
  transport_softness
]
```

`band` controls the low/mid/high carrier gains. It is independent from style
identity so two styles can share an identity distance while still using very
different physical actuators.

Expected field profiles:

```text
Hayao:
  high palette, high flatness, high contour, high highfreq_suppression,
  low high_texture

Van Gogh:
  high mid_texton, high high_texture, moderate palette,
  low highfreq_suppression
```

## Backbone-First Training

The training order should be:

1. Train a main backbone with tokenizer fields enabled.
2. Inspect `numeric_debug.jsonl` by target style:
   - `style_token_grammar`
   - `style_token_band_gains`
   - `body_transport_texton_band_alloc`
   - `body_transport_texton_flatten_delta`
   - low/mid/high texton deltas
3. Only if the backbone shows field-specific response, freeze it and refine the
   vocabulary.

The vocabulary-only phase is useful only after the backbone has learned the
field semantics. Otherwise the vocabulary is just another post-hoc style table
and will collapse like the previous adapter.

## First Structural Probe

Implemented variants:

```text
ema_style_vocab_texton_w34
ema_style_vocab_hayao_w36
```

These keep the guarded `transport_texton` carrier but add:

- factorized identity / grammar / band vocabulary;
- direct band-gain consumption by the texton carrier;
- grammar-driven flat-region high-frequency suppression;
- deterministic mean path for standard evaluation.

The point is not to add a larger style vector. The point is to make the carrier
read fields with explicit meaning.

## Acceptance Criteria

A useful tokenizer backbone must satisfy:

- global `clip_style` moves toward or beyond `0.72`;
- `content_lpips` remains near or below `0.50`;
- Hayao cross-style `clip_style` improves, not just the average;
- by-target debug shows Hayao using stronger flatness/band suppression than
  Van Gogh;
- flatness suppression reduces broken high-frequency fragments instead of
  erasing content edges;
- vocabulary-only calibration after the backbone improves or preserves the
  deterministic mean score.

## Seedream Diagnostic Protocol

Seedream 4.5 is a visual reference, not a teacher loss. It should answer:

```text
Which visual operation does the strong model perform that our tokenizer
backbone does not yet express?
```

For Hayao, inspect these axes first:

```text
texture_fragmentation_gap
  high value => our output has broken high-frequency fragments relative to
  Seedream.

flatness_deficit
  high value => our output does not form clean animation-like color planes.

edge_alignment_deficit
  high value => stylized changes are not phase-locked to source boundaries.

palette_shift_gap
  high value => broad color statistics drift differently from the reference.
```

Implementation:

```text
tools/experiments/diagnose_seedream_gap.py --focus-target Hayao
```

The script writes:

```text
seedream_gap_image_metrics.csv
seedream_gap_summary.csv
seedream_gap_readout.md
seedream_gap_worst_cases.png
```

Use the readout to decide which vocabulary field is under-expressed:

| dominant gap | tokenizer interpretation | model response |
|---|---|---|
| texture fragmentation | highfreq suppression not being read | increase grammar-driven flatten branch, not generic SWD |
| flatness deficit | color-plane grammar missing | strengthen low carrier and flat-region repaint |
| edge deficit | contour grammar missing | add/strengthen edge-aligned contour branch |
| palette gap | palette field missing | add low-frequency palette/moment token |

This is the correct role for Seedream: it provides a diagnosis of missing visual
operators. It should not define the main unsupervised training objective.

## Vocabulary Refinement Phase

After a good backbone exists:

```text
freeze backbone
optimize grammar_vocab, band_vocab, style_spatial_id_16, optional residual code
```

The optimizer should report:

- vocabulary field norms and margins;
- response separation on the same content batch;
- Hayao cross-style metrics;
- sampled or reference-conditioned modes only as diagnostics, not main scores.

Seedream remains a diagnostic reference for understanding the desired visual
grammar. It should not define the main training target unless a run is clearly
labeled as an external-teacher side experiment.
