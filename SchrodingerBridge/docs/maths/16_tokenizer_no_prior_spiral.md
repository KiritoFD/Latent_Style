# No-Prior Style Tokenizer Spiral

Date: 2026-05-27

## Position

The style tokenizer should be a learnable coordinate system, not a hand-coded
style table.

The previous tokenizer experiment was useful as a diagnostic, but it mixed in
manual style priors. That is not acceptable for the main route because it
answers the wrong question. We need to know whether the model can discover a
good style representation, especially for Hayao, under the native objective.

## Correct Abstraction

The tokenizer has two levels:

```text
field schema: fixed by model design
field values: learned from data
```

The schema can say that styles may differ by:

- identity;
- palette / low-frequency carrier;
- flat color-plane response;
- contour / edge alignment;
- mid-frequency texton;
- high-frequency texture;
- transport softness or spatial support.

The schema must not say that a particular style owns any of those properties at
initialization time. Hayao being flat and contour-driven should emerge as a
learned vocabulary state, then be verified through by-style debug and images.

## Code Correction

The current correction is deliberately small:

- `StyleTokenizer.grammar_vocab` starts at zero for every style.
- `StyleTokenizer.band_vocab` starts at zero for every style.
- Therefore all styles start with the same neutral grammar and band gains.
- The only non-anonymous initial distinction is the fixed simplex identity
  code, which identifies the style class but carries no semantic claim about
  what the style should do.

The flatten branch is changed from a positive ReLU gate to a signed response.
This matters because the old neutral zero point could be a dead gate:

```text
old: relu(tanh(grammar)) -> zero initialization has no useful negative side
new: tanh(grammar)       -> zero initialization is neutral but trainable
```

Positive learned values suppress high-frequency fragments in smooth regions.
Negative learned values can preserve or amplify local texture. This lets the
same field separate Hayao-like flat planes from Van-Gogh-like texton energy
without giving either style a manual head start.

## Spiral Training Protocol

The tokenizer should be developed in alternating phases.

### Phase A: Backbone Learns To Read Fields

Train the backbone with neutral tokenizer fields enabled. The target is not an
immediate score jump. The target is to prove that field changes affect the
right actuator:

- `style_token_grammar` differs by target style after training;
- `style_token_band_gains` differs by target style after training;
- Hayao cross-style rows produce a distinct flat / low-carrier response;
- texture styles produce distinct mid/high texton response;
- field deltas are large enough to matter visually.

### Phase B: Vocabulary-Only Refinement

Freeze the backbone and optimize only vocabulary-like parameters:

- `grammar_vocab`;
- `band_vocab`;
- optional `style_spatial_id_16`;
- optional residual style code only after grammar/band are proven active.

This phase asks whether the learned coordinate system has enough capacity. If
the vocabulary refit only makes outputs more content-safe or collapses style,
the bottleneck is not the token values; it is the actuator that consumes them.

### Phase C: Backbone Revision

Use the vocabulary readout to revise the model.

For the current Hayao failure, the likely missing actuator is not another global
style vector. It is a macro image operation:

```text
semantic flat-color repaint + edge-locked contour reinforcement
```

If vocabulary-only training learns large Hayao flatness values but the image
does not become more Hayao-like, the backbone lacks the correct flat/contour
operator. If the vocabulary never learns Hayao-specific fields, the training
loss does not expose Hayao's visual grammar strongly enough.

## Acceptance Gates

A tokenizer phase is successful only if it improves the real transfer problem:

- global `clip_style > 0.72`;
- `content_lpips <= 0.50` preferred;
- Hayao cross-target `clip_style` rises materially;
- images show clean animation-like planes and contours, not just metric noise;
- the by-style debug readout shows a different learned grammar for Hayao than
  for texture-heavy styles.

## Current Evidence

The first factorized-tokenizer runs did not satisfy these gates:

| run | best global style | LPIPS | Hayao cross style | interpretation |
|---|---:|---:|---:|---|
| `ema_style_vocab_texton_w34` | 0.7084 | 0.5144 | 0.6429 | weak/generic fields |
| `ema_style_vocab_hayao_w36` | 0.7069 | 0.5445 | 0.6531 | Hayao pressure hurts LPIPS |

Those runs are now treated as contaminated diagnostics because they used manual
style priors. The next tokenizer backbone must be rerun from neutral fields.

## New Clean Variants

Added variants:

| variant | purpose |
|---|---|
| `ema_style_vocab_neutral_w34` | clean backbone readout with no style-specific vocabulary priors |
| `ema_style_vocab_neutral_w36_stylepush` | balanced style-pressure readout with the same neutral vocabulary initialization |

The second variant must not emphasize Hayao through sampling or loss weights.
Hayao remains a diagnostic slice, not a training shortcut. If the neutral
tokenizer cannot learn a Hayao-specific field state under balanced exposure,
the conclusion is a missing field/operator or objective signal, not that the
run needs manual Hayao weighting.
