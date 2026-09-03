# Operator Tokenizer Code Reset

Date: 2026-05-28

This note records the codebase reset that removes the anonymous style-vector
path from the active `src` implementation. No experiment was launched for this
change; it is a structural cleanup and documentation step.

## Decision

The active model path now treats style as named operator fields:

- `identity`: pointwise/channel operator input for zero-order color drift.
- `grammar`: depthwise/spatial operator input for local geometry.
- `band_gains`: direct low/mid/high frequency energy gates.

The tokenizer returns `StyleTokenFields` only. It no longer projects fields
back into a single mixed vector, and active runtime code no longer accepts an
override vector for style control.

## Removed From Active Source

- Anonymous dynamic convolution head driven by a single style vector.
- Legacy style-conditioned highpass, lowpass, midband, decoder-feature heads.
- Style-conditioned high-resolution feature blocks and skip-router modulation.
- Style embedding adapter load/copy paths in training and inference.
- Obsolete config keys for the removed vector path:
  - `style_dim`
  - `style_tokenizer_enable`
  - `style_token_code_residual_scale`
  - `style_token_project_code`
  - `style_token_zero_init_projection`
  - `dynamic_style_operator_mode`
  - `dynamic_style_operator_hidden_mult`
  - legacy style-code highpass/lowpass/midband/decoder-feature operator keys
  - style-code texton allocation keys

Old experiment configs may still contain these fields; `ModelConfig.extra`
keeps historical configs loadable, but the active model does not read them.

## Current Active Contract

`StyleTokenizer.forward(style_id, batch_size, device, dtype)` returns:

- `style_id`
- `identity`
- `grammar`
- `band_logits`
- `band_gains`

`FactorizedDynamicOperatorHead` is the only dynamic output operator:

- `grammar -> 3x3 depthwise spatial kernels`
- `identity -> 1x1 pointwise channel kernels and bias`
- `band_gains -> direct residual band scaling`

`StyleBlender` may still use token-readable paths such as token reader,
grammar texture allocation, token carrier, prototype carrier, flattening, and
depthwise filter gates. These paths consume named fields, not an anonymous
style vector.

## Consequence

This reset intentionally breaks old adapter-only workflows that updated only
the former style embedding. Future tokenizer work must optimize one of the
remaining named surfaces:

- tokenizer vocabularies,
- style spatial priors,
- style memory bank,
- token-bound blender/operator gates,
- factorized dynamic output/feature operator.

The next valid development step is code/design refinement or a documented
smoke check. It should not be a blind scalar sweep.
