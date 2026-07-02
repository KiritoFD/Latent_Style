# tok_pure_latent_spatial Plan

- Wave: `wave1_tokenizer`
- Axis: `tokenizer`
- Notes: Wave-1 proposed tokenizer: latent-native spatial routing from z0 with deterministic ODE transport.
- DINO policy: archived-only unless overwhelming gain appears.
- Validation:
  - local trainer init passes
  - legacy compatibility tokenizer trainable count is `0`
  - active trainable tokenizer parameters live under `structured_style_tokenizer.*`
